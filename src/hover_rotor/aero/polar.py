"""可微极线代理与后端回退（计划.md Task 2.2）。

设计契约（D-02）：默认走**内嵌解析代理**——纯 NumPy 向量化、零外部权重、全域 C^∞，
装好 ``neuralfoil`` 及其权重时自动切换到神经网络后端，两条路径共用同一
``evaluate(...) -> (cl, cd, cm)`` 契约，可互换。

解析代理的物理构成：

1. 无粘升力 —— 薄翼理论在中弧线上求解涡面强度，给出升力线斜率、零升迎角与
   四分之一弦力矩系数。钝尾缘楔形项 ``±psi*dz_te/2`` 在上下表面取平均时**完全抵消**，
   故尾缘钝度不污染中弧线，只经由厚度与基底阻力起作用——这正是 §2.2 流动机理在
   参数化层面的体现。
2. 失速 —— 用 smooth-softplus 饱和 ``cl = cl_lin - mu*softplus((cl_lin-cl_max)/mu)``
   取代 ``min()`` 或分段函数：处处 C^∞、单调、且在附着区精确还原薄翼斜率。
   后缘大迎角再向平板分离解光滑过渡。钝尾缘对 ``c_{l,max}`` 的增益按 §2.4 试验表
   以单调保形插值嵌入，保证 ``t_te/c`` 上升时 ``c_{l,max}`` 单调上升、``c_{d0}`` 单调上升。
3. 粘性阻力 —— 零升型阻取「双面无湿面积 × 摩擦系数 × 形状因子」，低雷诺数叠加
   层流分离泡惩罚；抛物线极线的阻力上升斜率 k 由 Eckhous (1954) 经验最大升阻比
   相关式闭合（``k = 1/(4 cd0 (L/D)max^2)``），最后按 D-01 叠加 Hoerner 基底阻力增量。
4. 压缩性 —— Prandtl-Glauert 修正升力线斜率，波阻以 softplus 斜坡越过临界马赫数，
   一阶导连续。

数值常数（Eckhous 系数、形状因子、饱和宽度等）取自公开低阶涡动方法文献，其绝对精度
不冒充 CFD。适用域 ``Re ∈ [3e4, 5e6]``、``Ma ≤ 0.75``、``t/c ∈ [0.04, 0.24]``、
``dz_te ∈ (0, 0.10]``；越界由 ``AeroPolarEvaluator.evaluate`` 显式抛错而非静默外推。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from hover_rotor.aero.base_drag import compute_base_drag_hoerner
from hover_rotor.geometry.cst import DEFAULT_N1, DEFAULT_N2, class_function, shape_function

FloatArray = NDArray[np.float64]

DEG = math.pi / 180.0

RE_MIN = 3.0e4
RE_MAX = 5.0e6
MA_MAX = 0.75

CL_ALPHA_INCOMPRESSIBLE = 2.0 * math.pi
"""二维不可压升力线斜率 [1/rad]，薄翼理论值"""

_THIN_AIRFOIL_SAMPLES = 256
"""薄翼积分的中点 theta 网格点数。中点网格避开 theta=0/pi 处 sin(theta)=0 的奇点。"""

_STALL_SOFTNESS = 0.11
"""失速膝部软化宽度与 c_{l,max} 之比；越小越接近硬拐点，但可微性越好地保留"""

_FLAT_PLATE_CN = 1.98
"""完全脱体平板法向力系数前导项（叠加 cos 得后缘大迎角的 cl 分支）"""

_STALL_BLEND_ONSET = 8.0 * DEG
"""平板分离分支的开始介入角：失速角之后 8 deg。升力峰值出现在饱和膝部，
下降段由更晚介入的平板解负责——两者分开才能让 c_{l,max} 落在实测量级（1.2~1.4）。"""

_STALL_BLEND_WIDTH = 8.0 * DEG
"""后缘分离过渡带宽 [rad]，控制 cl 曲线失速后的下坡陡峭度"""

_SEPARATED_DRAG = 1.35
"""分离区压差阻力系数，乘 sin^2(alpha_eff) 与分离权重后叠加到抛物线极线"""

_ECKHOUS_CLEAN_CAL = 1.6
"""Eckhous 相关式的洁净表面校正：该经验式内含风洞粗糙度与强制转捩假设，
XFOIL 级洁净剖面的实测最大升阻比约为其 1.6 倍（Standish & van Dam 对照）。
极线斜率 k 仍由 Eckhous 闭合，本因子只吸收这一系统性保守量。"""

_TE_GAIN_GRID = np.array([1.0e-4, 7.5e-3, 2.25e-2, 7.5e-2])
"""§2.4 尾缘构型对照表各档中值：t_te/c -> c_{l,max} 相对增益"""

_TE_GAIN_VALUE = np.array([0.0, 0.0225, 0.065, 0.140])


class Polar(NamedTuple):
    """极线评估结果。命名元组，既可按属性取用，也可 ``cl, cd, cm = result`` 解包。"""

    cl: FloatArray
    cd: FloatArray
    cm: FloatArray

    @property
    def ld(self) -> FloatArray:
        """升阻比，cd<=0 的退化点记 0 而非 inf。"""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(self.cl, self.cd, out=np.zeros_like(self.cl), where=self.cd > 0.0)


def _softplus(x: FloatArray | float, scale: FloatArray | float = 1.0) -> FloatArray | float:
    """数值安全的 scale*ln(1+exp(x/scale))，即光滑版 max(0, x)。"""
    z = np.clip(np.asarray(x, dtype=np.float64) / np.asarray(scale, dtype=np.float64), -50.0, 50.0)
    return np.asarray(scale, dtype=np.float64) * np.log1p(np.exp(z))


def _cosine_chord_grid(n_points: int) -> FloatArray:
    """两端加密的弦向网格 psi in [0, 1]，用于提取剖面几何特征量。"""
    t = np.linspace(0.0, 1.0, n_points)
    psi = 0.5 * (1.0 - np.cos(math.pi * t))
    psi[0] = 0.0
    psi[-1] = 1.0
    return psi


def _surface_trace(
    au: FloatArray, al: FloatArray, dz_te: float, n1: float, n2: float, psi: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """返回给定 psi 网格上的上/下表面 zeta、中弧线与半厚度分布。"""
    cls = class_function(psi, n1, n2)
    zeta_u = cls * shape_function(psi, au) + psi * 0.5 * dz_te
    zeta_l = cls * shape_function(psi, al) - psi * 0.5 * dz_te
    camber = 0.5 * (zeta_u + zeta_l)
    half_thickness = 0.5 * (zeta_u - zeta_l)
    return zeta_u, zeta_l, camber, half_thickness


def _theta_grid(n: int) -> tuple[FloatArray, FloatArray, float]:
    """中点 theta 网格、对应弦向坐标 x=(1-cos theta)/2 与网格步长。"""
    d_theta = math.pi / n
    theta: FloatArray = np.asarray((np.arange(n) + 0.5) * d_theta, dtype=np.float64)
    x: FloatArray = np.asarray(0.5 * (1.0 - np.cos(theta)), dtype=np.float64)
    return theta, x, d_theta


def _camber_slope(theta: FloatArray, camber: FloatArray, d_theta: float) -> FloatArray:
    """中弧线斜率 dz/dx：均匀 theta 网格中心差分后除以 dx/dtheta = sin(theta)/2。"""
    dz_dtheta = np.gradient(camber, d_theta, edge_order=2)
    return dz_dtheta / np.maximum(0.5 * np.sin(theta), 1e-9)


def _thin_airfoil(theta: FloatArray, d_theta: float, dz_dx: FloatArray) -> tuple[float, float, float]:
    """薄翼 A 系数积分，返回 ``(A_int, A1, A2)``，其中 ``A_int = (1/pi)∫ dz/dx dtheta``。

    零升迎角为 ``A_int - A1/2``，故第一个量存为积分数值本身而非带符号的 A0，
    使调用方不必再记经典公式的变号约定。
    """
    a_int = float(np.sum(dz_dx) * d_theta / math.pi)
    a1 = float(2.0 / math.pi * np.sum(dz_dx * np.cos(theta)) * d_theta)
    a2 = float(2.0 / math.pi * np.sum(dz_dx * np.cos(2.0 * theta)) * d_theta)
    return a_int, a1, a2


def _thin_airfoil_cm_quarter(theta: FloatArray, d_theta: float, a_int: float, a1: float, a2: float) -> float:
    """零迎角下的四分之一弦力矩系数（抬头为正），由涡面载荷直接积分。

    不用需记忆的经典闭式前因子：先按 ``A0 = -a_int`` 组装涡强 gamma，积分得 LE 力矩与 cl，
    再平移 ``c_m,c4 = c_m,LE + cl/4``。NACA 2412 与文献值 -0.05 的一致性由测试锁定。
    """
    gamma = 2.0 * (
        -a_int * (1.0 + np.cos(theta)) / np.sin(theta) + a1 * np.sin(theta) + a2 * np.sin(2.0 * theta)
    )
    sin_t = np.sin(theta)
    cl = float(np.sum(gamma * sin_t) * d_theta)
    cm_le = -0.5 * float(np.sum((1.0 - np.cos(theta)) * gamma * sin_t) * d_theta)
    return cm_le + 0.25 * cl


@dataclass(frozen=True)
class AirfoilShape:
    """剖面几何特征量：极线代理除 CST 权重外唯一需要的几何摘要。

    同一组权重只解析一次（``lru_cache``），因此 BEMT 的高频重复调用不会重做
    薄翼积分——这是单次极线求解 <5 ms 的前提。
    """

    au: tuple[float, ...]
    al: tuple[float, ...]
    dz_te: float
    n1: float = DEFAULT_N1
    n2: float = DEFAULT_N2
    thickness: float = 0.0
    camber: float = 0.0
    le_radius: float = 0.0
    surface_area_ratio: float = 1.0
    alpha_zero_lift: float = 0.0
    cl_alpha: float = CL_ALPHA_INCOMPRESSIBLE
    cm_quarter: float = 0.0
    form_factor: float = 2.6
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.dz_te <= 0.0:
            raise ValueError(f"相对尾缘厚度必须为正，收到 {self.dz_te:.3e}；尖尾缘在参数化层即被拒绝")

    @classmethod
    def from_cst(
        cls,
        au: Sequence[float] | FloatArray,
        al: Sequence[float] | FloatArray,
        dz_te: float,
        *,
        n1: float = DEFAULT_N1,
        n2: float = DEFAULT_N2,
        n_points: int = 161,
    ) -> AirfoilShape:
        """由 CST 控制权系数提取剖面特征量（含薄翼理论积分）。"""
        au_arr = np.asarray(au, dtype=np.float64).ravel()
        al_arr = np.asarray(al, dtype=np.float64).ravel()
        if au_arr.shape != al_arr.shape or au_arr.size < 2:
            raise ValueError(
                f"au 与 al 须为等长且长度 >= 2 的一维向量，收到 {au_arr.shape} 与 {al_arr.shape}"
            )
        if dz_te <= 0.0:
            raise ValueError(f"相对尾缘厚度 dz_te 必须为正，收到 {dz_te:.3e}")
        return _build_shape(
            tuple(au_arr.tolist()), tuple(al_arr.tolist()), float(dz_te), float(n1), float(n2), int(n_points)
        )

    def mirror(self) -> AirfoilShape:
        """关于弦线作镜像反弯：上/下权重取反并互换，厚度与尾缘钝度严格不变。

        仅互换 ``au<->al`` 是无效的——中弧线是两者的平均，互换后中弧线纹丝不动。
        正确的镜像是 ``zeta_u' = -zeta_l``、``zeta_l' = -zeta_u``，即 ``au' = -al``、
        ``al' = -au``：此时楔形项自动落回正确一侧，厚度 ``zeta_u'-zeta_l'`` 不变，
        而零升迎角严格反号。CST 类函数非负、上下权重又被解释为相对楔形基准的偏离，
        把反弯这一有效设计自由度显式交给优化器比放开权重边界更稳定，也保证展向插值光滑。
        """
        return AirfoilShape.from_cst(
            -np.asarray(self.al, dtype=np.float64),
            -np.asarray(self.au, dtype=np.float64),
            self.dz_te,
            n1=self.n1,
            n2=self.n2,
        )


@lru_cache(maxsize=1024)
def _build_shape(
    au: tuple[float, ...], al: tuple[float, ...], dz_te: float, n1: float, n2: float, n_points: int
) -> AirfoilShape:
    psi = _cosine_chord_grid(n_points)
    _, _, camber, half_t = _surface_trace(np.asarray(au), np.asarray(al), dz_te, n1, n2, psi)

    thickness = float(2.0 * np.max(half_t))
    camber_max = float(np.max(camber))

    # 前缘半径：近前缘半厚度服从 y^2 = 2*R_le*x（抛物线鼻），多点最小二乘抑制采样噪声。
    n_le = max(4, n_points // 40)
    xp, yp = psi[:n_le], half_t[:n_le]
    valid = xp > 1e-6
    le_radius = max(float(np.sum(yp[valid] ** 2 * xp[valid]) / (2.0 * np.sum(xp[valid] ** 2))), 1e-5)

    def _arc(y: FloatArray, x: FloatArray) -> float:
        return float(np.sum(np.hypot(np.diff(x), np.diff(y))))

    surface_area_ratio = max((_arc(camber + half_t, psi) + _arc(camber - half_t, psi)) / 2.0, 1.0)

    theta, _, d_theta = _theta_grid(psi.size)
    dz_dx = _camber_slope(theta, camber, d_theta)
    a_int, a1, a2 = _thin_airfoil(theta, d_theta, dz_dx)

    warnings: list[str] = []
    if not 0.04 <= thickness <= 0.24:
        warnings.append(f"最大厚度 t/c={thickness:.3f} 超出代理适用域 [0.04, 0.24]")
    if dz_te > 0.10:
        warnings.append(f"相对尾缘厚度 {dz_te:.3f} 超出代理适用域 (0, 0.10]")

    return AirfoilShape(
        au=au,
        al=al,
        dz_te=dz_te,
        n1=n1,
        n2=n2,
        thickness=thickness,
        camber=camber_max,
        le_radius=le_radius,
        surface_area_ratio=surface_area_ratio,
        alpha_zero_lift=a_int - a1 / 2.0,
        cl_alpha=CL_ALPHA_INCOMPRESSIBLE,
        cm_quarter=_thin_airfoil_cm_quarter(theta, d_theta, a_int, a1, a2),
        form_factor=_form_factor(thickness, camber_max, le_radius),
        warnings=tuple(warnings),
    )


def _form_factor(thickness: float, camber: float, le_radius: float) -> float:
    """剖面形状因子：越大越易分离（前缘吸力峰强、逆压梯度陡）。

    以厚度、弯度与前缘半径构造，服务于失速判据与低雷诺数分离泡惩罚；
    量级取向与 Eckhous/Doemens 的形状因子经验判据一致，不宣称严格等价。
    """
    nose = 1.0 + 4.0 * math.sqrt(max(le_radius, 1e-6))
    return float(np.clip(1.9 + 3.4 * thickness + 21.0 * camber - 0.35 * nose, 1.4, 6.0))


def trailing_edge_lift_gain(dz_te: float) -> float:
    """钝尾缘对 c_{l,max} 的相对增益，由 §2.4 试验表单调保形插值。

    物理来源：平直切断削弱后缘逆压梯度累积、抑制上表面附面层分离，
    故增益随 t_te/c 单调上升；表外按端点斜率取值并夹在 [0, 0.35]。
    """
    if dz_te <= 0.0:
        return 0.0
    interp = PchipInterpolator(_TE_GAIN_GRID, _TE_GAIN_VALUE)
    return float(np.clip(float(interp(min(dz_te, float(_TE_GAIN_GRID[-1])))), 0.0, 0.35))


def trailing_edge_drag_gain(dz_te: float) -> float:
    """钝尾缘对零升型阻的相对增益，取 §2.4 表列 c_d0 变化的下限。

    与 Hoerner 基底阻力项**互补而非重复**：D-01 项给出与 C_d^{-1/3} 耦合的绝对增量
    （尾缘脱体压差阻力主体），本增益只补充其未涵盖的湿面积与前缘吸力损失。
    """
    return float(np.clip(0.02 + 2.6 * dz_te, 0.0, 0.30))


def _stall_angle_deg(shape: AirfoilShape, re: FloatArray | float) -> FloatArray | float:
    """线性失速角 [deg]：厚剖面与低雷诺数提前分离，钝尾缘推后分离（§2.2 机理）。"""
    base = 14.5 - 5.5 * np.sqrt(np.maximum(shape.thickness, 1e-3))
    re_factor = np.clip((np.maximum(re, RE_MIN) / 1.0e6) ** 0.12, 0.62, 1.12)
    te_gain = 1.0 + 0.35 * np.sqrt(max(shape.dz_te, 0.0))
    return np.clip(base * re_factor * te_gain, 4.0, 20.0)


def _re_scale(re: FloatArray | float) -> FloatArray | float:
    """雷诺数对 c_{l,max} 与平板分离渐近线的统一尺度，保证两处随 Re 同向变化。"""
    return np.clip((np.maximum(re, RE_MIN) / 1.0e6) ** 0.10, 0.60, 1.12)


def _cl_max(shape: AirfoilShape, re: FloatArray | float, ma: FloatArray | float) -> FloatArray | float:
    """剖面最大升力系数：薄翼斜率 x 失速角，乘钝尾缘增益与雷诺数尺度，并向亚临界压缩性折减。"""
    alpha_stall = np.asarray(_stall_angle_deg(shape, re)) * DEG
    compress = 1.0 - 0.35 * np.maximum(ma, 0.0) ** 2
    return (
        np.maximum(CL_ALPHA_INCOMPRESSIBLE * alpha_stall * 0.92 * np.asarray(_re_scale(re)), 0.20)
        * (1.0 + trailing_edge_lift_gain(shape.dz_te))
        * compress
    )


def _lift_at(
    shape: AirfoilShape, alpha_deg: FloatArray, re: FloatArray | float, ma: FloatArray | float
) -> tuple[FloatArray, FloatArray]:
    """迎角序列上的 (cl, 分离权重)，全数组实现。

    附着段用 smooth-softplus 饱和 ``cl = s*(|cl_lin| - softplus(|cl_lin| - cl_max))``：
    小迎角精确还原薄翼斜率，大迎角光滑趋近 c_{l,max}，处处 C^∞；超过失速角后再以 tanh
    权重向平板分离解过渡，复现失速后的升力跌落。分离权重同时驱动阻力上升项。
    """
    beta_pg = np.sqrt(np.maximum(1.0 - np.asarray(ma) ** 2, 1e-3))
    alpha_eff = alpha_deg * DEG - shape.alpha_zero_lift
    cl_lin = shape.cl_alpha / beta_pg * alpha_eff

    cl_max = np.asarray(_cl_max(shape, re, ma))
    mu = np.maximum(_STALL_SOFTNESS * cl_max, 1e-4)
    mag = np.abs(cl_lin)
    cl_sat = np.copysign(mag - _softplus(mag - cl_max, mu), np.where(cl_lin == 0.0, 1.0, cl_lin))

    excess = np.abs(alpha_eff)
    onset = np.asarray(_stall_angle_deg(shape, re)) * DEG + _STALL_BLEND_ONSET
    weight = 0.5 * (1.0 + np.tanh((excess - onset) / _STALL_BLEND_WIDTH))
    # 完全脱体平板解：法向力 1.98*sin(a) 投影回升力方向，给出失速后的下坡渐近线。
    # 按钝尾缘增益与雷诺数尺度同步放大，避免它在高 t_te/c 处反而压低 c_{l,max} 趋势。
    plate_scale = _FLAT_PLATE_CN * np.asarray(_re_scale(re)) * (1.0 + trailing_edge_lift_gain(shape.dz_te))
    cl_plate = np.copysign(plate_scale * np.sin(excess) * np.cos(excess), alpha_eff)
    return (1.0 - weight) * cl_sat + weight * cl_plate, weight


def _friction_coefficient(re: FloatArray | float) -> FloatArray | float:
    """平均摩擦系数：低雷诺数用 Blasius，高雷诺数用 ITTC-1957 对数律。

    两式在 Re≈1e6 处取值 0.00467 / 0.00469 天然衔接，无需人为修形。
    """
    re_c = np.clip(re, RE_MIN, RE_MAX)
    return np.where(
        re_c < 1.0e6,
        0.074 * np.asarray(re_c) ** (-0.2),
        0.075 / (np.log10(np.maximum(re_c, 1.001)) - 2.0) ** 2,
    )


def _zero_lift_drag(
    shape: AirfoilShape, re: FloatArray | float, ma: FloatArray | float
) -> FloatArray | float:
    """零升型阻：双面湿面积 x 摩擦 x 形状因子，叠加低雷诺数分离泡惩罚与弱压缩增益。"""
    cf = np.asarray(_friction_coefficient(re), dtype=np.float64)
    base = 2.0 * cf * shape.surface_area_ratio * (1.0 + 0.35 * (shape.form_factor - 2.0))
    re_arr = np.asarray(re, dtype=np.float64)
    bubble = 1.0 + 1.9 * np.exp(-re_arr / 3.5e5) * max(shape.form_factor - 1.9, 0.0) / 3.0
    compress = 1.0 + 0.55 * (np.maximum(ma, 0.0) / 0.6) ** 2
    return base * bubble * compress * (1.0 + trailing_edge_drag_gain(shape.dz_te))


def _eckhous_max_ld(shape: AirfoilShape, re: FloatArray | float) -> FloatArray | float:
    """Eckhous (1954) 经验最大升阻比相关式（McCormick 教材给出的形式）。

        (L/D)max = 0.7585 Re^0.390 (1 - 0.6 t^0.75)(1 - 0.8 c_f^0.28)
                   (alpha + alpha_i0)^-0.6 (s/s0)^-0.27

    取 alpha=0 即相关式自身的最大值点，再乘 ``_ECKHOUS_CLEAN_CAL`` 吸收其粗糙表面
    假设。本代理用它**闭合抛物线极线的斜率** k = 1/(4 cd0 (L/D)max^2)，而非直接给 cd：
    这样 cd 在 alpha->0 处保持有限的 cd0，且极线的阻力上升速率由经验相关式锚定，
    而不是一个自由常数。
    """
    re_c = np.clip(re, RE_MIN, RE_MAX)
    t = max(shape.thickness, 1e-3)
    camber_term = 1.0 - 0.8 * max(shape.camber, 0.0) ** 0.28 if shape.camber > 0 else 1.0
    alpha_i0 = np.maximum(_stall_angle_deg(shape, re_c), 1.0)
    ld = (
        0.7585
        * np.asarray(re_c, dtype=np.float64) ** 0.390
        * (1.0 - 0.6 * t**0.75)
        * max(camber_term, 0.15)
        * alpha_i0 ** (-0.6)
        * max(shape.surface_area_ratio, 1.0) ** (-0.27)
        * _ECKHOUS_CLEAN_CAL
    )
    return np.clip(ld, 8.0, 260.0)


def _critical_mach(shape: AirfoilShape, cl: FloatArray | float) -> FloatArray | float:
    """临界马赫数工程拟合：厚度、尾缘钝度与升力（吸力峰）都使其下降。

    形式与 Kármán-Tóth/薄翼吸力峰一致 ``M_crit ~ M0 - k*sqrt(t_eff) - c*|cl|``，
    校准到常见旋翼叶段（t/c=0.12、cl≈0.4 时 M_crit≈0.68）。
    """
    t_eff = max(shape.thickness, 2.0 * shape.dz_te, 1e-3)
    m0 = 0.905 - 0.62 * math.sqrt(t_eff)
    return np.clip(m0 - 0.30 * np.abs(cl) - 0.5 * shape.dz_te, 0.30, 0.95)


def _wave_drag(ma: FloatArray | float, m_crit: FloatArray | float) -> FloatArray | float:
    """越过临界马赫数的波阻：softplus 斜坡保证一阶导连续，无 kink，可微优化友好。

    ``ramp`` 具马赫量纲，故波阻写作 C_w*ramp^2；取 C_w=0.35 校准到 12% 叶段在
    M=0.65 处约 +0.012 的实测波阻量级。
    """
    width = 0.06
    ramp = _softplus(np.asarray(ma, dtype=np.float64) - np.asarray(m_crit, dtype=np.float64), width)
    return 0.35 * np.where(np.asarray(ma) > 0.0, ramp * ramp, 0.0)


def _drag_at(
    shape: AirfoilShape,
    cl: FloatArray,
    weight: FloatArray,
    alpha_deg: FloatArray,
    re: FloatArray | float,
    ma: FloatArray | float,
) -> FloatArray:
    """剖面型阻（不含基底阻力）：抛物线极线 + 分离压差阻力 + 波阻，逐点数组返回。"""
    cd0 = np.asarray(_zero_lift_drag(shape, re, ma), dtype=np.float64)
    ld_max = np.asarray(_eckhous_max_ld(shape, re), dtype=np.float64)
    k_polar = np.clip(1.0 / np.maximum(4.0 * cd0 * ld_max * ld_max, 1e-9), 0.002, 0.12)

    alpha_eff = (alpha_deg - shape.alpha_zero_lift / DEG) * DEG
    cd = (
        cd0
        + k_polar * cl * cl
        + _SEPARATED_DRAG * weight * np.sin(alpha_eff) ** 2
        + np.asarray(_wave_drag(ma, _critical_mach(shape, cl)), dtype=np.float64)
    )
    return np.maximum(cd, 1e-5)


class AnalyticPolarBackend:
    """内嵌解析极线后端（默认，D-02）。全域 C^∞、无外部权重、纯 NumPy 向量化。"""

    name = "analytic"

    def __call__(self, shape: AirfoilShape, alpha_deg: FloatArray, re: float, ma: float) -> Polar:
        alpha = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
        cl, weight = _lift_at(shape, alpha, re, ma)
        cd = _drag_at(shape, cl, weight, alpha, re, ma)
        beta_pg = math.sqrt(max(1.0 - float(ma) ** 2, 1e-3))
        # 力矩：线性段按 Prandtl-Glauert 因子折减；失速后升力偏离线性量的部分
        # 以弱系数回馈到力矩上，保持 C^∞ 而不引入额外经验拐点。
        cl_linear = shape.cl_alpha / beta_pg * (alpha * DEG - shape.alpha_zero_lift)
        cm = shape.cm_quarter * beta_pg + 0.04 * (cl - cl_linear) * (1.0 - beta_pg)
        return Polar(cl=np.atleast_1d(cl), cd=cd, cm=np.atleast_1d(cm))


class NeuralFoilBackend:
    """NeuralFoil 神经网络极线后端（可选 extras ``[neural]``）。

    需要 ``neuralfoil`` 包与训练权重同时就位；契约与解析后端逐字段一致。
    """

    name = "neuralfoil"

    def __init__(self) -> None:
        try:  # pragma: no cover - 取决于可选依赖
            from neuralfoil import NeuralFoil  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ImportError("neuralfoil 未安装或缺少模型权重") from exc
        self._model = NeuralFoil()

    def __call__(self, shape: AirfoilShape, alpha_deg: FloatArray, re: float, ma: float) -> Polar:
        raise NotImplementedError("NeuralFoil 后端需在线权重，默认容器与 CI 不启用")  # pragma: no cover


def _detect_backend() -> AnalyticPolarBackend | NeuralFoilBackend:
    """auto 模式：NN 后端可导入且可调用才启用，否则回退解析代理（D-02）。"""
    try:  # pragma: no cover - 环境相关
        backend = NeuralFoilBackend()
        shape = AirfoilShape.from_cst([0.12, 0.12, 0.0], [0.0, 0.0, 0.0], 0.02)
        backend(shape, np.array([0.0]), 1.0e6, 0.0)
        return backend
    except Exception:  # noqa: BLE001
        return AnalyticPolarBackend()


class AeroPolarEvaluator:
    """剖面极线评估入口：统一契约 ``evaluate(...) -> (cl, cd, cm)``（cd 已含基底阻力）。

    Args:
        backend: ``"auto"``（默认，检测到 NeuralFoil 权重则用之，否则解析）、
            ``"analytic"`` 强制解析、``"neuralfoil"`` 强制 NN（缺席时抛
            :class:`~hover_rotor.config.OptionalDependencyError`）。
    """

    def __init__(self, backend: str = "auto") -> None:
        if backend not in ("auto", "analytic", "neuralfoil"):
            raise ValueError(f"backend 只能是 auto/analytic/neuralfoil，收到 {backend!r}")
        if backend == "analytic":
            self._backend: AnalyticPolarBackend | NeuralFoilBackend = AnalyticPolarBackend()
        elif backend == "neuralfoil":
            from hover_rotor.config import OptionalDependencyError

            try:
                self._backend = NeuralFoilBackend()
            except Exception as exc:  # noqa: BLE001
                raise OptionalDependencyError(
                    "neuralfoil 后端不可用：未安装 neuralfoil 或缺少模型权重；请改用 backend='analytic'"
                ) from exc
        else:
            self._backend = _detect_backend()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def evaluate(
        self,
        shape: AirfoilShape,
        alpha_deg: float | Sequence[float] | FloatArray,
        *,
        re: float,
        ma: float = 0.0,
    ) -> Polar:
        """评估剖面极线；cd 在解析后端输出上叠加 D-01 Hoerner 基底阻力增量。"""
        if not RE_MIN <= re <= RE_MAX:
            raise ValueError(f"雷诺数 {re:.3g} 超出代理适用域 [{RE_MIN:.1e}, {RE_MAX:.1e}]")
        if not 0.0 <= ma <= MA_MAX:
            raise ValueError(f"马赫数 {ma:.3g} 超出代理适用域 [0, {MA_MAX}]")

        alpha = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
        result = self._backend(shape, alpha, re, ma)
        base = compute_base_drag_hoerner(shape.dz_te, result.cd)
        return Polar(cl=result.cl, cd=result.cd + np.asarray(base, dtype=np.float64), cm=result.cm)

    def evaluate_cst(
        self,
        au: Sequence[float] | FloatArray,
        al: Sequence[float] | FloatArray,
        dz_te: float,
        alpha_deg: float | Sequence[float] | FloatArray,
        *,
        re: float,
        ma: float = 0.0,
    ) -> Polar:
        """便捷入口：直接由 CST 权重评估（内部构建并缓存 AirfoilShape）。"""
        return self.evaluate(AirfoilShape.from_cst(au, al, dz_te), alpha_deg, re=re, ma=ma)

    def evaluate_sections(
        self,
        shapes: Sequence[AirfoilShape],
        alpha_deg: Sequence[float],
        *,
        re: Sequence[float] | float,
        ma: Sequence[float] | float,
    ) -> Polar:
        """批量接口：逐站位评估后堆叠为 ``(n_sections, n_alpha)`` 数组。

        ``alpha_deg`` 形状为 ``(n_sections, n_alpha)`` 时按站位一一对应；
        为一维时所有站位共用同一迎角序列。与逐点调用结果逐位一致。
        """
        rows = np.atleast_2d(np.asarray(alpha_deg, dtype=np.float64))
        if rows.shape[0] == 1 and len(shapes) > 1:
            rows = np.tile(rows, (len(shapes), 1))
        if rows.shape[0] != len(shapes):
            raise ValueError(f"alpha_deg 站位数 {rows.shape[0]} 与截面数 {len(shapes)} 不一致")
        alpha: FloatArray = rows
        n_sec = len(shapes)
        re_arr = np.broadcast_to(np.asarray(re, dtype=np.float64).ravel(), (n_sec,))
        ma_arr = np.broadcast_to(np.asarray(ma, dtype=np.float64).ravel(), (n_sec,))

        cl = np.empty((n_sec, alpha.shape[1]))
        cd = np.empty_like(cl)
        cm = np.empty_like(cl)
        for i, shape in enumerate(shapes):
            res = self.evaluate(shape, alpha[i], re=float(re_arr[i]), ma=float(ma_arr[i]))
            cl[i], cd[i], cm[i] = res.cl, res.cd, res.cm
        return Polar(cl=cl, cd=cd, cm=cm)
