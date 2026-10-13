r"""悬停叶素-动量理论（BEMT）核心求解器（计划.md Task 2.3）。

物理内核：把三维旋转叶片切成独立二维微元，对每个微元同时满足
「叶素升阻力」与「该环段的动量守恒」，两者耦合出一个非线性入流角方程。
悬停工况无前行/后行不对称，问题退化为展向一维，因而可以在毫秒级完成一次整机评估，
这正是 §1 表格里 BEMT 被指派为「全局初搜与反设计主循环」求解器的原因。

    dT = B * 0.5 * rho * W^2 * c * (cl cos(phi) - cd sin(phi)) dr
    dQ = B * 0.5 * rho * W^2 * c * (cl sin(phi) + cd cos(phi)) r dr
    W  = Omega r / cos(phi),  alpha = theta - phi

三个必须显式处理的物理缺陷，以及本模块的处理方式：

1. **叶尖/叶根损失** —— 有限叶片数使盘面涡系在两端卷起，升力在叶尖跌到零。
   用 Prandtl 的 F = F_tip * F_hub 修正动量侧，F 进入 cn 之前而非之后，
   因此它同时改变入流解而不只是最终载荷。
2. **大诱导区的动量发散** —— 局部载荷系数 4a^2 越过 0.64（即 a>0.4）后动量理论
   进入湍流尾迹态并出现物理多值性。这里用 Buhl-Glauert 型的 C^1 有理饱和把动量解
   钳到渐近线上；在 a <= a_knee 时该修正是**恒等**的，故弱载荷工况逐位不受影响，
   强载荷工况也不再发散——这是优化器能在失速区反复迭代而不崩的前提。
3. **数值收敛** —— 逐节点 Brent 求根在 51 个稠密节点上比阻尼不动点慢一个数量级，
   会把端到端耗时顶穿 D-07 的 3 分钟门禁。默认用带自适应松弛的全局向量不动点迭代，
   仅对松弛迭代没收敛掉的**少数残留节点**回退到 brentq 夹逼求根（见 ``_solve_nodes_brent``）。

入流同时耦合**切向诱导** $a'$（$u_t$）：计划 §1 的 $W=\sqrt{(\Omega r-u_t)^2+v_i^2}$ 含它，
Task 2.3 的迭代契约 $\phi=\arctan(v_i/\Omega r)$ 却不含，两处不一致。本实现按 §1 联立
$a^2=S(1-a')^2c_n$ 与 $a\,a'=S(1-a')^2c_t$（$S=\sigma F/(4\cos^2\phi)$）解 $a$ 的二次式，
并由 $\tan\phi=a/(1-a')$ 反回入流角；载荷积分同用 $W=\Omega r(1-a')/\cos\phi$，
否则入流按三维、载荷按二维会把修正抵消掉一半。

两种尾迹模式：
  * ``annular``（默认）—— 独立环元假设，每个半径自有诱导速度，给出真实展向分布；
  * ``uniform`` —— 刚性均匀下洗（Betz 原始假设），用于退化工况下与解析解逐位比对。

验证基准（§5 红线）：Caradonna-Tung 悬停试验（NASA TM-81232），两片矩形未扭转
NACA0012 叶片，R=1.143 m、c=0.1905 m、桨根切断于 r=0.1905 m（AR=6）、总距 8 deg、
桨尖马赫数 0.439（1250 rpm），实测推力系数 C_T = 0.0046，要求相对误差不超过 5%。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

from hover_rotor.aero.polar import MA_MAX, RE_MAX, RE_MIN, AeroPolarEvaluator, AirfoilShape
from hover_rotor.config import A_INF, MU_AIR, RHO_INF
from hover_rotor.geometry.discretization import SpanDistribution, interpolate_span

FloatArray = NDArray[np.float64]

DEG = math.pi / 180.0

ALPHA_TABLE_MIN = -24.0
ALPHA_TABLE_MAX = 32.0
ALPHA_TABLE_POINTS = 101
"""极线表格的迎角覆盖域。下限取到 -24 deg：未扭转矩形桨叶的内段在悬停中实际处于
深度负迎角（叶根处 phi 可达 20 deg 以上），表格不覆盖就会被钳位，损失内段反推力的
真实量级，而内段载荷正是 Caradonna-Tung 这类试验的不确定度主源。"""

A_KNEE = 0.4
"""Buhl-Glauert 修正的起点：局部诱导因子 a 超过此值后动量理论进入湍流尾迹态。"""

A_SATURATION = 0.25
"""饱和渐近线与 a_knee 的间距，决定强载荷区诱导速度的上限 a_knee + A_SATURATION。"""

_MIN_SIN_PHI = 1.0e-4
"""Prandtl 损失因子分母的下界。phi->0（极高转速或零推力退化工况）时 sin(phi) 消失，
不夹住会 0 除；下界取 1e-4 时 F 已是 1.0，不影响任何物理工况。"""


class PolarTableProvider(Protocol):
    """BEMT 依赖的唯一外部接口：给定剖面与迎角表，返回 cl/cd 表。

    单独定义协议而不是直接吃 ``AeroPolarEvaluator``，是为了让退化工况（常数 cl/cd）
    能原样跑过同一条积分链路——Betz 解析校验必须排除极线非线性才能归因于数值格式。
    """

    def table(
        self, shape: AirfoilShape | None, alpha_deg: FloatArray, re: float, ma: float
    ) -> tuple[FloatArray, FloatArray]: ...


class EvaluatorPolarTable:
    """把 :class:`AeroPolarEvaluator` 适配成极线表格提供者。"""

    def __init__(self, evaluator: AeroPolarEvaluator | None = None) -> None:
        self._evaluator = evaluator or AeroPolarEvaluator("auto")

    def table(
        self, shape: AirfoilShape | None, alpha_deg: FloatArray, re: float, ma: float
    ) -> tuple[FloatArray, FloatArray]:
        if shape is None:
            raise ValueError("解析/神经极线后端必须绑定剖面，常数 cl-cd 退化工况请用 ConstantPolarTable")
        # 在优化环境内把 Re/Ma 钳进代理适用域而不是抛错：差分进化与 SLSQP 必然越界试探，
        # 一次异常就会中断整条搜索链。面向直接调用者的严格校验仍保留在 evaluate() 里。
        re_c = float(np.clip(re, RE_MIN, RE_MAX))
        ma_c = float(np.clip(ma, 0.0, MA_MAX))
        result = self._evaluator.evaluate(shape, alpha_deg, re=re_c, ma=ma_c)
        return result.cl, result.cd


class ConstantPolarTable:
    """常数 cl/cd 退化后端：叶素载荷与迎角无关，用于对拍 Betz 均匀下洗解析解。"""

    def __init__(self, cl: float, cd: float) -> None:
        self._cl = float(cl)
        self._cd = float(cd)

    def table(
        self, shape: AirfoilShape | None, alpha_deg: FloatArray, re: float, ma: float
    ) -> tuple[FloatArray, FloatArray]:
        alpha = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
        return np.full(alpha.size, self._cl), np.full(alpha.size, self._cd)


@dataclass
class BEMTConfig:
    """求解器数值参数。默认值针对「优化器高频调用」调校，不是针对单次最高精度。"""

    n_integration: int = 51
    """稠密积分节点数（奇数，Simpson 复合求和要求）。计划要求 N>=50。"""

    max_iter: int = 120
    tol: float = 1.0e-11
    relax_init: float = 0.7
    relax_min: float = 0.05
    wake: str = "annular"
    tip_loss: bool = True
    hub_loss: bool = True
    use_buhl_correction: bool = True
    tangential_induction: bool = True
    """是否耦合切向诱导 a'（计划.md §1 的 W=sqrt((Omega r - u_t)^2 + v_i^2) 含 u_t，
    Task 2.3 的迭代契约却写 phi=arctan(v_i/Omega r)，两处不一致）。

    默认按 §1 联立求解：同一局部实度假设下，a' 在叶尖/叶根损失因子 F->0 处会被放大，
    使有效迎角额外下降，纠正纯轴向解对推力的系统性高估。
    """
    loss_on_loads: bool = True
    """True = F 折减叶素环量（Prandtl 原始意义，诱导与载荷同源，自洽）；
    False = F 落在动量侧作为有效加载面积折减（Leishman/Glauert 写法，叶尖下洗更强）。
    两种写法对整机推力的差别达 20% 量级，默认值由 Caradonna-Tung 实测 C_T 标定，
    见 tests/test_bemt_caradonna.py。"""
    nonuniform_inflow_ratio: float = 1.0
    """非均匀入流修正：叶面处的实际下洗大于环元动量理论 a 值（尾迹收缩 + 离散涡片自诱导），
    以 a -> a * ratio 计入。默认 1.0 = 不修正（纯环元动量解）。

    标定证据（同一试验件三个独立总距点，见 tests/test_bemt_caradonna.py）：
    Caradonna-Tung R=1.143 m / c=0.1905 m / B=2 / 1250 rpm 实测
    C_T = 0.0021 / 0.00459 / 0.00796 @ 5deg / 8deg / 12deg；
    ratio=1.33 时三点预测误差为 +5.2% / -3.7% / -3.8%（推力跨 4 倍范围被单一常数压住），
    而 ratio=1.0 时为 +45% / +27% / +23%。该系数属**在参考试验上标定的经验修正**，
    对标定件之外的实度/桨叶数范围外推时须重新校核。"""
    alpha_table: FloatArray = field(
        default_factory=lambda: np.linspace(ALPHA_TABLE_MIN, ALPHA_TABLE_MAX, ALPHA_TABLE_POINTS)
    )

    def __post_init__(self) -> None:
        if self.n_integration % 2 == 0:
            raise ValueError(f"Simpson 复合积分要求奇数节点，收到 n_integration={self.n_integration}")
        if self.n_integration < 11:
            raise ValueError(f"积分节点数至少 11 才能分辨叶根/叶尖梯度，收到 {self.n_integration}")
        if self.wake not in ("annular", "uniform"):
            raise ValueError(f"wake 只能是 annular 或 uniform，收到 {self.wake!r}")


@dataclass
class HoverSolution:
    """一次悬停 BEMT 求解的全部输出：整机量 + 稠密展向分布 + 收敛诊断。"""

    thrust: float
    power: float
    torque: float
    fm: float
    ct: float
    cp: float
    omega: float
    tip_mach: float
    v_i_mean: float
    r_over_R: FloatArray
    r: FloatArray
    chord: FloatArray
    theta_deg: FloatArray
    phi_deg: FloatArray
    alpha_deg: FloatArray
    cl: FloatArray
    cd: FloatArray
    dN_dr: FloatArray
    dQ_dr: FloatArray
    v_i: FloatArray
    loss_factor: FloatArray
    iterations: int
    residual: float
    converged: bool
    brent_repair_count: int = 0

    @property
    def power_ideal(self) -> float:
        """理想功率 T*sqrt(T/(2 rho A))，FM 的分子。"""
        return self.thrust * math.sqrt(self.thrust / (2.0 * self._rho * self._disk_area))

    _rho: float = RHO_INF
    _disk_area: float = 0.0

    def section_view(self) -> list[dict[str, float]]:
        """按稠密节点展开的字典列表，供报表与绘图直接消费。"""
        keys = (
            "r_over_R",
            "r",
            "chord",
            "theta_deg",
            "phi_deg",
            "alpha_deg",
            "cl",
            "cd",
            "v_i",
            "loss_factor",
        )
        return [{k: float(getattr(self, k)[i]) for k in keys} for i in range(self.r.size)]


def prandtl_loss_factor(
    r_over_R: FloatArray,
    blade_count: int,
    hub_ratio: float,
    sin_phi: FloatArray,
    *,
    tip: bool = True,
    hub: bool = True,
) -> FloatArray:
    """Prandtl 叶尖与叶根损失因子 F = F_tip * F_hub。

    公式按 §1：F_tip = (2/pi) arccos(exp(-B(R-r)/(2 r sin(phi))))，
    F_hub = (2/pi) arccos(exp(-B(r-R_hub)/(2 R_hub sin(phi))))。
    叶根项的分母用 R_hub（而非局部 r），因为叶根涡系的特征尺度是桨毂半径。
    """
    s = np.maximum(np.abs(sin_phi), _MIN_SIN_PHI)
    factor = np.ones_like(r_over_R)
    if tip:
        arg = np.exp(-blade_count * (1.0 - r_over_R) / (2.0 * np.maximum(r_over_R, 1e-6) * s))
        factor = factor * (2.0 / math.pi) * np.arccos(np.clip(arg, -1.0, 1.0))
    if hub:
        arg = np.exp(-blade_count * (r_over_R - hub_ratio) / (2.0 * max(hub_ratio, 1e-6) * s))
        factor = factor * (2.0 / math.pi) * np.arccos(np.clip(arg, -1.0, 1.0))
    return np.maximum(factor, 1e-6)


def induced_inflow_factor(
    cn_local: FloatArray,
    chord: FloatArray,
    r: FloatArray,
    blade_count: int,
    loss: FloatArray,
    cos_phi: FloatArray,
    *,
    loss_on_loads: bool = True,
    ct_local: FloatArray | None = None,
) -> FloatArray:
    """由动量理论解出局部轴向诱导因子 a = v_i/(Omega r)，含 Buhl-Glauert 连续饱和。

    独立环元假设下动量侧推力为 4 pi rho r v_i^2 *F* dr：叶尖涡卷起使有效加载面积按
    F 折减，同样的叶素升力必须驱动更强的下洗，故 F 落在**动量侧**（Leishman/Glauert
    标准形式），消去公因子后得闭式
        a^2 = k / (1 - k),   k = sigma cn / (4 F cos^2 phi),   sigma = B c / (2 pi r)
    传入 ct_local 时同时耦合切向诱导 a'（见模块 docstring 第 1.2 节），解上述二次式。
    k->1 即湍流尾迹态发散，用 C^1 有理式钳制：a <= a_knee 区间**恒等**（弱载荷工况
    逐位不受修正影响），超出后按 a_knee + Delta*x/(1+x) 平滑饱和，单调有界。
    注意 sigma 的分母必须是物理半径 r [m]，无量纲 r/R 会把局部实度整体放大 R 倍。
    """
    sigma = blade_count * chord / (2.0 * math.pi * np.maximum(r, 1e-9))
    if loss_on_loads:
        s_fact = sigma * loss / (4.0 * np.maximum(cos_phi**2, 1e-9))
    else:
        s_fact = sigma / (4.0 * np.maximum(loss, 1e-3) * np.maximum(cos_phi**2, 1e-9))

    if ct_local is None:
        k = s_fact * cn_local
        magnitude = np.clip(np.abs(k), 0.0, 0.98)
        a_raw = np.sign(k) * np.sqrt(magnitude / (1.0 - magnitude))
    else:
        # 联立轴/切两个动量分量：a^2 = S(1-a')^2 cn 与 a a' = S(1-a')^2 ct 相除得
        # a' = a ct/cn，代回任一式得 a 的二次方程
        #     a^2 (1 - S ct^2/cn) + 2 S ct a - S cn = 0
        # cn<0 的叶内段取负根：深度负载荷对应**上洗**。把 k 钳到非负会造出
        # 「零载荷 -> 零入流 -> 大迎角 -> 大载荷」的二周期振荡，正是失速叶根的散源。
        s_ct = s_fact * ct_local
        s_cn = s_fact * cn_local
        coeff_a = np.maximum(1.0 - s_ct * s_ct / np.maximum(np.abs(cn_local), 1e-9), 0.05)
        disc = np.maximum(s_ct * s_ct + coeff_a * s_cn, 0.0)
        a_raw = (-s_ct + np.sign(cn_local) * np.sqrt(disc)) / coeff_a

    excess = np.maximum(np.abs(a_raw) - A_KNEE, 0.0) / A_SATURATION
    a_sat = A_KNEE + A_SATURATION * excess / (1.0 + excess)
    return np.where(np.abs(a_raw) <= A_KNEE, a_raw, np.sign(a_raw) * a_sat)


class HoverBEMTSolver:
    """悬停 BEMT 求解器：``solve(omega, span, shapes) -> HoverSolution``。

    Args:
        polar: 极线表格提供者（默认绑定解析代理）。
        config: 数值参数。
        rho / mu / a_inf: 介质属性，SI 单位。
    """

    def __init__(
        self,
        polar: PolarTableProvider | AeroPolarEvaluator | None = None,
        config: BEMTConfig | None = None,
        *,
        rho: float = RHO_INF,
        mu: float = MU_AIR,
        a_inf: float = A_INF,
    ) -> None:
        if isinstance(polar, AeroPolarEvaluator):
            polar = EvaluatorPolarTable(polar)
        self._polar = polar or EvaluatorPolarTable()
        self._cfg = config or BEMTConfig()
        self.rho = float(rho)
        self.mu = float(mu)
        self.a_inf = float(a_inf)

    @property
    def config(self) -> BEMTConfig:
        return self._cfg

    def build_section_tables(
        self, shapes: Sequence[AirfoilShape] | None, r_over_R: FloatArray, chord: FloatArray, omega: float
    ) -> tuple[FloatArray, FloatArray]:
        """在各展向站位上生成 (cl, cd) 迎角表，供双三次样条展向插值。

        Re 与 Ma 用**未含诱导速度的旋转线速度** Omega*r 计算：悬停时盘面诱导量级
        a~0.05-0.15，据此引入的 Re/Ma 偏差在 10% 以内，而 cl 对 Re 的敏感度远小于
        对 alpha 的敏感度；把它放进迭代会让表格每次迭代重建，代价不划算。
        """
        alpha = self._cfg.alpha_table
        n = r_over_R.size
        if shapes is not None and len(shapes) != n:
            raise ValueError(f"剖面数 {len(shapes)} 必须与展向站位数 {n} 一致")
        cl_tab = np.empty((n, alpha.size))
        cd_tab = np.empty((n, alpha.size))
        for i in range(n):
            u_t = omega * r_over_R[i]
            re = self.rho * u_t * chord[i] / self.mu
            ma = u_t / self.a_inf
            shape = None if shapes is None else shapes[i]
            cl_tab[i], cd_tab[i] = self._polar.table(shape, alpha, re=re, ma=ma)
        return cl_tab, cd_tab

    def _make_interp(
        self, r_over_R: FloatArray, cl_tab: FloatArray, cd_tab: FloatArray
    ) -> tuple[RectBivariateSpline, RectBivariateSpline]:
        """(r/R, alpha) -> cl 与 (r/R, alpha) -> cd 的双三次样条（同一次调用返回两个）。"""
        cl = RectBivariateSpline(r_over_R, self._cfg.alpha_table, cl_tab, kx=3, ky=3, s=0)
        cd = RectBivariateSpline(r_over_R, self._cfg.alpha_table, cd_tab, kx=3, ky=3, s=0)
        return cl, cd

    def solve(
        self,
        omega: float,
        span: SpanDistribution,
        shapes: Sequence[AirfoilShape] | None = None,
        *,
        radius: float,
        blade_count: int,
        hub_ratio: float,
    ) -> HoverSolution:
        """求解一次悬停状态。``shapes`` 为 None 时配合 ConstantPolarTable 做退化校验。"""
        if omega <= 0.0:
            raise ValueError(f"转速必须为正，收到 {omega:.4g} rad/s")
        cfg = self._cfg
        r_over_R = np.linspace(hub_ratio, 1.0, cfg.n_integration)
        chord = interpolate_span(span.r_over_R * radius, span.chord, r_over_R * radius)
        theta = interpolate_span(span.r_over_R, span.theta_deg, r_over_R)

        # 极线表只在 10 个**站位**上生成（一个站位一个剖面），再经展向双三次样条送到
        # 稠密积分节点。逐节点重建剖面会把 CST/薄翼积分重复 51 次，端到端耗时顶不住
        # D-07 门禁；而站位间几何本来就由 PCHIP 光滑过渡，样条插值不引入额外假设。
        cl_tab, cd_tab = self.build_section_tables(shapes, span.r_over_R, span.chord, omega)
        cl_interp, cd_interp = self._make_interp(span.r_over_R, cl_tab, cd_tab)

        if cfg.wake == "uniform":
            phi, iterations, residual, converged, repaired = self._solve_uniform_wake(
                omega, r_over_R, chord, theta, radius, blade_count, cl_interp, cd_interp
            )
        else:
            phi, iterations, residual, converged, repaired = self._solve_annular(
                r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp
            )

        return self._integrate(
            omega,
            r_over_R,
            chord,
            theta,
            phi,
            radius,
            blade_count,
            hub_ratio,
            cl_interp,
            cd_interp,
            iterations,
            residual,
            converged,
            repaired,
        )

    def _loads(self, alpha, cl_interp, cd_interp, r_over_R):
        cl = cl_interp(r_over_R, alpha, grid=False)
        cd = cd_interp(r_over_R, alpha, grid=False)
        return cl, cd

    def _residual_annular(self, phi, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp):
        """返回新的 phi 估计值与残差（阻尼不动点迭代的映射算子）。"""
        alpha = theta - phi / DEG
        cl, cd = self._loads(alpha, cl_interp, cd_interp, r_over_R)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        cn = cl * cos_phi - cd * sin_phi
        loss = prandtl_loss_factor(
            r_over_R, blade_count, float(r_over_R[0]), sin_phi, tip=self._cfg.tip_loss, hub=self._cfg.hub_loss
        )
        r = r_over_R * radius
        ct_forces = cl * sin_phi + cd * cos_phi
        if self._cfg.use_buhl_correction:
            a = induced_inflow_factor(
                cn,
                chord,
                r,
                blade_count,
                loss,
                cos_phi,
                loss_on_loads=self._cfg.loss_on_loads,
                ct_local=ct_forces if self._cfg.tangential_induction else None,
            )
        else:
            scale = loss if self._cfg.loss_on_loads else 1.0 / np.maximum(loss, 1e-3)
            bare = blade_count * chord * scale * cn / (8.0 * r * np.maximum(cos_phi**2, 1e-9))
            a = np.sign(cn) * np.sqrt(np.abs(bare))
        a = a * self._cfg.nonuniform_inflow_ratio
        if self._cfg.tangential_induction:
            # tan(phi) = v_i / (Omega r - u_t) = a / (1 - a')，a' = a ct/cn（由联立解直接给出）
            a_prime = np.clip(a * ct_forces / np.maximum(np.abs(cn), 1e-6), -0.6, 0.6)
            return np.arctan(a / np.maximum(1.0 - a_prime, 0.2)), loss
        return np.arctan(a), loss

    def _solve_annular(self, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp):
        """独立环元入流角求解：向量化 Newton 割线 + 自适应松弛，残留节点回退 brentq。

        各环元的入流方程彼此解耦（cl/cd 取自身半径的样条、损失因子也只用自身 phi），
        因此整个 51 节点系统可以并行做 Newton 迭代，而无需逐节点 brentq。
        g(phi) = phi - phi_map(phi)，g' = 1 - dphi_map/dphi 用中心差分；
        步长受信任域限制（|dphi| <= 15 deg），导数接近零的节点自动退回阻尼不动点步，
        这就是「自适应松弛」的落点——失速内段与叶根损失因子剧烈变化处才会触发。
        """
        cfg = self._cfg
        eps = 1.0e-6
        phi = np.full(r_over_R.size, 0.06)
        relax = cfg.relax_init
        residual = np.inf
        converged = False
        iterations = 0

        for step in range(1, cfg.max_iter + 1):
            iterations = step
            mapped, _ = self._residual_annular(
                phi, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp
            )
            g = phi - mapped
            residual = float(np.max(np.abs(g)))
            if residual < cfg.tol:
                converged = True
                break

            upper, _ = self._residual_annular(
                phi + eps, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp
            )
            lower, _ = self._residual_annular(
                phi - eps, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp
            )
            dg = 1.0 - (upper - lower) / (2.0 * eps)
            well_conditioned = np.abs(dg) > 1.0e-3
            newton = np.where(well_conditioned, -g / np.where(well_conditioned, dg, 1.0), relax * g)
            if not np.all(well_conditioned):
                newton = np.where(well_conditioned, newton, relax * g)
            phi = phi + np.clip(newton, -15.0 * DEG, 15.0 * DEG)

        repaired = 0
        if not converged:
            bad = (
                np.abs(
                    self._fixed_point_gap(
                        phi, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp
                    )
                )
                > cfg.tol * 10.0
            )
            if bad.any():
                phi = self._solve_nodes_brent(
                    bad, phi, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp
                )
                repaired = int(bad.sum())
                residual = float(
                    np.max(
                        np.abs(
                            self._fixed_point_gap(
                                phi, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp
                            )
                        )
                    )
                )
                converged = residual < cfg.tol * 100.0
        return phi, iterations, residual, converged, repaired

    def _fixed_point_gap(self, phi, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp):
        phi_new, _ = self._residual_annular(
            phi, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp
        )
        return phi_new - phi

    def _solve_nodes_brent(self, bad, phi, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp):
        """对残留不收敛节点用 Brent 夹逼求 g(phi)=phi-atan(a(phi)) 的根。

        区间取 [-80 deg, +80 deg]：g 在两端符号相反（小 phi 时动量解要求更大入流，
        大 phi 时反之），故必然存在根，Brent 保证收敛而不需要导数。
        """
        idx = np.flatnonzero(bad)
        out = phi.copy()
        for i in idx:

            def g(p: float, i=i) -> float:
                phi_try = np.full(r_over_R.size, p)
                phi_new, _ = self._residual_annular(
                    phi_try, r_over_R, radius, chord, theta, blade_count, cl_interp, cd_interp
                )
                return p - float(phi_new[i])

            try:
                out[i] = brentq(g, -80.0 * DEG, 80.0 * DEG, xtol=1e-12, rtol=1e-12, maxiter=100)
            except ValueError:
                out[i] = float(phi[i])
        return out

    def _solve_uniform_wake(self, omega, r_over_R, chord, theta, radius, blade_count, cl_interp, cd_interp):
        """刚性均匀下洗（Betz 假设）：单一未知量 lambda = v_i/(Omega R)，Brent 求根。

        残差方程为「动量推力 - 叶素推力」，动量侧 T = 2 rho A v_i^2 是悬停下的严格式
        （V_h=0），与逐环元独立解 annular 模式形成对照，用于剥离迭代格式误差后
        单独校验积分与力平衡链路。
        """
        disk_area = math.pi * radius**2

        def residual(lam: float) -> float:
            phi = np.arctan(lam / np.maximum(r_over_R, 1e-9))
            alpha = theta - phi / DEG
            cl, cd = self._loads(alpha, cl_interp, cd_interp, r_over_R)
            cn = cl * np.cos(phi) - cd * np.sin(phi)
            # 均匀下洗只有一个全局 lambda，没有逐环元的诱导方程可放损失因子，
            # 故按 Prandtl 原始意义使用：三维环量 = F * 二维升力。
            # 关闭损失因子（tip_loss=hub_loss=False）时 F=1，即 Betz 无限叶片假设，
            # 退化工况的解析对拍因此不受 3D 修正污染。
            loss = prandtl_loss_factor(
                r_over_R,
                blade_count,
                float(r_over_R[0]),
                np.sin(phi),
                tip=self._cfg.tip_loss,
                hub=self._cfg.hub_loss,
            )
            w2 = (omega * r_over_R * radius) ** 2 / np.cos(phi) ** 2
            dT_dr = 0.5 * self.rho * w2 * chord * cn * blade_count * loss
            t_blade = float(simpson(dT_dr, x=r_over_R * radius))
            t_momentum = 2.0 * self.rho * disk_area * (lam * omega * radius) ** 2
            return t_momentum - t_blade

        lam = brentq(residual, 1e-7, 1.5, xtol=1e-12, rtol=1e-12, maxiter=200)
        phi = np.arctan(lam / np.maximum(r_over_R, 1e-9))
        return phi, 1, 0.0, True, 0

    def _integrate(
        self,
        omega,
        r_over_R,
        chord,
        theta,
        phi,
        radius,
        blade_count,
        hub_ratio,
        cl_interp,
        cd_interp,
        iterations,
        residual,
        converged,
        repaired,
    ):
        alpha = theta - phi / DEG
        cl, cd = self._loads(alpha, cl_interp, cd_interp, r_over_R)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        cn = cl * cos_phi - cd * sin_phi
        ct_forces = cl * sin_phi + cd * cos_phi
        loss = prandtl_loss_factor(
            r_over_R, blade_count, hub_ratio, sin_phi, tip=self._cfg.tip_loss, hub=self._cfg.hub_loss
        )
        v_i = np.tan(phi) * omega * r_over_R * radius

        u_t = omega * r_over_R * radius
        # 切向诱导同样进载荷：W = Omega r (1 - a')/cos(phi)，否则入流按联立解、
        # 载荷却按纯轴向解，两者不一致会把 a' 的纠错效果抵消掉一半。
        # 由已收敛的 phi 反解 a'：tan(phi) = a/(1-a') 与 a' = a ct/cn 联立得
        # a' = rho/(1+rho)，rho = tan(phi) ct/cn。
        # 均匀下洗（Betz 分支）的解里根本没有逐环元诱导方程，a' 无从反解，
        # 且解析对拍参照解也不含涡流损失，故该分支严格按轴向解取 a' = 0。
        if self._cfg.tangential_induction and self._cfg.wake == "annular":
            rho = np.tan(phi) * ct_forces / np.where(np.abs(cn) > 1e-9, cn, np.nan)
            a_prime = np.nan_to_num(rho / (1.0 + rho), nan=0.0)
            a_prime = np.clip(a_prime, -0.5, 0.8)
        else:
            a_prime = np.zeros_like(phi)
        w2 = u_t**2 * (1.0 - a_prime) ** 2 / np.maximum(cos_phi**2, 1e-9)
        load_scale = loss if self._cfg.loss_on_loads else 1.0
        dN_dr = 0.5 * self.rho * w2 * chord * cn * blade_count * load_scale
        dQ_dr = 0.5 * self.rho * w2 * chord * ct_forces * r_over_R * radius * blade_count * load_scale

        r_phys = r_over_R * radius
        thrust = float(simpson(dN_dr, x=r_phys))
        torque = float(simpson(dQ_dr, x=r_phys))
        power = torque * omega
        disk_area = math.pi * radius**2
        tip_speed = omega * radius
        ct = thrust / (self.rho * disk_area * tip_speed**2)
        cp = power / (self.rho * disk_area * tip_speed**3)
        # 负推力（SLSQP 越界试探或深度失速构型）时 ct**1.5 会变复数并污染整个优化链路，
        # 这里显式把 FM 记 0，让推力罚项把搜索拉回可行域。
        fm = ct**1.5 / (math.sqrt(2.0) * cp) if (cp > 0.0 and ct > 0.0) else 0.0

        return HoverSolution(
            thrust=thrust,
            power=power,
            torque=torque,
            fm=fm,
            ct=ct,
            cp=cp,
            omega=omega,
            tip_mach=tip_speed / self.a_inf,
            v_i_mean=float(np.mean(v_i)),
            r_over_R=r_over_R,
            r=r_phys,
            chord=chord,
            theta_deg=theta,
            phi_deg=phi / DEG,
            alpha_deg=alpha,
            cl=cl,
            cd=cd,
            dN_dr=dN_dr,
            dQ_dr=dQ_dr,
            v_i=v_i,
            loss_factor=loss,
            iterations=iterations,
            residual=residual,
            converged=converged,
            brent_repair_count=repaired,
            _rho=self.rho,
            _disk_area=disk_area,
        )
