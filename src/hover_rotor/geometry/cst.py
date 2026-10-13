"""钝尾缘拓展 CST 翼型几何发生器（计划.md Task 1.1）。

数学定义（弦向归一化坐标 psi = x/c，厚度坐标 zeta = z/c）：

    zeta_u(psi) = C(psi) * S_u(psi) + psi * dz_te / 2
    zeta_l(psi) = C(psi) * S_l(psi) - psi * dz_te / 2
    C(psi)      = psi**N1 * (1 - psi)**N2        类函数，默认 N1=0.5, N2=1.0
    S(psi)      = sum_i A_i * K_{i,n} * psi**i * (1-psi)**(n-i)   Bernstein 形函数

类函数在 psi=0 与 psi=1 处均为零，故钝尾缘厚度项在尾缘完全接管几何：
zeta_u(1) - zeta_l(1) == dz_te 恒成立，尖后缘在参数化层面被杜绝而非事后修补。

返回点阵的遍历方向：自尾缘下沿绕经前缘至尾缘上沿（与平面直角坐标系下的顺时针一致），
制造层若需 Selig/XFOIL 惯例（上表面自尾缘至前缘）由 cad 层翻转点序，本模块不做双份表示。
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

DEFAULT_N1 = 0.5
DEFAULT_N2 = 1.0

FloatArray = NDArray[np.float64]


def bernstein_basis(n: int, psi: FloatArray) -> FloatArray:
    """返回 (len(psi), n+1) 的 Bernstein 基矩阵 K_{i,n} psi^i (1-psi)^(n-i)。"""
    if n < 1:
        raise ValueError(f"形函数阶数 n 至少为 1，收到 {n}")
    psi = np.asarray(psi, dtype=np.float64)
    idx = np.arange(n + 1)
    k = np.array([math.comb(n, int(i)) for i in idx], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        # psi=0 且 i=0，或 psi=1 且 i=n 时 0**0 == 1，正是端点所需
        basis = (
            k[np.newaxis, :]
            * np.power(psi[:, None], idx[None, :])
            * np.power(1.0 - psi[:, None], (n - idx)[None, :])
        )
    return np.nan_to_num(basis, nan=0.0)


def class_function(psi: FloatArray, n1: float = DEFAULT_N1, n2: float = DEFAULT_N2) -> FloatArray:
    """类函数 C_{n1,n2}(psi) = psi^n1 * (1-psi)^n2。"""
    psi = np.asarray(psi, dtype=np.float64)
    return np.power(psi, n1) * np.power(1.0 - psi, n2)


def shape_function(psi: FloatArray, weights: FloatArray) -> FloatArray:
    """形函数 S(psi) = sum_i A_i K_{i,n} psi^i (1-psi)^(n-i)，阶数由 len(weights)-1 决定。"""
    weights = np.asarray(weights, dtype=np.float64)
    return bernstein_basis(weights.size - 1, psi) @ weights


def _circuit_grids(n_points: int) -> tuple[FloatArray, FloatArray]:
    """构造沿闭合回路单调推进的两段余弦加密弦向网格。

    下半面取 psi 自 1 降到 0，上半面取 psi 自 0+ 升到 1，前缘点恰好出现一次，
    两端各留一个尾缘角点。余弦映射在 psi=0 与 psi=1 两端同时加密，
    既分辨前缘曲率又给足尾缘采样。
    """
    if n_points < 20:
        raise ValueError(f"采样点数至少 20 才能解析前缘曲率，收到 {n_points}")
    n_lower = n_points - n_points // 2
    n_upper = n_points // 2

    psi_lower = np.cos(0.5 * math.pi * np.linspace(0.0, 1.0, n_lower))
    psi_upper = np.sin(0.5 * math.pi * np.linspace(0.0, 1.0, n_upper + 1)[1:])
    psi_lower[0] = 1.0
    psi_lower[-1] = 0.0
    psi_upper[-1] = 1.0
    return psi_lower, psi_upper


def generate_flatback_cst(
    au: FloatArray,
    al: FloatArray,
    dz_te: float,
    n_points: int = 200,
    *,
    chord: float = 1.0,
    te_style: str = "flat",
    n1: float = DEFAULT_N1,
    n2: float = DEFAULT_N2,
    te_arc_points: int = 16,
) -> FloatArray:
    """生成带钝尾缘的闭合翼型型线点阵。

    Args:
        au: 上表面形函数权重 A_u，长度 n+1。
        al: 下表面形函数权重 A_l，长度 n+1。
        dz_te: 相对尾缘厚度 t_te/c，必须为正（尖尾缘不被接受）。
        n_points: 返回点数。`te_style="flat"` 时精确等于该值；`"round"` 时额外追加半圆封头弧点。
        chord: 物理弦长。默认 1.0 返回归一化坐标，给定实弦长则返回同单位物理坐标。
        te_style: "flat" 垂直切平封闭；"round" 以半径 dz_te/2 的半圆封头向前包覆。
        n1, n2: 类函数指数，默认 (0.5, 1.0) 圆头钝尾基准。
        te_arc_points: "round" 模式的封头弧点数（不含两端角点）。

    Returns:
        (M, 2) 点阵。第 0 点为尾缘下沿角点 (psi=1, -dz_te/2)；`flat` 模式末点为尾缘上沿
        角点 (psi=1, +dz_te/2)，首末以垂直直线段闭合；`round` 模式末点为封头弧上紧邻
        尾缘下沿的最后一个弧点，多边形由短弦自动闭合。
    """
    if dz_te <= 0.0:
        raise ValueError(f"相对尾缘厚度 dz_te 必须为正，收到 {dz_te:.3e}；尖尾缘在本系统参数化层即被拒绝")
    if te_style not in ("flat", "round"):
        raise ValueError(f'te_style 只能是 "flat" 或 "round"，收到 {te_style!r}')

    au = np.asarray(au, dtype=np.float64)
    al = np.asarray(al, dtype=np.float64)
    if au.shape != al.shape or au.ndim != 1:
        raise ValueError(f"au 与 al 必须是等长一维向量，收到 {au.shape} 与 {al.shape}")

    psi_lower, psi_upper = _circuit_grids(n_points)
    psi = np.concatenate((psi_lower, psi_upper))

    cls = class_function(psi, n1, n2)
    half = 0.5 * dz_te
    zeta_u = cls * shape_function(psi, au) + psi * half
    zeta_l = cls * shape_function(psi, al) - psi * half
    n_lo = psi_lower.size
    # 下半面按 psi 递减序（尾缘->前缘），上半面按 psi 递增序（前缘->尾缘），直接拼接即成回路
    zeta = np.concatenate((zeta_l[:n_lo], zeta_u[n_lo:]))
    pts = np.column_stack((psi, zeta))

    if te_style == "round":
        # 封头：以尾缘中点 (1, 0) 为心、半径 dz_te/2 的半圆，自上角点绕过 x>1 区回到下角点。
        # 因此 round 模式的最大 x 包络为 1 + dz_te/2，flat 模式严格截止于 x = 1。
        r = 0.5 * dz_te
        theta = np.linspace(math.pi / 2.0, -math.pi / 2.0, te_arc_points + 2)[1:-1]
        arc = np.column_stack((1.0 + r * np.cos(theta), r * np.sin(theta)))
        pts = np.vstack((pts, arc))

    return pts * chord if chord != 1.0 else pts


def trailing_edge_gap(points: FloatArray) -> float:
    """`flat` 模式首末点的 z 向间距，即实际生成的相对尾缘厚度。"""
    points = np.asarray(points, dtype=np.float64)
    return float(points[-1, 1] - points[0, 1])


def relative_te_thickness(te_thickness: float, chord: float) -> float:
    """把全展恒定的绝对尾缘厚度换算为该站位的相对厚度 dz_te = t_te / c。"""
    if chord <= 0.0:
        raise ValueError(f"弦长必须为正值，收到 {chord:.4g}")
    return te_thickness / chord


def fit_shape_weights(
    psi: FloatArray,
    zeta: FloatArray,
    n: int,
    *,
    dz_te: float = 0.0,
    n1: float = DEFAULT_N1,
    n2: float = DEFAULT_N2,
) -> FloatArray:
    """CST 反算：由目标厚度坐标最小二乘还原 n 阶形函数权重。

    ``zeta`` 必须是**已包含**尾缘楔形贡献 psi*dz_te/2 的实际目标纵坐标；本函数在求
    形状函数前先扣除该项再除以类函数。若误传不含楔形项的原始型线，除法在 psi->1 处
    以非零分子除以趋于零的类函数，目标发散且还原权重完全失真。

    端点处类函数为零、除法奇异，故只用内部点参与拟合。
    用于把基准翼型（如 Caradonna-Tung 的 NACA0012）纳入 CST 参数空间。
    """
    psi = np.asarray(psi, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    if psi.size != zeta.size:
        raise ValueError(f"psi 与 zeta 长度须一致，收到 {psi.size} 与 {zeta.size}")

    cls = class_function(psi, n1, n2)
    interior = (psi > 0.0) & (psi < 1.0)
    if interior.sum() <= n + 1:
        raise ValueError(f"内部采样点 {interior.sum()} 个不足以稳定拟合 {n} 阶权重，需多于 {n + 1}")
    target = (zeta[interior] - psi[interior] * 0.5 * dz_te) / cls[interior]
    design = bernstein_basis(n, psi[interior])
    weights, *_ = np.linalg.lstsq(design, target, rcond=None)
    return weights
