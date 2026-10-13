"""展向非线性余弦聚点离散与单调保形样条插值（计划.md Task 1.2）。

悬停流场的气动载荷与环量梯度在叶根（附面层干扰）与叶尖（涡卷起）**两端**同时急剧变化，
等间距线性采样会在两端引入不可接受的截断误差。离散法则采用 Chebyshev-Lobatto 型
对称余弦聚点（计划.md D-05）：

    r_bar_i = r_bar_hub + (1 - r_bar_hub) * 0.5 * [1 - cos( pi*(i-1)/(N-1) )],  i = 1..N

半弦长映射 0.5*(1-cos) 使采样密度在 r_bar_hub 与 1.0 两端同时升高、中段最疏，
与十截面参考表的前三个站位（0.150 / 0.176 / 0.252）逐位吻合，端点严格闭合。
注意：非对称的 [1 - cos(...)] 形式只在叶根聚点，与上述物理要求相悖，不予采用。

展向弦长与扭转分布用 PCHIP（单调保形分段三次 Hermite）重构，避免高阶多项式龙格振荡。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from hover_rotor.config import DEFAULT_HUB_RATIO, DEFAULT_N_SECTIONS

FloatArray = NDArray[np.float64]


def cosine_span_positions(
    hub_ratio: float = DEFAULT_HUB_RATIO,
    n_sections: int = DEFAULT_N_SECTIONS,
) -> FloatArray:
    """无量纲展向站位 r/R，两端聚点，自 hub_ratio 严格终止于 1.0。"""
    if not 0.0 < hub_ratio < 1.0:
        raise ValueError(f"轮毂比必须落在 (0, 1) 内，收到 {hub_ratio:.4g}")
    if n_sections < 2:
        raise ValueError(f"截面数至少为 2，收到 {n_sections}")

    idx = np.arange(n_sections, dtype=np.float64) / (n_sections - 1)
    positions = hub_ratio + (1.0 - hub_ratio) * 0.5 * (1.0 - np.cos(math.pi * idx))
    positions[0] = hub_ratio
    positions[-1] = 1.0
    return positions


def discretize_rotor_span(
    radius: float,
    hub_ratio: float = DEFAULT_HUB_RATIO,
    n_sections: int = DEFAULT_N_SECTIONS,
) -> FloatArray:
    """物理展向站位 r [m]，端点严格为 hub_ratio*radius 与 radius。"""
    if radius <= 0.0:
        raise ValueError(f"旋翼半径必须为正值，收到 {radius:.4g}")
    return cosine_span_positions(hub_ratio, n_sections) * radius


def interpolate_span(r_ctrl: FloatArray, q_ctrl: FloatArray, r_eval: FloatArray) -> FloatArray:
    """以 PCHIP 在展向控制节点间单调保形插值。

    控制节点须覆盖 [r_min, r_max] 端点，否则评估点落界外时按线性外推处理——
    优化器给出的边界若不含站位端点属于建模错误，此处显式拒绝而非静默外插。
    """
    r_ctrl = np.asarray(r_ctrl, dtype=np.float64)
    q_ctrl = np.asarray(q_ctrl, dtype=np.float64)
    r_eval = np.asarray(r_eval, dtype=np.float64)

    if r_ctrl.shape != q_ctrl.shape or r_ctrl.ndim != 1:
        raise ValueError(f"控制节点数组形状不匹配：{r_ctrl.shape} 与 {q_ctrl.shape}")
    if r_ctrl.size < 2:
        raise ValueError(f"至少需要 2 个展向控制节点，收到 {r_ctrl.size}")
    if np.any(np.diff(r_ctrl) <= 0.0):
        raise ValueError("展向控制节点半径必须严格单调递增，收到非递增序列")

    lo, hi = float(np.min(r_eval)), float(np.max(r_eval))
    if lo < float(np.min(r_ctrl)) - 1e-12 or hi > float(np.max(r_ctrl)) + 1e-12:
        raise ValueError(
            f"评估站位 [{lo:.4g}, {hi:.4g}] 超出控制节点范围 "
            f"[{float(np.min(r_ctrl)):.4g}, {float(np.max(r_ctrl)):.4g}]"
        )
    return PchipInterpolator(r_ctrl, q_ctrl)(r_eval)


@dataclass(frozen=True)
class SpanDistribution:
    """一次反设计迭代中的展向几何分布，Task 2.3 BEMT 求解器的几何输入契约。"""

    r: FloatArray
    r_over_R: FloatArray
    chord: FloatArray
    theta_deg: FloatArray

    @property
    def n_sections(self) -> int:
        return int(self.r.size)

    def section_props(self) -> list[dict[str, float]]:
        """逐站位字典视图，供报表与导出层直接消费。"""
        return [
            {
                "index": i + 1,
                "r": float(self.r[i]),
                "r_over_R": float(self.r_over_R[i]),
                "chord": float(self.chord[i]),
                "theta_deg": float(self.theta_deg[i]),
            }
            for i in range(self.n_sections)
        ]


def build_span_distribution(
    radius: float,
    r_ctrl: FloatArray,
    chord_ctrl: FloatArray,
    theta_ctrl_deg: FloatArray,
    *,
    hub_ratio: float = DEFAULT_HUB_RATIO,
    n_sections: int = DEFAULT_N_SECTIONS,
) -> SpanDistribution:
    """把展向控制节点展开为 N 个制造截面的几何分布。

    单位契约：``r_ctrl`` 为无量纲站位 r/R（须含端点 hub_ratio 与 1.0），
    ``chord_ctrl`` 为**物理弦长 [m]**（与十截面参考表换算须先乘 R），
    ``theta_ctrl_deg`` 为安装角 [deg]。
    """
    r_over_R = cosine_span_positions(hub_ratio, n_sections)
    r = r_over_R * radius
    chord = interpolate_span(np.asarray(r_ctrl, dtype=np.float64) * radius, chord_ctrl, r)
    theta = interpolate_span(np.asarray(r_ctrl, dtype=np.float64), theta_ctrl_deg, r_over_R)
    return SpanDistribution(r=r, r_over_R=r_over_R, chord=chord, theta_deg=theta)
