"""钝尾缘基底阻力 Hoerner 半经验修正（计划.md Task 2.1）。

平直切断的后缘使上下表面流动在尾缘脱体分离，形成周期性交替脱落的卡门涡街与低压
回流死水区，产生附加压差阻力——基底阻力（Base Drag）。粘性-无粘耦合迭代求解器
（XFOIL 类）在该分离区常发散，故低阶可微链路必须以半经验项显式补回这部分损失。

计划.md D-01 裁定公式为
    dCd_base = 0.135 * (t_te / c) / cd_clean ** (1/3)
即基底阻力与尾缘相对厚度成正比、与干净剖面阻力的立方根成**反**比——低阻翼型
（附层 thinner、尾流窄）对同样的尾缘厚度更敏感，这一反向关系由 Hoerner 与
Standish-van Dam 的风洞数据确立，并被验收基准工况唯一确定。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

COEFFICIENT = 0.135
"""Hoerner 基底阻力系数，Standish & van Dam 风洞拟合值"""

CD_FLOOR = 1.0e-6
"""剖面阻力数值下界，防止零阻退化工况下立方根分母发散"""

FloatArray = NDArray[np.float64]


def compute_base_drag_hoerner(
    te_over_chord: float | FloatArray, cd_clean: float | FloatArray
) -> FloatArray | float:
    """由尾缘相对厚度与干净剖面阻力计算基底阻力增量 dCd_base。

    Args:
        te_over_chord: 局部相对尾缘厚度 t_te/c，>= 0；标量或数组。
        cd_clean: 无尾缘钝度时的剖面型阻系数，内部钳到 CD_FLOOR 以上。

    Returns:
        基底阻力增量，与总剖面阻力代数叠加：cd = cd_clean + dCd_base。
        两个输入均为标量时返回 float，便于优化器目标函数直接取用。
    """
    ratio = np.asarray(te_over_chord, dtype=np.float64)
    cd = np.asarray(cd_clean, dtype=np.float64)

    if np.any(ratio < 0.0):
        raise ValueError(f"尾缘相对厚度不得为负，收到最小值 {float(np.min(ratio)):.3e}")

    value = COEFFICIENT * ratio / np.cbrt(np.maximum(cd, CD_FLOOR))
    if value.ndim == 0:
        return float(value)
    return value


def apply_base_drag(cd_profile: float | FloatArray, te_over_chord: float | FloatArray) -> FloatArray | float:
    """把基底阻力叠加到剖面总阻力上，返回钝尾翼型的实际 cd。"""
    return np.asarray(cd_profile, dtype=np.float64) + compute_base_drag_hoerner(te_over_chord, cd_profile)
