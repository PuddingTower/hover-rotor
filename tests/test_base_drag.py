"""Task 2.1 验收：Hoerner 基底阻力修正器。"""

from __future__ import annotations

import numpy as np
import pytest

from hover_rotor.aero.base_drag import CD_FLOOR, COEFFICIENT, apply_base_drag, compute_base_drag_hoerner


def test_baseline_case_hits_accepted_band() -> None:
    """计划.md D-01 的唯一判定工况：t_te/c=0.02、cd_clean=0.01 -> [0.0125, 0.0130]。"""
    value = compute_base_drag_hoerner(0.02, 0.01)
    assert isinstance(value, float)
    assert 0.0125 <= value <= 0.0130


def test_formula_is_division_not_multiplication() -> None:
    """锁定 D-01：分母是 cd^(1/3)。乘法形式会把该工况算成 0.00058，直接证伪。"""
    assert compute_base_drag_hoerner(0.02, 0.01) == pytest.approx(0.135 * 0.02 / 0.01 ** (1 / 3))
    assert compute_base_drag_hoerner(0.02, 0.01) > COEFFICIENT * 0.02 * 0.01 ** (1 / 3)


def test_sharp_edge_yields_zero_base_drag() -> None:
    assert compute_base_drag_hoerner(0.0, 0.01) == 0.0


def test_low_drag_case_does_not_diverge() -> None:
    """零阻/负阻退化输入不得产生 inf 或 nan——优化器一旦撞上就会污染整条梯度。"""
    for cd in (0.0, -1.0, 1e-30):
        value = compute_base_drag_hoerner(0.02, cd)
        assert np.isfinite(value)
    assert compute_base_drag_hoerner(0.02, 0.0) == pytest.approx(COEFFICIENT * 0.02 / np.cbrt(CD_FLOOR))


@pytest.mark.parametrize("cd", [0.004, 0.01, 0.02])
def test_monotone_increasing_in_te_ratio(cd: float) -> None:
    ratios = np.linspace(0.0, 0.06, 25)
    values = compute_base_drag_hoerner(ratios, cd)
    assert np.all(np.diff(values) > 0.0)


@pytest.mark.parametrize("ratio", [0.005, 0.02, 0.05])
def test_monotone_decreasing_in_clean_drag(ratio: float) -> None:
    """低阻翼型对同样的尾缘钝度更敏感：dCd_base 随 cd_clean 上升而下降。"""
    cds = np.linspace(0.002, 0.05, 20)
    values = compute_base_drag_hoerner(ratio, cds)
    assert np.all(np.diff(values) < 0.0)


def test_vectorized_across_span() -> None:
    """十截面一次性求值：各站位弦长不同 -> 各自 t_te/c 不同。"""
    chords = np.array([0.090, 0.0885, 0.084, 0.0765, 0.0683, 0.0608, 0.0548, 0.051, 0.0486, 0.045])
    t_te = 0.0025
    ratios = t_te / chords
    values = compute_base_drag_hoerner(ratios, np.full(chords.shape, 0.008))
    assert values.shape == (10,)
    assert np.all(values > 0.0)
    # 叶尖弦长最小 -> 相对厚度最大 -> 基底阻力惩罚最重
    assert values[-1] == pytest.approx(float(np.max(values)))


def test_negative_ratio_rejected() -> None:
    with pytest.raises(ValueError, match="不得为负"):
        compute_base_drag_hoerner(-0.01, 0.01)


def test_apply_base_drag_adds_to_total() -> None:
    cd_profile = 0.01
    total = apply_base_drag(cd_profile, 0.02)
    assert total == pytest.approx(cd_profile + compute_base_drag_hoerner(0.02, cd_profile))
    assert total > cd_profile
