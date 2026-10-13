"""Task 1.2 验收：余弦聚点分布律与 PCHIP 单调保形插值。"""

from __future__ import annotations

import numpy as np
import pytest

from hover_rotor.geometry.discretization import (
    build_span_distribution,
    cosine_span_positions,
    discretize_rotor_span,
    interpolate_span,
)

# §4.1 金标准站位数组（D-05 对称余弦聚点在 r_hub=0.15, N=10 下的直接计算值）
GOLDEN = np.array([0.1500, 0.1756, 0.2494, 0.3625, 0.5012, 0.6488, 0.7875, 0.9006, 0.9744, 1.0000])


def test_positions_match_golden_table() -> None:
    pos = cosine_span_positions(0.15, 10)
    assert pos.shape == (10,)
    assert np.max(np.abs(pos - GOLDEN)) < 5e-4


def test_endpoints_are_exact() -> None:
    """物理端点必须严格落在桨毂面与桨尖，不得因浮点漂移越出叶片。"""
    for hub in (0.05, 0.15, 0.20):
        pos = cosine_span_positions(hub, 10)
        assert pos[0] == pytest.approx(hub, abs=0.0)
        assert pos[-1] == pytest.approx(1.0, abs=0.0)

    r = discretize_rotor_span(0.75, 0.15, 10)
    assert r[0] == pytest.approx(0.75 * 0.15, abs=0.0)
    assert r[-1] == pytest.approx(0.75, abs=0.0)


def test_positions_strictly_increasing() -> None:
    pos = cosine_span_positions(0.15, 10)
    assert np.all(np.diff(pos) > 0.0)


def test_both_ends_cluster_at_least_twice_mid_span_density() -> None:
    """叶根与叶尖两端聚点密度均不低于中段最大间距的两倍。"""
    pos = cosine_span_positions(0.15, 10)
    gap = np.diff(pos)
    mid_max = gap.max()
    hub_band = gap[:3].mean()
    tip_band = gap[-3:].mean()
    assert mid_max == pytest.approx(gap[4], rel=1e-2), "最大间距应位于中段"
    assert hub_band < 0.5 * mid_max
    assert tip_band < 0.5 * mid_max


@pytest.mark.parametrize("n", [4, 10, 21])
def test_distribution_refines_toward_both_ends(n: int) -> None:
    pos = cosine_span_positions(0.15, n)
    assert pos[0] == pytest.approx(0.15)
    assert pos[-1] == 1.0
    gap = np.diff(pos)
    assert gap.mean() < 2.0 * gap[n // 2 - 1]  # 两端平均密于中段


def test_pchip_preserves_monotonic_chord_taper() -> None:
    """单调递减的弦长控制点插值后必须处处单调不增且无过冲——弦长过冲等于制造废品。"""
    r_ctrl = np.array([0.15, 0.40, 0.70, 1.00])
    c_ctrl = np.array([0.120, 0.102, 0.081, 0.060]) * 0.75  # 物理弦长 [m]
    r_eval = np.linspace(0.15, 1.0, 201)

    c = interpolate_span(r_ctrl, c_ctrl, r_eval)
    assert np.all(np.diff(c) <= 1e-15), "PCHIP 插值出现非单调段"
    assert c.max() <= c_ctrl.max() + 1e-15
    assert c.min() >= c_ctrl.min() - 1e-15
    assert c[0] == pytest.approx(c_ctrl[0])
    assert c[-1] == pytest.approx(c_ctrl[-1])


def test_interpolation_rejects_unsorted_control_nodes() -> None:
    with pytest.raises(ValueError, match="严格单调递增"):
        interpolate_span(
            np.array([0.15, 0.70, 0.40, 1.0]), np.array([0.1, 0.08, 0.09, 0.06]), np.array([0.5])
        )


def test_interpolation_rejects_out_of_range_eval() -> None:
    """站位超出控制节点覆盖范围属于建模错误，必须显式拒绝而非线性外插。"""
    with pytest.raises(ValueError, match="超出控制节点范围"):
        interpolate_span(np.array([0.2, 0.5, 1.0]), np.array([0.1, 0.08, 0.06]), np.array([0.15]))


def test_invalid_hub_ratio_and_radius_rejected() -> None:
    with pytest.raises(ValueError, match="轮毂比"):
        cosine_span_positions(0.0, 10)
    with pytest.raises(ValueError, match="轮毂比"):
        cosine_span_positions(1.0, 10)
    with pytest.raises(ValueError, match="旋翼半径"):
        discretize_rotor_span(-0.5)


def test_build_span_distribution_unit_contract() -> None:
    """chord_ctrl 为物理米、r_ctrl 为无量纲站位；叶尖站位弦长必须回到控制点末值。"""
    radius = 0.75
    dist = build_span_distribution(
        radius,
        r_ctrl=np.array([0.15, 0.40, 0.70, 1.00]),
        chord_ctrl=np.array([0.120, 0.102, 0.081, 0.060]) * radius,
        theta_ctrl_deg=np.array([22.5, 12.9, 6.4, 0.5]),
    )
    assert dist.n_sections == 10
    assert dist.r[-1] == pytest.approx(radius)
    assert dist.chord[-1] == pytest.approx(0.060 * radius, rel=1e-9)
    assert dist.theta_deg[-1] == pytest.approx(0.5, abs=1e-9)
    assert np.all(np.diff(dist.theta_deg) < 0.0), "安装角应沿展向连续卸载"

    props = dist.section_props()
    assert props[0]["index"] == 1 and props[-1]["r_over_R"] == pytest.approx(1.0)
