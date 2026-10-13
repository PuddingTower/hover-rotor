"""Task 1.1 验收：钝尾缘 CST 型线发生器的厚度、光顺性与几何完备性。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hover_rotor.geometry.cst import (
    fit_shape_weights,
    generate_flatback_cst,
    relative_te_thickness,
    trailing_edge_gap,
)

N_PTS = 200


def naca_half_thickness(psi: np.ndarray, t: float = 0.12) -> np.ndarray:
    """NACA 4 位数模半厚度分布（尖后缘系数 -0.1036）。"""
    psi = np.clip(psi, 0.0, 1.0)
    return (
        5.0 * t * (0.2969 * np.sqrt(psi) - 0.1260 * psi - 0.3516 * psi**2 + 0.2843 * psi**3 - 0.1036 * psi**4)
    )


def naca_weights(t: float = 0.12, dz_te: float = 0.0, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """把 NACA 对称翼型反算进 CST 参数空间，同时承载钝尾缘厚度项。

    传入反算的 zeta 必须是**已含**尾缘楔形贡献 psi*dz_te/2 的目标纵坐标，
    否则 (zeta - psi*dz_te/2)/C(psi) 在 psi->1 处以非零分子除以趋于零的类函数，
    反算目标奇异发散、还原权重完全失真。
    """
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, 161)))
    yt = naca_half_thickness(psi, t)
    wedge = psi * dz_te / 2.0
    au = fit_shape_weights(psi, yt + wedge, n, dz_te=dz_te)
    al = fit_shape_weights(psi, -(yt + wedge), n, dz_te=-dz_te)
    return au, al


def split_surfaces(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按前缘分割闭合回路，返回 (下表面, 上表面)，均按 psi 升序。"""
    i_le = int(np.argmin(pts[:, 0]))
    lower = pts[: i_le + 1][::-1]
    upper = pts[i_le + 1 :]
    return lower, upper


def _ccw(a, b, c) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, p3, p4) -> bool:
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


def is_simple_polygon(pts: np.ndarray) -> bool:
    """闭合多边形是否存在非相邻边相交（自相交型面无法生成刀轨）。"""
    n = pts.shape[0]
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if segments_intersect(a, b, c, d):
                return False
    return True


@pytest.mark.parametrize("dz_te", [0.002, 0.005, 0.01, 0.03, 0.10])
def test_cst_te_thickness(dz_te: float) -> None:
    """计划.md 硬指标：|z_u(1) - z_l(1) - dz_te| < 1e-7。"""
    au, al = naca_weights(dz_te=dz_te)
    pts = generate_flatback_cst(au, al, dz_te, n_points=N_PTS)
    assert abs(trailing_edge_gap(pts) - dz_te) < 1e-7
    assert pts.shape == (N_PTS, 2)


def test_te_corners_are_geometrically_vertical() -> None:
    """尾缘两端必须同处 psi=1 竖直线上，杜绝任何斜切或缝隙。"""
    au, al = naca_weights(dz_te=0.02)
    pts = generate_flatback_cst(au, al, 0.02, n_points=N_PTS)
    assert pts[0, 0] == pytest.approx(1.0, abs=1e-15)
    assert pts[-1, 0] == pytest.approx(1.0, abs=1e-15)
    assert pts[0, 1] == pytest.approx(-0.01, abs=1e-12)
    assert pts[-1, 1] == pytest.approx(0.01, abs=1e-12)


def _max_interior_turn(pts: np.ndarray) -> float:
    """除首尾两个尾缘角点外，回路折线的最大顶点转角。"""
    d = np.diff(pts, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    turn = np.abs(np.diff(ang))
    return float(turn[1:-2].max())


def test_surface_is_c1_continuous_except_te() -> None:
    """一阶导连续性的可操作判据：真实折角（kink）在采样加密后转角不衰减，
    而合法的高曲率区（前缘圆头）转角随弧长步长同步下降。
    故要求采样数翻倍时最大内部转角至少下降 20%。"""
    au, al = naca_weights(dz_te=0.02)
    coarse = _max_interior_turn(generate_flatback_cst(au, al, 0.02, n_points=200))
    fine = _max_interior_turn(generate_flatback_cst(au, al, 0.02, n_points=400))
    assert fine < 0.8 * coarse, f"加密后转角未衰减（{coarse:.3f} -> {fine:.3f}），型线存在折角"
    assert fine < 1.0, "前缘曲率区残差过大，圆头接不合律"


def test_profile_has_no_self_intersection() -> None:
    au, al = naca_weights(dz_te=0.015)
    pts = generate_flatback_cst(au, al, 0.015, n_points=N_PTS)
    assert is_simple_polygon(pts), "闭合型线自相交，无法生成刀轨"


def test_leading_edge_radius_positive() -> None:
    """前缘必须由有限曲率半径圆接，R_le > 0 否则刀具无法切入。"""
    t = 0.12
    au, al = naca_weights(t=t, dz_te=0.02)
    pts = generate_flatback_cst(au, al, 0.02, n_points=N_PTS)
    lower, upper = split_surfaces(pts)

    p0 = pts[int(np.argmin(pts[:, 0]))]
    p1 = upper[1]
    p2 = lower[1]
    a, b, c = np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p0 - p2)
    # NumPy 2.x 移除了二维向量的 np.cross，显式取 z 分量
    area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    r_le = a * b * c / (4.0 * area) if area > 0 else float("inf")
    assert np.isfinite(r_le) and r_le > 0.0
    assert r_le > 0.5 * 1.1019 * t**2, "前缘半径远小于 NACA 理论值，圆头接不合律"


def test_naca0012_flatback_thickness_matches_theory() -> None:
    """NACA0012 等效权重 + 钝尾缘：厚度分布与 (NACA 解析 + 线性尾缘斜坡) 偏差 < 0.1% 弦长。"""
    dz_te = 1.0e-3 / 0.10  # 1 mm 钝尾缘落在 100 mm 弦长上
    au, al = naca_weights(t=0.12, dz_te=dz_te)
    pts = generate_flatback_cst(au, al, dz_te, n_points=401)
    lower, upper = split_surfaces(pts)

    yt_upper = naca_half_thickness(upper[:, 0], 0.12) + upper[:, 0] * dz_te / 2.0
    yt_lower = -(naca_half_thickness(lower[:, 0], 0.12) + lower[:, 0] * dz_te / 2.0)
    assert np.max(np.abs(upper[:, 1] - yt_upper)) < 1e-3
    assert np.max(np.abs(lower[:, 1] - yt_lower)) < 1e-3

    # 最大厚度位置应仍位于半弦附近，钝尾缘不得把厚度峰推向尾缘。
    # 上下表面采样点数不同（前缘点只归下表面），故插值到公共弦向网格后再作差。
    psi_common = np.linspace(0.005, 1.0, 200)
    z_u = np.interp(psi_common, upper[:, 0], upper[:, 1])
    z_l = np.interp(psi_common, lower[:, 0], lower[:, 1])
    x_tmax = psi_common[int(np.argmax(z_u - z_l))]
    assert 0.25 < x_tmax < 0.38


def test_round_te_cap_arc_length() -> None:
    """半圆封头弧长趋近 pi * dz_te / 2（直径即尾缘厚度）。"""
    dz_te = 0.02
    au, al = naca_weights(dz_te=dz_te)
    pts = generate_flatback_cst(au, al, dz_te, n_points=200, te_style="round", te_arc_points=64)

    arc = pts[pts[:, 0] > 1.0 + 1e-15]
    assert arc.size > 0
    # 封头自上角点 (1, +dz/2) 绕至下角点 (1, -dz/2)，两端各以短弦接入
    path = np.vstack(([[1.0, 0.5 * dz_te]], arc, [[1.0, -0.5 * dz_te]]))
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    assert length == pytest.approx(math.pi * dz_te / 2.0, rel=5e-3)
    assert pts[:, 0].max() == pytest.approx(1.0 + 0.5 * dz_te, rel=1e-3)


def test_physical_chord_scaling() -> None:
    chord = 0.09
    t_te = 0.0015
    dz = relative_te_thickness(t_te, chord)
    au, al = naca_weights(dz_te=dz)
    pts = generate_flatback_cst(au, al, dz, n_points=N_PTS, chord=chord)
    assert trailing_edge_gap(pts) == pytest.approx(t_te, abs=1e-9)
    assert pts[:, 0].max() == pytest.approx(chord, abs=1e-12)


@pytest.mark.parametrize("bad", [0.0, -0.01])
def test_sharp_or_negative_te_rejected(bad: float) -> None:
    au, al = naca_weights()
    with pytest.raises(ValueError, match="尾缘厚度"):
        generate_flatback_cst(au, al, bad)


def test_weight_vectors_must_match() -> None:
    with pytest.raises(ValueError, match="等长一维向量"):
        generate_flatback_cst(np.zeros(6), np.zeros(9), 0.02)


def test_unknown_te_style_rejected() -> None:
    au, al = naca_weights()
    with pytest.raises(ValueError, match="te_style"):
        generate_flatback_cst(au, al, 0.02, te_style="wedge")


def test_sampling_below_minimum_rejected() -> None:
    """采样过稀无法分辨前缘曲率，直接拒绝而不是悄悄返回低保真点阵。"""
    au, al = naca_weights()
    with pytest.raises(ValueError, match="采样点数"):
        generate_flatback_cst(au, al, 0.02, n_points=19)
    assert generate_flatback_cst(au, al, 0.02, n_points=20).shape == (20, 2)
