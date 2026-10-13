"""Task 4.3 验收：制造报表三件套与截面结构力学量。

判定准则：manifest CSV 字段完整覆盖 §4.3 全部几何/气动/结构量；
rotor_report.json 可经 RotorResult.from_dict 无损反序列化（往返等价）；
NACA0012 解析校验——多边形面积分给出的 A 与 I_xx 对解析厚度分布的高分辨率
求积参考偏差 < 1%；J（单闭室薄壁 Bredt）对加密采样参考自洽收敛。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson

from hover_rotor.cad.table_export import (
    MANIFEST_COLUMNS,
    export_manufacturing_tables,
    polygon_structural_properties,
)
from hover_rotor.config import RotorInputs, RotorResult, SectionResult
from hover_rotor.geometry.cst import generate_flatback_cst
from hover_rotor.geometry.discretization import cosine_span_positions

N_PTS = 200
CHORD_OVER_R = [0.120, 0.118, 0.112, 0.102, 0.091, 0.081, 0.073, 0.068, 0.064, 0.060]
THETA_DEG = [22.5, 20.8, 17.1, 12.9, 9.2, 6.4, 4.1, 2.5, 1.3, 0.5]


def _rotate(points: np.ndarray, theta_deg: float) -> np.ndarray:
    a = math.radians(theta_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return points @ rot.T


def _make_section(idx, r_over_R, radius, chord_over_R, theta_deg, au, al, te) -> SectionResult:
    chord = chord_over_R * radius
    dz_te = te / chord
    pts_norm = generate_flatback_cst(au, al, dz_te, n_points=N_PTS)
    return SectionResult(
        index=idx,
        r_over_R=r_over_R,
        r=r_over_R * radius,
        chord=chord,
        theta_deg=theta_deg,
        te_thickness=te,
        dz_te=dz_te,
        alpha_deg=theta_deg - 2.0 - 3.0 * r_over_R,
        cl=min(1.4, max(0.0, 0.11 * (theta_deg - 2.0 - 3.0 * r_over_R))),
        cd=0.008 + 0.004 * (1.0 - r_over_R),
        cm=-0.02,
        re=2.0e5 + 5.0e5 * r_over_R,
        ma=0.55 * r_over_R,
        au=tuple(float(v) for v in au),
        al=tuple(float(v) for v in al),
        points_normalized=pts_norm,
        points_mm=_rotate(pts_norm * (chord * 1000.0), theta_deg),
        dN=10.0 * r_over_R,
        dQ=0.4 * r_over_R,
        v_i=3.0 + r_over_R,
        phi_deg=5.0,
    )


@pytest.fixture(scope="session")
def synthetic_result(naca_to_cst) -> RotorResult:
    inputs = RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.0025)
    au, al = naca_to_cst("0012")
    rbar = cosine_span_positions(inputs.hub_ratio, 10)
    sections = [
        _make_section(i, rb, inputs.radius, c, th, au, al, inputs.te_thickness)
        for i, (rb, c, th) in enumerate(zip(rbar, CHORD_OVER_R, THETA_DEG, strict=True))
    ]
    omega = inputs.tip_mach_max * 0.85 * inputs.a_inf / inputs.radius
    return RotorResult(
        inputs=inputs,
        omega=omega,
        rpm=math.degrees(omega),
        tip_mach=0.85 * inputs.tip_mach_max,
        thrust=inputs.thrust * 1.002,
        power=105.0,
        torque=105.0 / omega,
        fm=0.712,
        ct=0.0081,
        cp=0.0106,
        power_ideal=74.8,
        sections=sections,
        r_over_R=rbar,
        chord=np.array([s.chord for s in sections]),
        theta_deg=np.array(THETA_DEG),
        v_i=np.array([s.v_i for s in sections]),
        circulation=np.array([s.cl * s.chord for s in sections]),
        cl_distribution=np.array([s.cl for s in sections]),
        seed=42,
    )


# --------------------------------------------------------------------------------------
# NACA0012 解析校验（判定准则核心）
# --------------------------------------------------------------------------------------


def naca0012_half_thickness(psi: np.ndarray) -> np.ndarray:
    """NACA0012 解析半厚度分布（系数 -0.101516，与 conftest 基准面一致）。"""
    psi = np.clip(psi, 0.0, 1.0)
    return (
        5.0
        * 0.12
        * (0.2969 * np.sqrt(psi) - 0.1260 * psi - 0.3516 * psi**2 + 0.2843 * psi**3 - 0.101516 * psi**4)
    )


def test_naca0012_area_and_ixx_within_one_percent(naca_to_cst) -> None:
    """多边形面积分的 A 与 I_xx 对比解析厚度分布的高分辨率求积，偏差 < 1%。

    参考值不硬编码：对称剖面 A = c^2 * 2∫yt dpsi、I_xx = c^4 * (2/3)∫yt^3 dpsi，
    由 20 万点复合 Simpson 在解析分布上现算。
    """
    au, al = naca_to_cst("0012")
    chord_mm = 100.0
    dz_te = 0.001  # conftest 权重内嵌的微钝尾缘，对 A/I_xx 的影响 << 1%
    pts = generate_flatback_cst(au, al, dz_te, n_points=N_PTS, chord=chord_mm)
    props = polygon_structural_properties(pts)

    psi = np.linspace(0.0, 1.0, 200_001)
    yt = naca0012_half_thickness(psi)
    area_ref = chord_mm**2 * 2.0 * simpson(yt, x=psi)
    i_xx_ref = chord_mm**4 * (2.0 / 3.0) * simpson(yt**3, x=psi)

    assert props.area_mm2 == pytest.approx(area_ref, rel=0.01)
    assert props.i_xx_mm4 == pytest.approx(i_xx_ref, rel=0.01)
    assert abs(props.y_cg_mm) < 0.05  # 对称剖面形心几乎落在弦线上
    assert 0.3 * chord_mm < props.x_cg_mm < 0.6 * chord_mm
    assert props.j_torsion_mm4 > 0.0


def test_bredt_torsion_converges_with_sampling(naca_to_cst) -> None:
    """J 的 200 点求积对 2000 点参考收敛，确认前缘可积奇异性处理正确。

    前缘 t ~ sqrt(psi) 使 1/t 求积仅 O(h^{1/2}) 慢收敛（制造端 200 点定义
    即权威几何），故容差取 5% 而非通常的二阶收敛带。
    """
    au, al = naca_to_cst("0012")
    coarse = polygon_structural_properties(generate_flatback_cst(au, al, 0.001, n_points=200, chord=100.0))
    fine = polygon_structural_properties(generate_flatback_cst(au, al, 0.001, n_points=2000, chord=100.0))
    assert coarse.j_torsion_mm4 == pytest.approx(fine.j_torsion_mm4, rel=0.05)
    # 内接多边形对凸弧的面积低估随弦长平方收缩：200 点带 0.2% 量级残差属几何必然
    assert coarse.area_mm2 == pytest.approx(fine.area_mm2, rel=5e-3)


# --------------------------------------------------------------------------------------
# export_manufacturing_tables 三件套
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def export_dir(tmp_path_factory: pytest.TempPathFactory, synthetic_result) -> tuple[Path, RotorResult]:
    out = tmp_path_factory.mktemp("tables")
    enriched = export_manufacturing_tables(synthetic_result, out)
    return out, enriched


def test_manifest_csv_field_coverage(export_dir) -> None:
    out, enriched = export_dir
    manifest = out / "manufacturing_manifest.csv"
    assert manifest.exists()
    with manifest.open(newline="", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    assert set(MANIFEST_COLUMNS) <= set(rows[0])
    required = {
        "section_no",
        "r_over_R",
        "radius_mm",
        "chord_mm",
        "theta_deg",
        "te_thickness_mm",
        "re",
        "ma",
        "alpha_deg",
        "ld",
        "area_mm2",
        "x_cg_mm",
        "y_cg_mm",
        "i_xx_mm4",
        "j_torsion_mm4",
    }
    assert required <= set(rows[0])
    assert len(rows) == 10
    # CSV 落盘值与补齐后的截面档案自洽
    for row, sec in zip(rows, enriched.sections, strict=True):
        assert float(row["area_mm2"]) == pytest.approx(sec.area, rel=1e-6)
        assert float(row["i_xx_mm4"]) == pytest.approx(sec.i_xx, rel=1e-6)
        assert float(row["j_torsion_mm4"]) == pytest.approx(sec.j_torsion, rel=1e-6)
        assert float(row["radius_mm"]) == pytest.approx(sec.r * 1000.0, rel=1e-6)
        assert float(row["te_thickness_mm"]) == pytest.approx(sec.te_thickness * 1000.0, rel=1e-6)
        assert float(row["ld"]) == pytest.approx(sec.ld, rel=1e-6)


def test_json_roundtrip_is_lossless(export_dir) -> None:
    out, enriched = export_dir
    payload = json.loads((out / "rotor_report.json").read_text(encoding="utf-8"))
    restored = RotorResult.from_dict(payload)
    assert restored.to_dict() == enriched.to_dict()
    assert restored.n_sections == 10
    assert restored.sections[0].area == pytest.approx(enriched.sections[0].area)
    assert restored.seed == 42


def test_markdown_report_present(export_dir) -> None:
    out, _ = export_dir
    md = (out / "manufacturing_report.md").read_text(encoding="utf-8")
    assert "# 悬停旋翼制造数据报告" in md
    assert "FM = 0.7120" in md
    assert "I_xx [mm^4]" in md
    assert md.count("| 01 |") == 1 and md.count("| 10 |") == 1


def test_export_does_not_mutate_input(synthetic_result, tmp_path: Path) -> None:
    """导出返回补齐副本，输入 RotorResult 的截面结构字段保持零值不被改写。"""
    assert synthetic_result.sections[0].area == 0.0
    enriched = export_manufacturing_tables(synthetic_result, tmp_path)
    assert enriched.sections[0].area > 0.0
    assert synthetic_result.sections[0].area == 0.0


def test_polygon_input_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="点阵"):
        polygon_structural_properties(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="点阵"):
        polygon_structural_properties(np.zeros((50, 3)))
