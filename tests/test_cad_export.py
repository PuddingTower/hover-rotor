"""Task 4.1 / 4.2 验收：闭合 DXF/DAT/CSV 二维图纸与放样 STEP/STL 三维实体。

判定准则逐条取自计划.md：DXF 回读 LWPOLYLINE 200 顶点且 closed 为真、首尾封闭段
即无间隙的竖直钝尾缘；.dat 首末点 y 向间距 = t_te（1e-6 mm）；依赖缺席时抛
OptionalDependencyError（D-09）；STEP 经 OCP 重导入 BRepCheck 无缺陷、单次闭合、
无非流形边；STL 水密可定向流形（每条有向边恰出现一次、欧拉示性数 2）；
最外截面位于 r = R。
"""

from __future__ import annotations

import math
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hover_rotor.cad.dxf_exporter import export_sections_dxf
from hover_rotor.cad.step_builder import blade_solid, build_blade_step
from hover_rotor.config import OptionalDependencyError, RotorInputs, RotorResult, SectionResult
from hover_rotor.geometry.cst import generate_flatback_cst
from hover_rotor.geometry.discretization import cosine_span_positions

N_PTS = 200

# §4.1 参考表：相对弦长与典型扭转角（非约束，用于合成自洽档案）
CHORD_OVER_R = [0.120, 0.118, 0.112, 0.102, 0.091, 0.081, 0.073, 0.068, 0.064, 0.060]
THETA_DEG = [22.5, 20.8, 17.1, 12.9, 9.2, 6.4, 4.1, 2.5, 1.3, 0.5]


def _rotate(points: np.ndarray, theta_deg: float) -> np.ndarray:
    a = math.radians(theta_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return points @ rot.T


def _make_section(
    idx: int,
    r_over_R: float,
    radius: float,
    chord_over_R: float,
    theta_deg: float,
    au: np.ndarray,
    al: np.ndarray,
    te_thickness: float,
) -> SectionResult:
    chord = chord_over_R * radius
    dz_te = te_thickness / chord
    pts_norm = generate_flatback_cst(au, al, dz_te, n_points=N_PTS)
    chord_mm = chord * 1000.0
    pts_mm = _rotate(pts_norm * chord_mm, theta_deg)
    ma = 0.55 * r_over_R
    alpha = theta_deg - 2.0 - 3.0 * r_over_R
    return SectionResult(
        index=idx,
        r_over_R=r_over_R,
        r=r_over_R * radius,
        chord=chord,
        theta_deg=theta_deg,
        te_thickness=te_thickness,
        dz_te=dz_te,
        alpha_deg=alpha,
        cl=min(1.4, 0.11 * alpha) if alpha > 0 else 0.0,
        cd=0.008 + 0.004 * (1.0 - r_over_R),
        cm=-0.02,
        re=2.0e5 + 5.0e5 * r_over_R,
        ma=ma,
        au=tuple(float(v) for v in au),
        al=tuple(float(v) for v in al),
        points_normalized=pts_norm,
        points_mm=pts_mm,
        dN=10.0 * r_over_R,
        dQ=0.4 * r_over_R,
        v_i=3.0 + r_over_R,
        phi_deg=math.degrees(math.atan2(3.5, 0.55 * r_over_R * 340.3 / 0.55)),
    )


@pytest.fixture(scope="session")
def synthetic_result(naca_to_cst) -> RotorResult:
    """10 截面合成档案：NACA0012 等效 CST 权重 + §4.1 参考展向分布。

    与 conftest.baseline_inputs 同一基准工况；此处直接构造以便 session 作用域
    供 STEP/STL 昂贵放样复用。
    """
    inputs = RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.0025)
    au, al = naca_to_cst("0012")
    rbar = cosine_span_positions(inputs.hub_ratio, 10)
    sections = [
        _make_section(i, rb, inputs.radius, c, th, au, al, inputs.te_thickness)
        for i, (rb, c, th) in enumerate(zip(rbar, CHORD_OVER_R, THETA_DEG, strict=True))
    ]
    omega = inputs.tip_mach_max * 0.85 * inputs.a_inf / inputs.radius
    chord_vec = np.array([s.chord for s in sections])
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
        chord=chord_vec,
        theta_deg=np.array(THETA_DEG),
        v_i=np.array([s.v_i for s in sections]),
        circulation=np.array([s.cl * s.chord for s in sections]),
        cl_distribution=np.array([s.cl for s in sections]),
        seed=42,
    )


@pytest.fixture(scope="session")
def synthetic_sections(synthetic_result) -> list[SectionResult]:
    return synthetic_result.sections


# --------------------------------------------------------------------------------------
# Task 4.1 — DXF / DAT / CSV
# --------------------------------------------------------------------------------------


def test_dxf_lwpolyline_closed_200_vertices(tmp_path: Path, synthetic_sections) -> None:
    """回读断言：单一 LWPOLYLINE、200 顶点、closed 为真、首尾封闭段即竖直钝尾缘。"""
    import ezdxf

    files = export_sections_dxf(synthetic_sections, tmp_path)
    assert len(files) == 3 * len(synthetic_sections)
    for pos, sec in enumerate(synthetic_sections, start=1):
        dxf_path = tmp_path / f"sec{pos:02d}_contour.dxf"
        assert dxf_path.exists()
        assert dxf_path in files
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        polylines = msp.query("LWPOLYLINE")
        assert len(polylines) == 1, f"{dxf_path.name} 应只有一条轮廓多段线"
        poly = polylines[0]
        pts = list(poly.get_points())
        assert len(pts) == N_PTS
        assert poly.closed is True
        first, last = pts[0], pts[-1]
        te_mm = sec.te_thickness * 1000.0
        # 钝尾缘封闭段：首末点同弦向位置（竖直、无缝隙），间距恰为 t_te
        assert abs(first[0] - last[0]) < 1e-9
        assert abs(abs(last[1] - first[1]) - te_mm) < 1e-9
        chord_mm = sec.chord * 1000.0
        assert first[0] == pytest.approx(chord_mm, abs=1e-9)


def test_dat_selig_order_and_te_gap(tmp_path: Path, synthetic_sections) -> None:
    """.dat 为 Selig 遍历序（上表面 TE->LE 后下表面 LE->TE），首末点 y 距 = t_te。"""
    export_sections_dxf(synthetic_sections, tmp_path)
    for pos, sec in enumerate(synthetic_sections, start=1):
        dat_path = tmp_path / f"sec{pos:02d}_contour.dat"
        assert dat_path.exists()
        lines = dat_path.read_text(encoding="ascii").splitlines()
        coords = np.array([[float(v) for v in ln.split()] for ln in lines[1:]])
        assert coords.shape == (N_PTS, 2)
        # 首点 = 上尾缘角（y>0），末点 = 下尾缘角（y<0），均在 x = 弦长处
        assert coords[0, 1] > 0.0 > coords[-1, 1]
        te_mm = sec.te_thickness * 1000.0
        assert abs((coords[0, 1] - coords[-1, 1]) - te_mm) < 1e-6
        assert coords[0, 0] == pytest.approx(sec.chord * 1000.0, abs=1e-6)
        # Selig 序：x 自弦长单调降至前缘（上表面），再单调升回尾缘（下表面）
        i_le = int(np.argmin(coords[:, 0]))
        assert 0 < i_le < N_PTS - 1
        assert np.all(np.diff(coords[: i_le + 1, 0]) < 0.0)
        assert np.all(np.diff(coords[i_le:, 0]) > 0.0)


def test_csv_columns_and_spatial_coordinates(tmp_path: Path, synthetic_sections) -> None:
    """CSV 含归一化 + 弦对齐 + 含扭转绝对空间毫米坐标与截面气动量。"""
    import csv as _csv

    export_sections_dxf(synthetic_sections, tmp_path)
    for pos, sec in enumerate(synthetic_sections, start=1):
        with (tmp_path / f"sec{pos:02d}.csv").open(newline="", encoding="ascii") as fh:
            rows = list(_csv.DictReader(fh))
        assert len(rows) == N_PTS
        required = {
            "psi",
            "zeta",
            "x_chord_mm",
            "y_chord_mm",
            "x_mm",
            "y_mm",
            "r_over_R",
            "chord_mm",
            "theta_deg",
            "alpha_deg",
            "cl",
            "cd",
            "cm",
            "re",
            "ma",
            "te_thickness_mm",
        }
        assert required <= set(rows[0])
        spatial = np.array([[float(r["x_mm"]), float(r["y_mm"])] for r in rows])
        assert np.max(np.abs(spatial - sec.points_mm)) < 5e-5  # %.6f 落盘精度


def test_dxf_raises_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synthetic_sections
) -> None:
    """ezdxf 缺席必须抛 OptionalDependencyError（D-09），而非 ImportError。"""
    monkeypatch.setitem(sys.modules, "ezdxf", None)
    with pytest.raises(OptionalDependencyError, match="ezdxf"):
        export_sections_dxf(synthetic_sections, tmp_path)


def test_step_raises_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synthetic_sections
) -> None:
    monkeypatch.setitem(sys.modules, "build123d", None)
    with pytest.raises(OptionalDependencyError, match="build123d"):
        build_blade_step(synthetic_sections, tmp_path)


# --------------------------------------------------------------------------------------
# Task 4.2 — STEP / STL
# 实测：10 截面 x200 点放样 + STEP/STL 落盘 + OCP 重校验整组 <10 s，
# 未达 slow 门禁量级（对照 DE 全局搜索），故不加 @pytest.mark.slow。
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def blade_artifacts(synthetic_sections, tmp_path_factory: pytest.TempPathFactory) -> dict:
    solid = blade_solid(synthetic_sections)
    step_path, stl_path = build_blade_step(synthetic_sections, tmp_path_factory.mktemp("blade"))
    return {"solid": solid, "step": step_path, "stl": stl_path}


def _read_binary_stl(path: Path) -> np.ndarray:
    """纯 Python 读取二进制 STL，返回 (F, 3, 3) 三角形顶点阵（毫米）。"""
    raw = path.read_bytes()
    assert raw[:5] != b"solid", "应为二进制 STL"
    (n_tri,) = struct.unpack("<I", raw[80:84])
    assert len(raw) == 84 + 50 * n_tri
    tris = np.empty((n_tri, 3, 3), dtype=np.float64)
    for i in range(n_tri):
        vals = struct.unpack("<12fH", raw[84 + 50 * i : 84 + 50 * i + 50])
        tris[i] = np.array([vals[3:6], vals[6:9], vals[9:12]], dtype=np.float32).astype(np.float64)
    return tris


def _mesh_key(p: np.ndarray) -> tuple[int, int, int]:
    # 0.1 微米量化栅格：吸收 float32 落盘舍入（750 mm 处 ulp≈4.5e-5 mm），
    # 又远小于尾缘角点处最小网格特征（~5 微米），不会误并相异顶点
    return tuple(np.rint(np.asarray(p, dtype=np.float64) * 1e4).astype(np.int64))  # type: ignore[return-value]


def test_blade_solid_is_valid_closed_manifold(blade_artifacts) -> None:
    """STEP 级实体判定：BRepCheck 无缺陷、单实体单闭合壳、无非流形边。"""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher,
        IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher,
    )
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS

    solid = blade_artifacts["solid"]
    shape = solid.wrapped
    assert BRepCheck_Analyzer(shape).IsValid()
    assert solid.is_valid

    solids = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, TopAbs_SOLID, solids)
    shells = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, TopAbs_SHELL, shells)
    assert solids.Extent() == 1 and shells.Extent() == 1
    assert TopoDS.Shell(shells.FindKey(1)).Closed()  # 壳单次闭合（IsClosed）

    edge_faces = IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_faces)
    assert edge_faces.Extent() > 0
    non_manifold = [i for i in range(1, edge_faces.Extent() + 1) if edge_faces.FindFromIndex(i).Extent() != 2]
    assert not non_manifold, f"存在非流形边：{len(non_manifold)} 条"
    assert solid.volume > 0.0


def test_step_reimported_is_valid(blade_artifacts) -> None:
    """STEP 文件经 OpenCASCADE 读取器重导入后 BRepCheck_Analyzer.IsValid()。"""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.STEPControl import STEPControl_Reader

    step_path: Path = blade_artifacts["step"]
    assert step_path.exists() and step_path.stat().st_size > 10_000
    header = step_path.read_bytes()[:4096].decode("ascii", errors="replace")
    assert "AUTOMOTIVE_DESIGN" in header, "STEP 应落盘 AP214 协议"

    reader = STEPControl_Reader()
    assert reader.ReadFile(str(step_path)) == 1
    reader.TransferRoots()
    shape = reader.OneShape()
    assert BRepCheck_Analyzer(shape).IsValid()


def test_outboard_section_sits_at_rotor_radius(blade_artifacts, synthetic_result) -> None:
    """几何自洽红线：最外截面（叶尖）位于 z = R，叶根截面位于 r_hub*R。"""
    solid = blade_artifacts["solid"]
    bb = solid.bounding_box()
    r_tip = synthetic_result.sections[-1].r * 1000.0
    r_root = synthetic_result.sections[0].r * 1000.0
    assert r_tip == pytest.approx(synthetic_result.inputs.radius * 1000.0, abs=1e-9)
    assert pytest.approx(r_tip, abs=1e-2) == bb.max.Z
    assert pytest.approx(r_root, abs=1e-2) == bb.min.Z


def test_stl_is_watertight_orientable_manifold(blade_artifacts) -> None:
    """STL 水密流形判定：每条无向边恰被 2 个三角面共享且以相反方向经过、
    零边界边、无退化三角形、闭合零亏格曲面欧拉示性数 V-E+F=2。"""
    tris = _read_binary_stl(blade_artifacts["stl"])
    n_tri = tris.shape[0]
    assert n_tri > 100

    directed: Counter = Counter()
    vertices: set = set()
    degenerate = 0
    for tri in tris:
        a, b, c = tri
        u, v = b - a, c - a
        n = np.cross(u, v)
        if np.linalg.norm(n) < 1e-9:
            degenerate += 1
        vertices.update(_mesh_key(p) for p in tri)
        for j in range(3):
            directed[(_mesh_key(tri[j]), _mesh_key(tri[(j + 1) % 3]))] += 1
    assert degenerate == 0, "存在退化三角形（破面）"

    undirected: Counter = Counter()
    for (p, q), count in directed.items():
        undirected[tuple(sorted((p, q)))] += count
        assert count == 1, "同一有向边重复出现，绕序不一致"
    assert all(count == 2 for count in undirected.values()), "存在非流形边"

    n_edge = len(undirected)
    euler = len(vertices) - n_edge + n_tri
    assert euler == 2, f"欧拉示性数 {euler} != 2（零亏格闭合曲面要求）"

    # 水密性 + 绕序一致的直接推论：散度定理网格体积与精确实体体积自洽
    mesh_volume = float(np.einsum("fi,fi->", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0)
    assert mesh_volume > 0.0, "法向应一致朝外（有符号体积为正）"
    assert mesh_volume == pytest.approx(blade_artifacts["solid"].volume, rel=0.02)
