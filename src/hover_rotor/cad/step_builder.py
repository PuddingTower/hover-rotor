"""钝尾缘叶片三维放样实体与 STEP/STL 导出器（计划.md Task 4.2，决策 D-09）。

数控多轴铣削与模具加工需要的是保形实体而非点云：本模块把十个展向截面按其
真实径向站位堆叠，用非直纹（``ruled=False``）B 样条放样生成光滑蒙皮，再现
优化器反演出的连续扭转/弦长规律；钝尾缘两角点间的直线段随截面一同放样，
自动生成封闭的平整尾缘带，实体在拓扑上单次闭合（watertight manifold），
可被 OpenCASCADE 校验器与 3D 打印切片软件直接消费。

布置约定（导出边界，单位毫米；内部 SI 见 config.py）：
  * 展向（径向）沿全局 +Z，站位 z = r * 1000；
  * 剖面位于垂直于 Z 的平面，全局 X 为 0 扭转时的弦线方向；
  * 截面二维点直接取 ``SectionResult.points_mm`` —— 该字段已含弦长缩放与
    安装角旋转，本模块只做径向平移，不再施加任何缩放（契约见 config.py）。

STEP/STL 均按加工惯例以毫米落盘；AP214 schema 由 OpenCASCADE STEP 写入器
默认输出，满足 §4.2 的 AP214/AP242 要求。ezdxf/build123d 缺席时抛
OptionalDependencyError 而非 ImportError（D-09），核心气动链路不受牵连。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hover_rotor.config import OptionalDependencyError, SectionResult

FloatArray = NDArray[np.float64]

STEP_FILENAME = "rotor_blade_solid.step"
STL_FILENAME = "rotor_blade_solid.stl"
DEFAULT_STL_DEFLECTION_MM = 0.02
DEFAULT_STL_ANGULAR_TOLERANCE_RAD = 0.1


def _require_build123d() -> Any:
    """惰性导入 build123d（OpenCASCADE 内核封装），缺席时按 D-09 降级。"""
    try:
        import build123d
    except ImportError as exc:  # pragma: no cover - 环境相关分支
        raise OptionalDependencyError(
            "STEP/STL 导出需要 build123d，请安装可选依赖：pip install -e '.[step]'"
        ) from exc
    return build123d


def _section_points_mm(section: SectionResult) -> FloatArray:
    pts = np.asarray(section.points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError(f"截面 {section.index} 的 points_mm 需为 (M>=3, 2) 点阵，收到形状 {pts.shape}")
    return pts


def section_wires(sections: list[SectionResult] | tuple[SectionResult, ...]) -> list[Any]:
    """各截面在叶片全局坐标系（毫米）中的空间闭合导线，z = 径向站位。

    导线按 CST 回路逐点连直段并首尾封闭：末点（尾缘上角）回到首点（尾缘下角）
    的那一段即钝尾缘垂直封闭段，放样后形成尾缘平面带。
    """
    b3d = _require_build123d()
    wires = []
    for section in sections:
        pts = _section_points_mm(section)
        z = float(section.r) * 1000.0
        verts = [b3d.Vector(float(x), float(y), z) for x, y in pts]
        edges = [b3d.Edge.make_line(verts[i], verts[(i + 1) % len(verts)]) for i in range(len(verts))]
        wires.append(b3d.Wire(edges))
    return wires


def blade_solid(sections: list[SectionResult] | tuple[SectionResult, ...], *, ruled: bool = False) -> Any:
    """由截面档案放样出单个闭合叶片实体（build123d ``Part``，单位毫米）。

    ``ruled=False`` 走 BRepOffsetAPI_ThruSections 的 B 样条蒙皮，再现 §3 反设计的
    光滑展向扭转规律；直纹放样（ruled=True）仅作调试对照。
    """
    b3d = _require_build123d()
    if len(sections) < 2:
        raise ValueError(f"放样至少需要 2 个截面，收到 {len(sections)}")
    faces = [b3d.make_face(wire) for wire in section_wires(sections)]
    return b3d.loft(faces, ruled=ruled)


def build_blade_step(
    sections: list[SectionResult] | tuple[SectionResult, ...],
    output_dir: str | Path = "output",
    *,
    step_filename: str = STEP_FILENAME,
    stl_filename: str = STL_FILENAME,
    stl_deflection_mm: float = DEFAULT_STL_DEFLECTION_MM,
    stl_angular_tolerance_rad: float = DEFAULT_STL_ANGULAR_TOLERANCE_RAD,
    ruled: bool = False,
) -> tuple[Path, Path]:
    """放样叶片实体并落盘 STEP（AP214）与 STL，返回两文件路径。

    Args:
        sections: 叶根至叶尖的截面档案，``points_mm`` 已含弦长缩放与扭转。
        output_dir: 输出目录，不存在则创建。
        step_filename / stl_filename: 文件名，默认按计划 §4.2 命名。
        stl_deflection_mm: STL 三角化线性挠度 [mm]，0.02 mm 兼顾切片精度与体积。
        stl_angular_tolerance_rad: STL 三角化角向容差。
        ruled: 直纹放样开关，仅供调试。

    Raises:
        OptionalDependencyError: build123d 未安装（D-09）。
        ValueError: 截面数或点阵形状非法。
    """
    b3d = _require_build123d()
    solid = blade_solid(sections, ruled=ruled)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    step_path = out / step_filename
    stl_path = out / stl_filename
    if not b3d.export_step(solid, step_path, unit=b3d.Unit.MM):
        raise RuntimeError(f"STEP 写出失败：{step_path}")
    if not b3d.export_stl(
        solid, stl_path, tolerance=stl_deflection_mm, angular_tolerance=stl_angular_tolerance_rad
    ):
        raise RuntimeError(f"STL 写出失败：{stl_path}")
    return step_path, stl_path
