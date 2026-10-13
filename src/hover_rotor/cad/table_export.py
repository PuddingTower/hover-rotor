"""制造参数综合数据包打包器（计划.md Task 4.3，§4.3 截面结构力学量）。

气动最优构型能否过强度关，取决于剖面几何量：截面面积 A、形心偏移
(x_cg, y_cg)、抗弯惯性矩 I_xx 与扭转惯性矩 J 直接决定铺层与强度校核，
也是 §2.4 钝尾缘结构收益（I_xx 显著提升）的兑现凭证。本模块把它们连同
站位几何/气动工作点打包为数控车间、复材铺层与质量追溯三方共用的数据：
``manufacturing_manifest.csv``（逐截面）、``rotor_report.json``
（可无损反序列化回 RotorResult，含配置与种子保证复现）与人类可读的
``manufacturing_report.md``。

截面力学量由多边形面积分自行实现，不引入外部结构库：
  * A 与形心：shoelace 一阶/二阶矩；
  * I_xx：对弦向形心轴的二次矩，先算对 x=0 轴的 Ix0（逐边解析积分），
    再用平行轴定理平移，避免离散求积的坐标系误差；
  * J：按**单闭室薄壁剖面 Bredt 公式** J = 4A^2 / ∮ds/t 计算——翼型剖面
    是近似单室闭合薄壁，钝尾缘保证尾缘壁厚度下限 t_te；壁厚 t 取同弦向
    位置上下表面的垂直间距（小厚度近似下与法向间距等价），前缘处 t -> 0 的
    奇异性可积（t ~ sqrt(psi)），200 点余弦加密网格上的逐段求积收敛。
    厚壁修正对悬停旋翼玻璃钢/碳纤铺层剖面的 J 影响远小于铺层角散差，
    故取薄壁工程精度。

单位契约：SI 内部量在此转为毫米落盘（A [mm^2]、x/y_cg [mm]、I_xx/J [mm^4]），
力学量在弦对齐局部系（前缘原点、弦线沿 +x）计算——扭转是堆叠参数，不进入
剖面力学属性坐标系。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hover_rotor.config import RotorResult, SectionResult

FloatArray = NDArray[np.float64]

MANIFEST_FILENAME = "manufacturing_manifest.csv"
JSON_FILENAME = "rotor_report.json"
MARKDOWN_FILENAME = "manufacturing_report.md"

MIN_POLYGON_POINTS = 20


@dataclass(frozen=True)
class SectionStructuralProperties:
    """单截面结构力学量，毫米单位制（导出边界约定见模块 docstring）。"""

    area_mm2: float
    x_cg_mm: float
    y_cg_mm: float
    i_xx_mm4: float
    j_torsion_mm4: float


def _polygon_moments(points: FloatArray) -> tuple[float, float, float, float]:
    """闭合多边形的有符号面积、形心与对 x=0 轴的二次矩（解析逐边积分）。

    返回 (signed_area, x_cg, y_cg, ix0_about_origin)。点序顺时针时面积为负，
    各量带符号一致，最终取绝对值即可，故对回路方向不敏感。
    """
    x, y = points[:, 0], points[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area2 = float(cross.sum())
    signed_area = 0.5 * area2
    cx = float(((x + x1) * cross).sum() / (3.0 * area2))
    cy = float(((y + y1) * cross).sum() / (3.0 * area2))
    ix0 = float(((y * y + y * y1 + y1 * y1) * cross).sum() / 12.0)
    return signed_area, cx, cy, ix0


def _wall_thicknesses(points: FloatArray) -> FloatArray:
    """回路各节点处的壁厚：与对侧表面同弦向位置的垂直间距。

    CST 回路点序为「下表面 TE->LE，上表面 LE->TE」，前缘点是两段分界。
    前缘节点对侧插值按端点钳位，得到小正数而非 0，逐段 1/t 求积保持有界；
    尾缘封闭段两端厚度即 dz_te。
    """
    i_le = int(np.argmin(points[:, 0]))
    lower = points[: i_le + 1][::-1]  # psi 升序（LE->TE）
    upper = points[i_le + 1 :]  # psi 升序（LE+->TE）
    h = np.zeros(points.shape[0], dtype=np.float64)
    h[: i_le + 1] = np.abs(points[: i_le + 1, 1] - np.interp(points[: i_le + 1, 0], lower[:, 0], upper[:, 1]))
    h[i_le + 1 :] = np.abs(points[i_le + 1 :, 1] - np.interp(points[i_le + 1 :, 0], upper[:, 0], lower[:, 1]))
    return np.maximum(h, 1e-9)


def polygon_structural_properties(points: FloatArray) -> SectionStructuralProperties:
    """由闭合毫米点阵计算 A / 形心 / I_xx / J（多边形面积分 + Bredt 薄壁）。

    Args:
        points: (M, 2) 闭合回路点阵 [mm]，弦对齐局部系（前缘原点），
            点序遵循 CST 回路约定（见 geometry.cst 模块 docstring）。

    Raises:
        ValueError: 点阵形状非法。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < MIN_POLYGON_POINTS:
        raise ValueError(f"型线点阵需为 (M>={MIN_POLYGON_POINTS}, 2)，收到形状 {pts.shape}")

    signed_area, cx, cy, ix0 = _polygon_moments(pts)
    area = abs(signed_area)
    # 平行轴定理：Ix_c = Ix0 - A * y_cg^2（带符号量下整体变号，取绝对值）
    i_xx = abs(ix0 - signed_area * cy * cy)

    # Bredt：∮ds/t 逐段以节点平均壁厚求积；封闭段（末点->首点）即尾缘壁
    seg = np.roll(pts, -1, axis=0) - pts
    ds = np.linalg.norm(seg, axis=1)
    h = _wall_thicknesses(pts)
    h_avg = 0.5 * (h + np.roll(h, -1))
    contour_sl = float((ds / h_avg).sum())
    j_torsion = 4.0 * area * area / contour_sl

    return SectionStructuralProperties(
        area_mm2=area, x_cg_mm=cx, y_cg_mm=cy, i_xx_mm4=i_xx, j_torsion_mm4=j_torsion
    )


def _chord_aligned_mm(section: SectionResult) -> FloatArray:
    return np.asarray(section.points_normalized, dtype=np.float64) * (section.chord * 1000.0)


def enrich_section_structural(section: SectionResult) -> SectionResult:
    """返回补齐结构力学量字段的截面副本（不修改输入，契约对象按值传递）。"""
    props = polygon_structural_properties(_chord_aligned_mm(section))
    return replace(
        section,
        area=props.area_mm2,
        x_cg=props.x_cg_mm,
        y_cg=props.y_cg_mm,
        i_xx=props.i_xx_mm4,
        j_torsion=props.j_torsion_mm4,
    )


MANIFEST_COLUMNS = [
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
]


def _manifest_row(pos: int, s: SectionResult) -> list[Any]:
    return [
        pos,
        f"{s.r_over_R:.6f}",
        f"{s.r * 1000.0:.6f}",
        f"{s.chord * 1000.0:.6f}",
        f"{s.theta_deg:.6f}",
        f"{s.te_thickness * 1000.0:.6f}",
        f"{s.re:.4f}",
        f"{s.ma:.8f}",
        f"{s.alpha_deg:.6f}",
        f"{s.ld:.6f}",
        f"{s.area:.8f}",
        f"{s.x_cg:.8f}",
        f"{s.y_cg:.8f}",
        f"{s.i_xx:.6f}",
        f"{s.j_torsion:.6f}",
    ]


def _write_markdown(result: RotorResult, path: Path) -> None:
    inputs = result.inputs
    fm_ok = "通过" if result.fm >= 0.68 else "未达标"
    thrust_ok = "通过" if result.thrust_error <= 0.005 else "未达标"
    tip_ok = "通过" if result.tip_mach <= inputs.tip_mach_max else "未达标"
    lines = [
        "# 悬停旋翼制造数据报告",
        "",
        f"- 目标推力 T = {inputs.thrust:.3g} N（终态 {result.thrust:.3g} N，"
        f"绝对误差 {result.thrust_error:.3%}）",
        f"- 旋翼半径 R = {inputs.radius:.3g} m，叶片数 B = {inputs.blade_count}，"
        f"轮毂比 = {inputs.hub_ratio:.3g}",
        f"- 钝尾缘绝对厚度 t_te = {inputs.te_thickness * 1000:.3g} mm（全展恒定，D-08 flat 封闭）",
        f"- 转速 Omega = {result.omega:.2f} rad/s（{result.rpm:.1f} rpm），"
        f"叶尖马赫数 {result.tip_mach:.3f}（上限 {inputs.tip_mach_max:.2f}）",
        f"- 轴功率 P = {result.power:.3g} W，理想功率 P_i = {result.power_ideal:.3g} W，"
        f"扭矩 Q = {result.torque:.3g} N·m",
        f"- C_T = {result.ct:.5f}，C_P = {result.cp:.5f}，**FM = {result.fm:.4f}**",
        f"- 求解后端 {result.backend}，阶段 {result.solver_stage}，"
        f"迭代 {result.iterations} 次，seed = {result.seed}",
        "",
        "## 十截面制造清单",
        "",
        "| Sec | r/R | r [mm] | c [mm] | theta [deg] | t_te [mm] | Re | Ma | alpha [deg] |"
        " L/D | A [mm^2] | x_cg | y_cg | I_xx [mm^4] | J [mm^4] |",
        "| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for pos, s in enumerate(result.sections, start=1):
        lines.append(
            f"| {pos:02d} | {s.r_over_R:.4f} | {s.r * 1000:.2f} | {s.chord * 1000:.2f} | {s.theta_deg:.2f} "
            f"| {s.te_thickness * 1000:.3f} | {s.re:.3g} | {s.ma:.3f} | {s.alpha_deg:.2f} | {s.ld:.1f} "
            f"| {s.area:.2f} | {s.x_cg:.2f} | {s.y_cg:.3f} | {s.i_xx:.4g} | {s.j_torsion:.4g} |"
        )
    lines += [
        "",
        "## 验收红线对照（§5）",
        "",
        f"- FM >= 0.68（硬门禁 D-06）：{fm_ok}",
        f"- 推力误差 <= 0.5%：{thrust_ok}（{result.thrust_error:.3%}）",
        f"- 叶尖马赫数 <= {inputs.tip_mach_max:.2f}：{tip_ok}",
        "",
        "结构力学量 A/x_cg/y_cg 单位 mm（系）、mm^2；I_xx、J 单位 mm^4；",
        "坐标系为弦对齐局部系（前缘原点、弦线沿 +x），J 按单闭室薄壁 Bredt 公式。",
        "",
    ]
    if result.warnings:
        lines += ["## 警告", ""] + [f"- {w}" for w in result.warnings] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_manufacturing_tables(
    result: RotorResult,
    output_dir: str | Path = "output",
) -> RotorResult:
    """补齐截面结构力学量并导出 manifest CSV / JSON / Markdown 三件套。

    Args:
        result: 反设计终态（或任意自洽合成档案）。
        output_dir: 输出目录，不存在则创建。

    Returns:
        新的 RotorResult：sections 为补齐 area/x_cg/y_cg/i_xx/j_torsion 字段的
        副本（输入对象不被修改），JSON 即由该副本序列化而来。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    enriched = replace(result, sections=[enrich_section_structural(s) for s in result.sections])

    with (out / MANIFEST_FILENAME).open("w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        writer.writerow(MANIFEST_COLUMNS)
        for pos, s in enumerate(enriched.sections, start=1):
            writer.writerow(_manifest_row(pos, s))

    (out / JSON_FILENAME).write_text(
        json.dumps(enriched.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(enriched, out / MARKDOWN_FILENAME)
    return enriched
