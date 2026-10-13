"""钝尾缘截面二维制造图纸导出器（计划.md Task 4.1，决策 D-08/D-09）。

气动反设计的终点是车间上架：激光切割与慢走丝线切割软件要求刀轨是一条无断点的
闭合轮廓。计划 §4.2 规定 DXF 走 `ezdxf` 通道输出 AutoCAD R2010 标准，每截面
以单一 LWPOLYLINE 写入并强制 ``is_closed=True``；钝尾缘两角点之间由直线段封闭
（D-08 的 flat 切平约定），彻底消除微米级缝隙，CAM 端无需修补几何即可生成刀轨。
同时输出 Selig 格式 ``.dat`` 点集与毫米 CSV，供 XFOIL 类工具与人工校核复用。

文件名裁定：§4.1 导出标识表（``secNN_contour.dxf`` / ``secNN_contour.dat`` /
``secNN.csv``）与 Task 4.1 正文的 ``section_01.dxf`` 不一致，以标识表为准——
该表是 M5 CLI 验收清单（DXF×10、DAT×10、CSV×10）逐条引用的对外接口。

单位契约：``SectionResult`` 内部全程 SI（m），本模块是导出边界，落盘坐标一律
换算为毫米。截面图按弦对齐局部坐标系写出（前缘原点、弦线沿 +x）：安装角是叶片
堆叠参数而非剖面形状，线切割只关心轮廓本身；含扭转角后的绝对空间坐标在 CSV 中
并列给出。STEP/STL（step_builder）才使用旋转后的 ``points_mm``。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hover_rotor.config import OptionalDependencyError, SectionResult

FloatArray = NDArray[np.float64]

DXF_VERSION = "R2010"
INSUNITS_MILLIMETER = 4
"""DXF 头变量 $INSUNITS=4 即毫米，与导出文件单位声明一致"""
DEFAULT_LAYER = "AIRFOIL_CONTOUR"


def _require_ezdxf() -> Any:
    """惰性导入 ezdxf：核心气动链路不得因 CAD extras 缺席而不可用（D-09）。"""
    try:
        import ezdxf
    except ImportError as exc:  # pragma: no cover - 环境相关分支
        raise OptionalDependencyError("DXF 导出需要 ezdxf，请安装可选依赖：pip install -e '.[cad]'") from exc
    return ezdxf


def _chord_aligned_mm(section: SectionResult) -> FloatArray:
    """把归一化 (psi, zeta) 回路放到真实弦长的毫米弦对齐坐标系。

    points_normalized 即 CST 回路点序（尾缘下角 -> 前缘 -> 尾缘上角），
    乘弦长后首末点恰好是钝尾缘两角点，垂直封闭段的长度严格等于 t_te。
    """
    pts = np.asarray(section.points_normalized, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError(
            f"截面 {section.index} 的 points_normalized 需为 (M>=3, 2) 点阵，收到形状 {pts.shape}"
        )
    return pts * (section.chord * 1000.0)


def _section_tag(pos: int) -> str:
    return f"sec{pos:02d}"


def _write_dxf(ezdxf: Any, contour_mm: FloatArray, path: Path, *, layer: str) -> None:
    doc = ezdxf.new(DXF_VERSION, setup=True)
    doc.header["$INSUNITS"] = INSUNITS_MILLIMETER
    if layer not in doc.layers:
        doc.layers.add(layer)
    doc.modelspace().add_lwpolyline(
        [(float(x), float(y)) for x, y in contour_mm],
        close=True,
        dxfattribs={"layer": layer},
    )
    # 落盘 API 兼容带：ezdxf 1.4 起 Drawing.writefile 更名为 saveas
    save = getattr(doc, "saveas", None) or doc.writefile
    save(str(path))


def _write_dat(contour_mm: FloatArray, path: Path, *, header: str) -> None:
    """Selig/XFOIL 格式：上表面自尾缘至前缘，下表面自前缘至尾缘。

    CST 回路是「下表面 TE->LE 接上表面 LE->TE」，整条回路反转即为 Selig 遍历序，
    且首点（上尾缘角）与末点（下尾缘角）的 y 向间距严格等于 t_te。
    """
    selig = contour_mm[::-1]
    lines = [header] + [f"{x:.10f} {y:.10f}" for x, y in selig]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_csv(section: SectionResult, contour_mm: FloatArray, spatial_mm: FloatArray, path: Path) -> None:
    """逐点毫米 CSV：归一化坐标 + 弦对齐坐标 + 含扭转绝对空间坐标 + 截面气动量。"""
    header = [
        "psi",
        "zeta",
        "x_chord_mm",
        "y_chord_mm",
        "x_mm",
        "y_mm",
        "r_over_R",
        "r_mm",
        "chord_mm",
        "theta_deg",
        "alpha_deg",
        "cl",
        "cd",
        "cm",
        "re",
        "ma",
        "te_thickness_mm",
        "dz_te",
    ]
    with path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for (psi, zeta), (xc, yc), (xs, ys) in zip(
            np.asarray(section.points_normalized, dtype=np.float64), contour_mm, spatial_mm, strict=True
        ):
            writer.writerow(
                [
                    f"{psi:.10f}",
                    f"{zeta:.10f}",
                    f"{xc:.6f}",
                    f"{yc:.6f}",
                    f"{xs:.6f}",
                    f"{ys:.6f}",
                    f"{section.r_over_R:.6f}",
                    f"{section.r * 1000.0:.6f}",
                    f"{section.chord * 1000.0:.6f}",
                    f"{section.theta_deg:.6f}",
                    f"{section.alpha_deg:.6f}",
                    f"{section.cl:.8f}",
                    f"{section.cd:.8f}",
                    f"{section.cm:.8f}",
                    f"{section.re:.4f}",
                    f"{section.ma:.8f}",
                    f"{section.te_thickness * 1000.0:.8f}",
                    f"{section.dz_te:.8f}",
                ]
            )


def export_sections_dxf(
    sections: list[SectionResult] | tuple[SectionResult, ...],
    output_dir: str | Path = "output",
    *,
    layer: str = DEFAULT_LAYER,
) -> list[Path]:
    """批量导出每截面的闭合 DXF 图纸、Selig ``.dat`` 与毫米 CSV。

    Args:
        sections: 按展向自叶根至叶尖排序的截面档案；文件名站位号取列表顺序
            （1 基），与 §4.1 表的 Sec 01..10 对齐。
        output_dir: 输出目录，不存在则创建。
        layer: DXF 图层名。

    Returns:
        写入的文件路径列表，每截面依次 (dxf, dat, csv)。

    Raises:
        OptionalDependencyError: ezdxf 未安装（D-09，区别于 ImportError）。
        ValueError: 点阵形状非法。
    """
    ezdxf = _require_ezdxf()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for pos, section in enumerate(sections, start=1):
        tag = _section_tag(pos)
        contour_mm = _chord_aligned_mm(section)
        spatial_mm = np.asarray(section.points_mm, dtype=np.float64)
        if spatial_mm.shape != contour_mm.shape:
            raise ValueError(f"截面 {tag} 的 points_mm 形状 {spatial_mm.shape} 与 points_normalized 不一致")

        dxf_path = out / f"{tag}_contour.dxf"
        dat_path = out / f"{tag}_contour.dat"
        csv_path = out / f"{tag}.csv"

        _write_dxf(ezdxf, contour_mm, dxf_path, layer=layer)
        header = f"{tag} chord={section.chord * 1000.0:.4f}mm te={section.te_thickness * 1000.0:.4f}mm"
        _write_dat(contour_mm, dat_path, header=header)
        _write_csv(section, contour_mm, spatial_mm, csv_path)
        written.extend([dxf_path, dat_path, csv_path])
    return written
