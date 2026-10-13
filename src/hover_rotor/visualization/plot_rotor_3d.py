"""三维叶片放样曲面的交互式 Plotly 可视化（计划.md Task 5.2）。

STEP/STL 交付给车间，但工程师与甲方需要一个**脱离 Python 环境**就能全角度旋转、缩放、
悬停读数的检查手段：反设计出来的桨叶是不是真的扭转了？钝尾缘的垂直平面在三维里有没有
被放样抹平？法向载荷的峰值落在哪一站？静态 PNG 只能看投影，浏览器里的交互曲面才能直接
判读。计划.md §4.2 因此要求"输出可离线在全浏览器中全角度缩放查看的独立 HTML"。

三条实现纪律：

* **曲面只依赖结果契约，不依赖 CAD 内核**。放样走纯 NumPy：周向等弧长重采样 + 展向
  PCHIP，不需要 build123d/OpenCASCADE（Task 4.2 的可选 extras）。可视化层必须能在未装
  ``[cad]`` 的容器里独立工作——这是 §0 D-09 的分层纪律。
* **单文件、零网络**。``include_plotlyjs=True`` 把 plotly.js 内联进 HTML（约 4.86 MB），
  产物不再引用任何外部 ``src=``/``href=`` 资源，断网、U 盘拷贝、邮件附件都能打开。
  代价是 Task 5.2 的 5 MB 门禁只给网格数据留下百余 KB 预算，故坐标按 5 位小数落盘、
  去掉内嵌模板 JSON，并按 :data:`_SIZE_STEPS` 自动降采样兜底。
* **色谱贴的是局部法向载荷**。BEMT 只给站位级 $dN$ [N/m]，不含周向压力分布，因此把
  $dN/c$（面平均法向压力 [Pa]）沿同一站位周向均布。这是可辩护的降维表示，不臆造
  吸/压力面差异；拿到真实压力分布时由调用方显式传 ``loading`` 覆盖。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from hover_rotor.config import RotorResult, SectionResult
from hover_rotor.visualization.plot_sections import resolve_figure_path, section_outline_in_chord_frame

HTML_BASENAME = "rotor_interactive_3d.html"
"""默认落盘文件名，与计划.md Task 5.2 的输出契约一致。"""

GATE_MAX_HTML_BYTES = 5_000_000
"""计划.md Task 5.2 的硬门禁：单文件 HTML 必须小于 5 MB。"""

SAFE_MAX_HTML_BYTES = GATE_MAX_HTML_BYTES - 40_000
"""落盘用的实际预算：在门禁下再留 40 KB，吸收 plotly 版本升级带来的字节漂移。"""

COORD_DECIMALS = 5
"""坐标保留小数位（0.01 mm）。桨叶尺度 O(1 m) 下绘图精度足够，而每个浮点少写 8~10 个
字符是 5 MB 预算里最有效的瘦身手段。"""

DEFAULT_N_CHORD = 26
DEFAULT_N_SPAN = 48

FloatArray = NDArray[np.float64]

_SIZE_STEPS: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (2, 2), (3, 3))
"""超预算时的降采样档位（展向步长, 周向步长）。展向先砍：其梯度已由 PCHIP 平滑，
周向砍狠了会抹掉钝尾缘的直角端面。"""


@dataclass(frozen=True)
class BladeSurface:
    """放样得到的结构化叶片曲面网格（Task 5.2 的中间契约）。"""

    vertices: FloatArray
    """``(n_span, n_chord, 3)`` 节点坐标 [m]：X 径向、Y 桨盘切向、Z 旋翼轴/推力方向。"""

    loading: FloatArray
    """``(n_span, n_chord)`` 节点局部法向载荷（默认 $dN/c$ [Pa]），作色谱贴图。"""

    r_over_R: FloatArray
    """``(n_span,)`` 展向节点的无量纲站位，供悬停读数。"""

    station_outlines: FloatArray
    """``(n_sections, n_chord, 3)`` 原始制造截面在三维中的定向轮廓，叠加显示用。"""

    def decimated(self, span_step: int, chord_step: int) -> BladeSurface:
        """按步长抽稀，用于把 HTML 压回体积门禁之内。"""
        return BladeSurface(
            vertices=self.vertices[::span_step, ::chord_step, :],
            loading=self.loading[::span_step, ::chord_step],
            r_over_R=self.r_over_R[::span_step],
            station_outlines=self.station_outlines[:, ::chord_step, :],
        )


@dataclass(frozen=True)
class _StationFrame:
    """十个站位的放样控制量，全部按 r 升序对齐。"""

    r_over_R: FloatArray
    r: FloatArray
    chord: FloatArray
    theta_rad: FloatArray
    dn: FloatArray
    rings: FloatArray
    """``(n_sections, n_chord, 2)`` 弦向归一型线，起点为尾缘下沿角点。"""


# --------------------------------------------------------------------------------------
# 曲面构造
# --------------------------------------------------------------------------------------


def _resample_closed_ring(psi: FloatArray, zeta: FloatArray, n_nodes: int) -> tuple[FloatArray, FloatArray]:
    """把闭合型线按周长等弧长重采样成固定 ``n_nodes`` 个点。

    放样要求各站位截面的节点数与节点相位一致，否则相邻截面连出的三角网会扭曲。起点沿用
    型线自身的第 0 点（尾缘下沿角点），于是周向 wrap 边恰好落在钝尾缘平面上——放样出的
    曲面在尾缘自动生成那个可加工的垂直端面。
    """
    x = np.concatenate((psi, [psi[0]]))
    y = np.concatenate((zeta, [zeta[0]]))
    segment = np.hypot(np.diff(x), np.diff(y))
    total = float(np.sum(segment))
    if total <= 0.0:
        raise ValueError("截面型线周长退化为零，无法放样")
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    target = np.linspace(0.0, total, n_nodes, endpoint=False)
    return np.interp(target, arc, x), np.interp(target, arc, y)


def _orient(
    xi: FloatArray, zeta: FloatArray, radius: FloatArray, chord: FloatArray, theta_rad: FloatArray
) -> FloatArray:
    """把无量纲型线投射进三维桨叶坐标系。

    坐标系：X 沿径向（叶展）、Y 在桨盘平面内（旋转切向）、Z 为旋翼轴即推力方向。安装角
    $\\theta$ 绕 X 轴转动剖面：$\\theta>0$ 须让前缘高于后缘，剖面才与沿 $-Y$ 的相对风夹出
    正迎角；符号写反会交出一片反扭转的桨叶，而数值上完全看不出来。
    """
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    out = np.empty(xi.shape + (3,), dtype=np.float64)
    out[..., 0] = radius[:, None]
    out[..., 1] = chord[:, None] * (xi * cos_t[:, None] + zeta * sin_t[:, None])
    out[..., 2] = chord[:, None] * (-xi * sin_t[:, None] + zeta * cos_t[:, None])
    return out


def _station_frame(sections: Sequence[SectionResult], n_chord: int) -> _StationFrame:
    """整理站位控制量：几何直接取契约字段，型线经弦向坐标系归一后等弧长重采样。"""
    ordered = sorted(sections, key=lambda s: float(s.r))
    rings = np.empty((len(ordered), n_chord, 2), dtype=np.float64)
    for i, section in enumerate(ordered):
        psi, zeta = section_outline_in_chord_frame(section)
        rings[i, :, 0], rings[i, :, 1] = _resample_closed_ring(psi, zeta, n_chord)
    return _StationFrame(
        r_over_R=np.array([float(s.r_over_R) for s in ordered], dtype=np.float64),
        r=np.array([float(s.r) for s in ordered], dtype=np.float64),
        chord=np.array([float(s.chord) for s in ordered], dtype=np.float64),
        theta_rad=np.array([float(s.theta_deg) for s in ordered], dtype=np.float64) * (np.pi / 180.0),
        dn=np.array([float(s.dN) for s in ordered], dtype=np.float64),
        rings=rings,
    )


def _loft_along_span(frame: _StationFrame, values: FloatArray, n_span: int) -> FloatArray:
    """在展向控制站位间用 PCHIP 单调插值把展向分布加密到 ``n_span`` 站。

    用保形样条而不是高阶样条：弦长/环量沿展向单调，任何数值过冲都会在曲面上表现为不存在的
    鼓包，直接误导判读。
    """
    if frame.r.size < 2:
        raise ValueError("至少需要 2 个展向截面才能放样曲面")
    grid = np.linspace(frame.r[0], frame.r[-1], n_span)
    return np.asarray(PchipInterpolator(frame.r, values, axis=0)(grid), dtype=np.float64)


def build_blade_surface(
    result: RotorResult, *, n_chord: int = DEFAULT_N_CHORD, n_span: int = DEFAULT_N_SPAN
) -> BladeSurface:
    """由结果契约放样出叶片曲面网格与局部法向载荷场。

    周向：每个截面重采样为 ``n_chord`` 个等弧长节点；展向：以 PCHIP 把 $(\\psi,\\zeta)$、
    $c$、$\\theta$ 与 $dN/c$ 加密到 ``n_span`` 站，再逐站定向为三维节点。
    """
    if n_chord < 8:
        raise ValueError(f"周向节点数至少 8 才能保住钝尾缘形状，收到 {n_chord}")
    if n_span < 4:
        raise ValueError(f"展向站点数至少 4，收到 {n_span}")

    sections = list(result.sections)
    if not sections:
        raise ValueError("结果中没有截面档案（sections 为空），无法放样叶片曲面")

    frame = _station_frame(sections, n_chord)
    psi_grid = _loft_along_span(frame, frame.rings[:, :, 0], n_span)
    zeta_grid = _loft_along_span(frame, frame.rings[:, :, 1], n_span)
    chord_grid = _loft_along_span(frame, frame.chord, n_span)
    theta_grid = _loft_along_span(frame, frame.theta_rad, n_span)
    radius_grid = np.linspace(frame.r[0], frame.r[-1], n_span)
    pressure_grid = _loft_along_span(frame, frame.dn / np.maximum(frame.chord, 1.0e-9), n_span)

    outline_grid = _orient(frame.rings[:, :, 0], frame.rings[:, :, 1], frame.r, frame.chord, frame.theta_rad)
    return BladeSurface(
        vertices=_orient(psi_grid, zeta_grid, radius_grid, chord_grid, theta_grid),
        loading=np.repeat(pressure_grid[:, None], n_chord, axis=1),
        r_over_R=radius_grid / float(result.inputs.radius),
        station_outlines=outline_grid,
    )


def _surface_from_points(mesh_points: NDArray[Any], loading: NDArray[Any] | None) -> BladeSurface:
    """接受调用方自算好的结构化网格 ``(n_span, n_chord, 3)``（Task 5.2 的备选输入）。"""
    pts = np.asarray(mesh_points, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise TypeError(
            "mesh_points 需为放样曲面的结构化节点阵列，形状 (n_span, n_chord, 3)，"
            f"收到 {pts.shape}；若手上是 RotorResult 请直接传结果对象"
        )
    field = np.zeros(pts.shape[:2], dtype=np.float64)
    if loading is not None:
        field = np.asarray(loading, dtype=np.float64)
    if field.ndim == 1:
        field = np.repeat(field.reshape(-1, 1), pts.shape[1], axis=1)
    if field.shape != pts.shape[:2]:
        raise ValueError(f"loading 形状需为 {pts.shape[:2]} 或 ({pts.shape[0]},)，收到 {field.shape}")
    # 网格直通输入没有站位档案，取 10 个代表性展向行当轮廓叠加（至多 10 条环）。
    rows = np.linspace(0, pts.shape[0] - 1, min(10, pts.shape[0])).round().astype(int)
    return BladeSurface(
        vertices=pts,
        loading=field,
        r_over_R=np.linspace(0.0, 1.0, pts.shape[0], dtype=np.float64),
        station_outlines=pts[rows],
    )


# --------------------------------------------------------------------------------------
# Trace 组装
# --------------------------------------------------------------------------------------


def _rounded(values: FloatArray, decimals: int = COORD_DECIMALS) -> list[Any]:
    """取位后转 Python float：numpy 标量的 ``repr`` 会给 JSON 多塞十几个字符。"""
    return [float(v) for v in np.round(np.asarray(values, dtype=np.float64), decimals).ravel()]


def _grid_triangles(n_span: int, n_chord: int) -> tuple[list[int], list[int], list[int]]:
    """结构化 $(n_{span}, n_{chord})$ 网格的三角剖分，周向按闭合环 wrap。"""
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for s in range(n_span - 1):
        near = s * n_chord
        far = (s + 1) * n_chord
        for c in range(n_chord):
            a, b = near + c, near + (c + 1) % n_chord
            d, e = far + c, far + (c + 1) % n_chord
            i += [a, b]
            j += [b, e]
            k += [d, d]
    return i, j, k


def _ring_line_kwargs(outlines: FloatArray) -> dict[str, Any]:
    """把若干定向截面轮廓拼成一条以 ``None`` 断线的 3D 折线——一条 trace 画完十个截面。"""
    xs: list[Any] = []
    ys: list[Any] = []
    zs: list[Any] = []
    for ring in outlines:
        closed = np.vstack((ring, ring[:1]))
        for axis, buffer in zip((closed[:, 0], closed[:, 1], closed[:, 2]), (xs, ys, zs), strict=True):
            buffer.extend(float(v) for v in np.round(axis, COORD_DECIMALS))
            buffer.append(None)
    return {"x": xs, "y": ys, "z": zs}


def _rotated(vertices: FloatArray, angle_rad: float) -> FloatArray:
    """绕旋翼轴 Z 复制相邻叶片。"""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    out = vertices.copy()
    out[..., 0] = vertices[..., 0] * cos_a - vertices[..., 1] * sin_a
    out[..., 1] = vertices[..., 0] * sin_a + vertices[..., 1] * cos_a
    return out


def _traces(
    surface: BladeSurface, *, blades: int, colorscale: str, show_sections: bool, loading_label: str
) -> list[dict[str, Any]]:
    """每片桨叶一条 mesh3d 曲面 trace，外加一条十截面轮廓折线 trace。"""
    n_span, n_chord = surface.vertices.shape[0], surface.vertices.shape[1]
    i, j, k = _grid_triangles(n_span, n_chord)
    r_flat = _rounded(np.repeat(surface.r_over_R, n_chord), 3)
    quantity = loading_label.split("  [")[0].split(" [")[0]
    traces: list[dict[str, Any]] = []

    for blade in range(max(1, blades)):
        verts = _rotated(surface.vertices, 2.0 * np.pi * blade / max(1, blades))
        hover = (
            f"Blade {blade + 1}<br>r/R=%{{customdata:.3f}}<br>{quantity}=%{{intensity:.4g}}<extra></extra>"
        )
        traces.append(
            {
                "type": "mesh3d",
                "x": _rounded(verts[..., 0]),
                "y": _rounded(verts[..., 1]),
                "z": _rounded(verts[..., 2]),
                "i": i,
                "j": j,
                "k": k,
                "intensity": _rounded(surface.loading, 6),
                "customdata": r_flat,
                "colorscale": colorscale,
                "name": f"Blade {blade + 1}",
                # 同一把色标只画一次，多叶片时避免三根 colorbar 抢版面。
                "showscale": blade == 0,
                "colorbar": {"title": {"text": loading_label, "side": "right"}, "thickness": 16, "len": 0.7},
                "lighting": {"ambient": 0.6, "diffuse": 0.5, "specular": 0.12, "roughness": 0.62},
                "hovertemplate": hover,
            }
        )

    if show_sections:
        traces.append(
            {
                "type": "scatter3d",
                **_ring_line_kwargs(surface.station_outlines),
                "mode": "lines",
                "line": {"color": "rgb(24,24,24)", "width": 3.0},
                "opacity": 0.9,
                "name": f"{surface.station_outlines.shape[0]} 个截面轮廓",
                "hoverinfo": "name",
            }
        )
    return traces


def _scene_layout(title: str, aspect_mode: str) -> dict[str, Any]:
    return {
        "title": {"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 14}},
        "margin": {"l": 0, "r": 0, "t": 56, "b": 0},
        "scene": {
            "xaxis": {"title": {"text": "径向 X（叶展方向） [m]"}},
            "yaxis": {"title": {"text": "切向 Y（桨盘平面） [m]"}},
            "zaxis": {"title": {"text": "旋翼轴 Z（推力方向） [m]"}},
            "aspectmode": aspect_mode,
            "camera": {"eye": {"x": 1.3, "y": -1.2, "z": 0.7}},
        },
    }


# --------------------------------------------------------------------------------------
# 公共入口
# --------------------------------------------------------------------------------------


RotorSource = RotorResult | NDArray[np.float64]
"""Task 5.2 的两个合法输入形态：结果契约对象，或已放样好的结构化节点阵。"""


def build_rotor_figure(
    result_or_mesh_points: RotorSource,
    *,
    loading: NDArray[np.float64] | None = None,
    loading_label: str = "局部法向压力 dN/c  [Pa]",
    blade_count: int | None = 1,
    n_chord: int = DEFAULT_N_CHORD,
    n_span: int = DEFAULT_N_SPAN,
    colorscale: str = "Viridis",
    show_sections: bool = True,
    aspect_mode: str = "data",
    title: str | None = None,
) -> Any:
    """返回 :class:`plotly.graph_objects.Figure` 本体，不落盘。

    与 HTML 出口拆开，测试与报告层才能检视 trace 数、色谱与悬停模板，而不必去解析一个
    4.8 MB 的字符串。plotly 在此延迟导入：本模块的几何放样部分只用 NumPy/scipy。
    """
    surface = _as_surface(result_or_mesh_points, loading, n_chord=n_chord, n_span=n_span)
    blades = _resolve_blade_count(blade_count, result_or_mesh_points)
    return _assemble_figure(
        surface,
        blades=blades,
        loading_label=loading_label,
        colorscale=colorscale,
        show_sections=show_sections,
        title=_resolve_title(title, result_or_mesh_points, blades),
        aspect_mode=aspect_mode,
    )


def render_interactive_rotor_html(
    result_or_mesh_points: RotorSource,
    output_path: str | Path,
    *,
    loading: NDArray[np.float64] | None = None,
    loading_label: str = "局部法向压力 dN/c  [Pa]",
    blade_count: int | None = 1,
    n_chord: int = DEFAULT_N_CHORD,
    n_span: int = DEFAULT_N_SPAN,
    colorscale: str = "Viridis",
    show_sections: bool = True,
    aspect_mode: str = "data",
    title: str | None = None,
    max_html_bytes: int = SAFE_MAX_HTML_BYTES,
    div_id: str = "hover-rotor-3d",
) -> Path:
    """把叶片三维曲面写成单文件离线交互 HTML ``rotor_interactive_3d.html``（计划.md Task 5.2）。

    Args:
        result_or_mesh_points: 反设计结果 :class:`~hover_rotor.config.RotorResult`，或调用方
            自行放样好的结构化节点阵列 ``(n_span, n_chord, 3)`` [m]。
        output_path: 目标文件路径；目录或无后缀路径自动补 ``rotor_interactive_3d.html``。
        loading: 节点/站位载荷场，覆盖默认的 $dN/c$。结果对象输入时可给
            ``(n_sections,)``、``(n_span,)`` 或 ``(n_span, n_chord)``；网格输入时同理。
        loading_label: 色谱标题（含物理单位）。
        blade_count: 渲染叶片数。默认 1，与 Task 4.2 的单体 STEP/STL 导出物一致；
            传 ``None`` 取 ``result.inputs.blade_count``；传 >1 按 $2\\pi/B$ 旋转复制。
        n_chord: 周向等弧长重采样节点数（含钝尾缘 wrap 边）。
        n_span: 展向放样站点数。
        colorscale: Plotly 色谱名。
        show_sections: 是否叠加各制造截面的空间定向轮廓。
        aspect_mode: 三维场景比例模式；``"data"`` 保持真实长细比。
        title: 场景标题，缺省由结果整机量生成。
        max_html_bytes: 体积预算，超出时按 :data:`_SIZE_STEPS` 逐档降采样。
        div_id: HTML 容器 id。

    Returns:
        实际写出的文件 :class:`~pathlib.Path`。

    Raises:
        ValueError: 截面数/载荷形状不合法，或降到最粗档仍超预算。
        TypeError: 传入的网格形状不是 ``(n_span, n_chord, 3)``。
    """
    surface = _as_surface(result_or_mesh_points, loading, n_chord=n_chord, n_span=n_span)
    blades = _resolve_blade_count(blade_count, result_or_mesh_points)
    html = _serialize_page(
        surface,
        blades=blades,
        loading_label=loading_label,
        colorscale=colorscale,
        show_sections=show_sections,
        title=_resolve_title(title, result_or_mesh_points, blades),
        aspect_mode=aspect_mode,
        div_id=div_id,
        max_html_bytes=max_html_bytes,
    )

    path = resolve_figure_path(output_path, HTML_BASENAME)
    path.write_text(html, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# 内部装配
# --------------------------------------------------------------------------------------


def _as_surface(
    source: RotorSource, loading: NDArray[np.float64] | None, *, n_chord: int, n_span: int
) -> BladeSurface:
    if isinstance(source, RotorResult):
        surface = build_blade_surface(source, n_chord=n_chord, n_span=n_span)
        if loading is None:
            return surface
        return _override_loading(surface, source, np.asarray(loading, dtype=np.float64))
    return _surface_from_points(np.asarray(source, dtype=np.float64), loading)


def _assemble_figure(
    surface: BladeSurface,
    *,
    blades: int,
    loading_label: str,
    colorscale: str,
    show_sections: bool,
    title: str,
    aspect_mode: str,
) -> Any:
    """把放样好的曲面装成 :class:`plotly.graph_objects.Figure`。

    ``build_rotor_figure``（供检视）与 ``_serialize_page``（供落盘）共用这一份装配，
    避免两个出口画出不同形状的场景。plotly 在此延迟导入：几何放样只用 NumPy/scipy，
    没装 plotly 的环境依然能做放样与单测。
    """
    import plotly.graph_objects as go

    fig = go.Figure(
        data=_traces(
            surface,
            blades=blades,
            colorscale=colorscale,
            show_sections=show_sections,
            loading_label=loading_label,
        )
    )
    fig.update_layout(_scene_layout(title, aspect_mode))
    # 清空默认模板：那段模板 JSON 约 6.6 KB，在 5 MB 门禁里是实打实的网格预算。
    fig.layout.template = None
    return fig


def _override_loading(surface: BladeSurface, result: RotorResult, values: FloatArray) -> BladeSurface:
    """用调用方的载荷场替换默认 $dN/c$（例如 CFD 表面压力或 $c_l$ 分布）。"""
    n_span, n_chord = surface.loading.shape
    flat = values.ravel()
    if flat.size == surface.loading.size:
        field = flat.reshape(n_span, n_chord)
    elif flat.size == n_span:
        field = np.repeat(flat[:, None], n_chord, axis=1)
    elif flat.size == len(result.sections):
        ordered = sorted(result.sections, key=lambda s: float(s.r))
        ctrl = np.array([float(s.r) for s in ordered], dtype=np.float64)
        grid = np.linspace(ctrl[0], ctrl[-1], n_span)
        lofted = np.asarray(PchipInterpolator(ctrl, flat)(grid), dtype=np.float64)
        field = np.repeat(lofted[:, None], n_chord, axis=1)
    else:
        raise ValueError(
            f"loading 长度需为 {n_span}、{n_span * n_chord} 或 {len(result.sections)}，收到 {flat.size}"
        )
    return BladeSurface(surface.vertices, field, surface.r_over_R, surface.station_outlines)


def _resolve_blade_count(blade_count: int | None, source: RotorSource) -> int:
    if blade_count is None:
        if isinstance(source, RotorResult):
            return int(source.inputs.blade_count)
        raise ValueError("传网格点阵时 blade_count=None 无从确定叶片数，请给出显式整数")
    if blade_count < 1:
        raise ValueError(f"叶片数至少 1，收到 {blade_count}")
    return int(blade_count)


def _resolve_title(title: str | None, source: RotorSource, blades: int) -> str:
    if title is not None:
        return title
    if isinstance(source, RotorResult):
        return (
            "三维放样叶片曲面 · 局部法向载荷色谱    "
            f"$R$={source.inputs.radius:.3f} m    $\\Omega$={source.rpm:.0f} rpm    "
            f"$M_{{tip}}$={source.tip_mach:.3f}    $FM$={source.fm:.3f}    "
            f"{source.n_sections} 截面放样 / {blades} 片桨叶"
        )
    return "三维放样叶片曲面 · 局部法向载荷色谱"


def _serialize_page(
    surface: BladeSurface,
    *,
    blades: int,
    loading_label: str,
    colorscale: str,
    show_sections: bool,
    title: str,
    aspect_mode: str,
    div_id: str,
    max_html_bytes: int,
) -> str:
    """序列化为内联 plotly.js 的完整 HTML，必要时逐档降采样以守住 5 MB 门禁。"""
    produced = 0
    for span_step, chord_step in _SIZE_STEPS:
        coarse = surface if (span_step, chord_step) == (1, 1) else surface.decimated(span_step, chord_step)
        fig = _assemble_figure(
            coarse,
            blades=blades,
            loading_label=loading_label,
            colorscale=colorscale,
            show_sections=show_sections,
            title=title,
            aspect_mode=aspect_mode,
        )
        # include_plotlyjs=True 把库本体写进 <head>：产物因此不含任何外部 script src，
        # 这才是"脱离 Python 环境可独立交互"的字面保证。
        html = fig.to_html(include_plotlyjs=True, full_html=True, div_id=div_id, config={"responsive": True})
        produced = len(html.encode("utf-8"))
        if produced <= max_html_bytes:
            return html
    raise ValueError(
        f"内联 plotly.js 后 HTML 仍为 {produced:,} B，超出 {max_html_bytes:,} B 预算；"
        "请降低 blade_count / n_chord / n_span 或改用 include_plotlyjs='cdn' 的联机交付"
    )


__all__ = [
    "DEFAULT_N_CHORD",
    "DEFAULT_N_SPAN",
    "GATE_MAX_HTML_BYTES",
    "HTML_BASENAME",
    "SAFE_MAX_HTML_BYTES",
    "BladeSurface",
    "build_blade_surface",
    "build_rotor_figure",
    "render_interactive_rotor_html",
]
