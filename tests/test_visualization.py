"""Task 5.1 / 5.2 验收：全维气动可视化（计划.md §4.2 第三条）。

判定准则取自计划.md：
  5.1 —— ``Agg`` 后端下生成无报错、图例/坐标轴标签/物理单位完备、文件 >100 KB 且可被
         PIL 正常读取、分辨率 >=300 DPI；
  5.2 —— ``include_plotlyjs`` 内联、无外部网络依赖、文件 <5 MB、HTML 含
         ``plotly-graph-div`` 且 trace 数 >=1。

优化器（M3）与 CAD 层（M4）与本里程碑并行开发，故这里**自带一份合成 RotorResult**：几何由
Task 1.1/1.2 的真实生成器产出（余弦聚点 + 钝尾缘 CST），气动工作点由 Task 2.2 的解析极线代理
就地求解，并缩放设计迎角使积分推力命中给定目标。可视化层因此只验收"结果契约 -> 图 / 网页"
这一段，不被上游进度卡住。

关于"无网络依赖"的断言口径：内联的 plotly.min.js 本体里带有一个 ``topojsonURL`` 默认常量
（地理图层用，本产物永不触发），对整份 HTML 做 ``cdn.plot.ly`` 裸串 grep 必然假阳性。测试因此
改为：页面骨架与数据脚本内不含任何 ``src=``/``href=`` 外部资源，且内联库确实存在——这才是
"断网也能打开"的字面保证。
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from hover_rotor.aero.polar import AeroPolarEvaluator, AirfoilShape
from hover_rotor.config import A_INF, RHO_INF, RotorInputs, RotorResult, SectionResult
from hover_rotor.geometry.cst import generate_flatback_cst
from hover_rotor.geometry.discretization import cosine_span_positions, interpolate_span
from hover_rotor.visualization import (
    BladeSurface,
    build_blade_surface,
    build_rotor_figure,
    build_summary_figure,
    generate_summary_plots,
    render_interactive_rotor_html,
)
from hover_rotor.visualization.plot_rotor_3d import (
    DEFAULT_N_CHORD,
    DEFAULT_N_SPAN,
    GATE_MAX_HTML_BYTES,
    _orient,
)
from hover_rotor.visualization.plot_sections import FIGURE_BASENAME, MIN_DPI, section_outline_in_chord_frame

RADIUS = 0.75
THRUST = 80.0
TE_THICKNESS = 0.0025
TIP_MACH = 0.35
"""夹具转速取 M_tip=0.35：它让反演出的扭转落在 21.4 -> 3.2 deg，与 §4.1 参考表的
22.5 -> 0.5 deg 同量级。"""
BLADE_COUNT = 2
N_SECTIONS = 10
HUB_OVER_R = 0.15
MU_AIR = 1.81e-5
"""动力粘度 [Pa·s]（标准海平面），仅用于合成雷诺数。"""

R_CTRL = np.array([HUB_OVER_R, 0.42, 0.72, 1.00])
CHORD_CTRL = np.array([0.120, 0.095, 0.073, 0.060]) * RADIUS
ALPHA_GRID = np.linspace(-4.0, 16.0, 41)


def _polygon_properties(points: np.ndarray) -> tuple[float, float, float, float]:
    """多边形面积、形心与关于形心的 $I_{xx}$（真实几何积分，点序自动闭合）。"""
    x, y = points[:, 0], points[:, 1]
    xi, yi = np.roll(x, -1), np.roll(y, -1)
    cross = x * yi - xi * y
    signed_area = 0.5 * float(np.sum(cross))
    if abs(signed_area) < 1.0e-14:
        return 0.0, 0.0, 0.0, 0.0
    cx = float(np.sum((x + xi) * cross) / (6.0 * signed_area))
    cy = float(np.sum((y + yi) * cross) / (6.0 * signed_area))
    ix_about_chord = float(np.sum((y * y + y * yi + yi * yi) * cross) / 12.0)
    i_xx = ix_about_chord - signed_area * cy * cy
    return abs(signed_area), cx, cy, abs(i_xx)


def _to_mm_points(local: np.ndarray, radius_mm: float, theta_rad: float) -> np.ndarray:
    """局部弦向点阵 -> 制造用物理毫米点阵：绕桨轴转 theta 后平移到站位 r（§4.1 惯例）。"""
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    out = np.empty_like(local)
    out[:, 0] = local[:, 0] * cos_t - local[:, 1] * sin_t + radius_mm
    out[:, 1] = local[:, 0] * sin_t + local[:, 1] * cos_t
    return out


def _span_loads(
    evaluator: AeroPolarEvaluator,
    shapes: list[AirfoilShape],
    alpha: np.ndarray,
    chord: np.ndarray,
    w: np.ndarray,
    re: np.ndarray,
    ma: np.ndarray,
    r: np.ndarray,
    phi: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """给定迎角分布下的整机推力、扭矩与逐剖面 $c_l$、$c_d$。"""
    dr = np.gradient(r)
    cl = np.empty(len(shapes))
    cd = np.empty(len(shapes))
    for i, shape in enumerate(shapes):
        res = evaluator.evaluate(shape, np.array([alpha[i]]), re=float(re[i]), ma=float(ma[i]))
        cl[i], cd[i] = float(res.cl[0]), float(res.cd[0])
    q = 0.5 * RHO_INF * w**2 * chord
    thrust = float(np.sum(q * cl * dr) * BLADE_COUNT)
    profile_drag = cl * np.sin(np.radians(phi)) + cd * np.cos(np.radians(phi))
    torque = float(np.sum(q * profile_drag * r * dr) * BLADE_COUNT)
    return thrust, torque, cl, cd


def _calibrate_alpha(
    evaluator: AeroPolarEvaluator,
    shapes: list[AirfoilShape],
    chord: np.ndarray,
    w: np.ndarray,
    re: np.ndarray,
    ma: np.ndarray,
    r: np.ndarray,
    phi: np.ndarray,
    r_over_R: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """把设计迎角分布整体缩放，直到积分推力命中 T_target（§5 红线：绝对误差 <0.5%）。

    夹具因此不是"随手编的一组数"，而是真的做了反设计里最关键的一次闭合：给定 $\\Omega$、
    $c(r)$ 与剖面，求出使 T=T_target 的 $\\alpha(r)$，再由 $\\theta=\\alpha+\\phi$ 反演几何扭转。
    面板 (b) 的扭转曲线就是这么来的，不是硬编码的典型值。
    """
    schedule = 7.6 - 6.4 * (r_over_R - HUB_OVER_R) / (1.0 - HUB_OVER_R)
    scale = 1.0
    alpha = schedule
    loads = (0.0, 0.0, np.zeros(len(shapes)), np.zeros(len(shapes)))
    for _ in range(30):
        alpha = schedule * scale
        loads = _span_loads(evaluator, shapes, alpha, chord, w, re, ma, r, phi)
        if abs(loads[0] / THRUST - 1.0) < 2.0e-3:
            break
        scale *= (THRUST / loads[0]) ** 0.7
    return (alpha, *loads)


def _minor_extent(row: np.ndarray) -> float:
    """站点环在自身平面内最小主轴方向的跨度，即该站物理最大厚度 $(t/c)\\cdot c$ [m]。"""
    centered = row[:, 1:] - row[:, 1:].mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    return float(np.ptp(centered @ vectors[:, 0]))


def _line_data(ax: Axes, *, min_points: int) -> np.ndarray:
    """取坐标轴上所有足够长的折线数据，过滤掉图例 proxy 与 axhline 这类两点辅助线。"""
    segments = [line.get_xydata() for line in ax.get_lines()]
    return np.vstack([data for data in segments if data.shape[0] >= min_points])


@pytest.fixture(scope="module")
def synthetic_result(naca_to_cst) -> RotorResult:
    """物理自洽的合成反设计终态（不依赖仍在并行开发的 M3 优化器）。

    几何严格按计划.md §4.1：10 站余弦聚点、弦长 0.12R 收缩到 0.06R、钝尾缘绝对厚度 2.5 mm
    全展恒定（故 $\\Delta\\bar z_{te}=t_{te}/c$ 逐站增大）、最大厚度由叶根 12% 向叶尖收到约 10%。
    气动量全部闭合：$\\phi=\\arctan(v_i/\\Omega r)$、$W=\\sqrt{(\\Omega r)^2+v_i^2}$、$\\alpha$ 由
    推力约束标定、$\\theta=\\alpha+\\phi$、$c_l/c_d$ 由解析极线代理在该站 $Re$/$Ma$ 下实算。

    夹具的 $FM\\approx0.28$ 远低于 D-06 的 0.68 门禁，这是刻意的、也不是 bug：§4.1 的参考弦长
    对 80 N 这个量级偏大（叶尖相对来流把型阻与 D-01 基底阻力一起按 $W^3$ 计入功率），而 FM
    门禁属于 M3 优化器的验收对象。可视化层要验收的是"契约字段能否画成正确的图"，夹具越接近
    一次未收敛的中间设计，四个面板反而越能暴露画错的地方。
    """
    au0, al0 = naca_to_cst("0012")
    r_over_R = cosine_span_positions(HUB_OVER_R, N_SECTIONS)
    chord = interpolate_span(R_CTRL, CHORD_CTRL, r_over_R)
    r = r_over_R * RADIUS

    omega = TIP_MACH * A_INF / RADIUS
    v_uniform = float(np.sqrt(THRUST / (2.0 * RHO_INF * np.pi * RADIUS**2)))
    v_i = v_uniform * (1.0 + 0.08 * (1.0 - r_over_R))
    w = np.hypot(omega * r, v_i)
    phi = np.degrees(np.arctan2(v_i, omega * r))
    re = RHO_INF * w * chord / MU_AIR
    ma = w / A_INF

    evaluator = AeroPolarEvaluator("analytic")
    thickness_scale = 1.0 - 0.18 * (r_over_R - HUB_OVER_R) / (1.0 - HUB_OVER_R)
    dz_te = TE_THICKNESS / chord
    shapes = [
        AirfoilShape.from_cst(
            float(thickness_scale[i]) * np.asarray(au0),
            float(thickness_scale[i]) * np.asarray(al0),
            float(dz_te[i]),
        )
        for i in range(N_SECTIONS)
    ]

    alpha, thrust, torque, cl, cd = _calibrate_alpha(evaluator, shapes, chord, w, re, ma, r, phi, r_over_R)
    theta = alpha + phi
    power = torque * omega
    fm = v_uniform * thrust / power

    sections: list[SectionResult] = []
    polars: list[dict[str, list[float]]] = []
    for i in range(N_SECTIONS):
        au = float(thickness_scale[i]) * np.asarray(au0)
        al = float(thickness_scale[i]) * np.asarray(al0)
        curve = evaluator.evaluate(shapes[i], ALPHA_GRID, re=float(re[i]), ma=float(ma[i]))
        moment = evaluator.evaluate(shapes[i], np.array([alpha[i]]), re=float(re[i]), ma=float(ma[i]))
        normalized = generate_flatback_cst(au, al, float(dz_te[i]), n_points=200, chord=1.0)
        physical = generate_flatback_cst(au, al, float(dz_te[i]), n_points=200, chord=float(chord[i]))
        area_norm, x_cg_norm, y_cg_norm, i_xx_norm = _polygon_properties(normalized)

        sections.append(
            SectionResult(
                index=i + 1,
                r_over_R=float(r_over_R[i]),
                r=float(r[i]),
                chord=float(chord[i]),
                theta_deg=float(theta[i]),
                te_thickness=TE_THICKNESS,
                dz_te=float(dz_te[i]),
                alpha_deg=float(alpha[i]),
                cl=float(cl[i]),
                cd=float(cd[i]),
                cm=float(moment.cm[0]),
                re=float(re[i]),
                ma=float(ma[i]),
                au=tuple(float(v) for v in au),
                al=tuple(float(v) for v in al),
                points_normalized=normalized,
                points_mm=_to_mm_points(physical * 1000.0, float(r[i]) * 1000.0, np.radians(float(theta[i]))),
                dN=float(0.5 * RHO_INF * w[i] ** 2 * chord[i] * cl[i]),
                dQ=float(
                    0.5
                    * RHO_INF
                    * w[i] ** 2
                    * chord[i]
                    * (cl[i] * np.sin(np.radians(phi[i])) + cd[i] * np.cos(np.radians(phi[i])))
                    * r[i]
                ),
                v_i=float(v_i[i]),
                phi_deg=float(phi[i]),
                area=area_norm * float(chord[i]) ** 2,
                x_cg=x_cg_norm * float(chord[i]),
                y_cg=y_cg_norm * float(chord[i]),
                i_xx=i_xx_norm * float(chord[i]) ** 4,
                j_torsion=1.35 * i_xx_norm * float(chord[i]) ** 4,
            )
        )
        polars.append(
            {
                "alpha_deg": [float(v) for v in ALPHA_GRID],
                "cl": [float(v) for v in curve.cl],
                "cd": [float(v) for v in curve.cd],
            }
        )

    return RotorResult(
        inputs=RotorInputs(thrust=THRUST, radius=RADIUS, te_thickness=TE_THICKNESS, blade_count=BLADE_COUNT),
        omega=float(omega),
        rpm=float(np.degrees(omega) / 6.0),
        tip_mach=TIP_MACH,
        thrust=thrust,
        power=power,
        torque=torque,
        fm=float(fm),
        ct=thrust / (RHO_INF * np.pi * RADIUS**2 * (omega * RADIUS) ** 2),
        cp=power / (RHO_INF * np.pi * RADIUS**2 * (omega * RADIUS) ** 3),
        power_ideal=float(thrust * v_uniform),
        sections=sections,
        r_over_R=r_over_R,
        chord=chord,
        theta_deg=theta,
        v_i=v_i,
        circulation=0.5 * cl * w * chord,
        cl_distribution=cl,
        polars=polars,
        objective=-float(fm),
        iterations=50,
        convergence_history=[
            {
                "iteration": float(k),
                "objective": -0.55 - 0.13 * k / 50.0,
                "thrust": THRUST * (0.72 + 0.28 * k / 50.0),
                "fm": 0.55 + 0.13 * k / 50.0,
            }
            for k in (0.0, 12.0, 25.0, 38.0, 50.0)
        ],
        seed=7,
        backend="analytic",
        solver_stage="synthetic-fixture",
    )


def _script_blocks(html: str) -> list[str]:
    return re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.S)


def _page_payload(html: str, div_id: str = "hover-rotor-3d") -> str:
    """取 ``Plotly.newPlot(...)`` 所在的数据脚本。

    对产物的任何裸串 grep 都会被 4.8 MB 的库本体污染（库内既有 ``Plotly.newPlot`` 这个导出
    符号、也有 ``topojsonURL`` 之类与页面加载无关的常量），所以先按容器 id 锁定数据脚本，
    外部引用与 trace 检查只落在骨架与数据脚本上。
    """
    blocks = [block for block in _script_blocks(html) if div_id in block]
    assert blocks, "找不到 Plotly.newPlot 数据脚本"
    return max(blocks, key=len)


def _skeleton(html: str) -> str:
    """去掉所有 ``<script>`` 内容后的 HTML 骨架。

    外部资源检查只能在骨架上做：内联的库本体里有 ``href="https://..."`` 这类 JS 字符串
    （maplibre 许可证文本等），它们不是页面属性，不会触发任何网络请求。
    """
    return re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)


def _json_literal(text: str) -> str:
    """plotly 以 ``ensure_ascii=True`` 序列化，中文在 HTML 里落成 ``\\uXXXX`` 转义。

    浏览器解析后照常显示，但 grep 必须比对同一种形态，否则永远匹配不上。
    """
    return json.dumps(text)[1:-1]


def _panel(fig: Figure, code: str) -> Axes:
    """按面板编号取主坐标轴（副轴用 ``twinx`` 生成、被追加在 ``fig.axes`` 末尾，下标不可依赖）。"""
    matches = [ax for ax in fig.axes if ax.get_title(loc="left").startswith(f"({code})")]
    assert len(matches) == 1, f"面板 ({code}) 定位失败，找到 {len(matches)} 个"
    return matches[0]


# --------------------------------------------------------------------------------------
# 夹具自证：合成结果本身得过 §5 红线，否则图版验收毫无意义
# --------------------------------------------------------------------------------------


def test_synthetic_fixture_is_aerodynamically_consistent(synthetic_result: RotorResult) -> None:
    assert synthetic_result.thrust_error < 0.005, "夹具推力未命中目标，面板显示的是坏设计"
    assert 0.0 < synthetic_result.fm < 1.0
    assert synthetic_result.tip_mach <= synthetic_result.inputs.tip_mach_max
    assert all(s.te_thickness == pytest.approx(TE_THICKNESS) for s in synthetic_result.sections)
    assert all(s.cl > 0.0 and s.cd > 0.0 for s in synthetic_result.sections), "出现负升力剖面"
    assert all(abs(s.alpha_deg - (s.theta_deg - s.phi_deg)) < 1e-9 for s in synthetic_result.sections)
    assert synthetic_result.theta_deg[0] > 15.0 and synthetic_result.theta_deg[-1] < 6.0
    assert np.all(np.diff(synthetic_result.theta_deg) < 0.0), "扭转未沿展向递减"
    assert np.all(np.diff(synthetic_result.chord) < 0.0), "弦长未沿展向收缩"


# --------------------------------------------------------------------------------------
# Task 5.1 —— 四联静态图版
# --------------------------------------------------------------------------------------


def test_headless_agg_backend() -> None:
    """验收前置：无显示环境下的确运行在 Agg（conftest 的 autouse 装置生效）。"""
    assert matplotlib.get_backend().lower() == "agg"


def test_summary_plot_written_above_size_gate(tmp_path: Path, synthetic_result: RotorResult) -> None:
    """Agg 下生成 rotor_summary.png：无异常、文件名与体积门禁达标、PIL 可正常读取。"""
    path = generate_summary_plots(synthetic_result, tmp_path)

    assert path.name == FIGURE_BASENAME
    assert path.is_file()
    assert path.stat().st_size > 100 * 1024, "图版小于 100 KB，说明面板被画空或分辨率被压低"

    with Image.open(path) as handle:
        handle.verify()
    with Image.open(path) as image:
        width, height = image.size
        dpi = np.atleast_1d(image.info["dpi"])
    assert width >= MIN_DPI * 12 and height >= MIN_DPI * 8
    assert round(float(dpi[0])) >= MIN_DPI
    assert round(float(dpi[1])) >= MIN_DPI


def test_every_panel_region_carries_ink(tmp_path: Path, synthetic_result: RotorResult) -> None:
    """栅格化后四个象限都必须有真实墨迹：空面板或整块压糊都会在这一步被抓到。"""
    path = generate_summary_plots(synthetic_result, tmp_path)
    with Image.open(path) as handle:
        gray = np.asarray(handle.convert("L"))
    cut_h, cut_w = gray.shape[0] // 2, gray.shape[1] // 2
    quadrants = (
        gray[:cut_h, :cut_w],
        gray[:cut_h, cut_w:],
        gray[cut_h:, :cut_w],
        gray[cut_h:, cut_w:],
    )
    coverage = [float(np.mean(block < 200)) for block in quadrants]
    assert all(0.005 < value < 0.20 for value in coverage), f"四联墨迹覆盖率失衡：{coverage}"


def test_summary_plot_accepts_explicit_file_path(tmp_path: Path, synthetic_result: RotorResult) -> None:
    """给定带后缀的路径时按原样落盘，不强行改名——CLI 需要能指定自定义输出名。"""
    target = tmp_path / "campaign" / "case_A.png"
    assert generate_summary_plots(synthetic_result, target) == target
    assert target.is_file()


def test_four_panels_all_have_legend_and_unit_axis_labels(synthetic_result: RotorResult) -> None:
    """四联图版每一幅都必须有图例 + 带物理单位的坐标轴标签（计划.md 5.1 验收判据）。"""
    fig = build_summary_figure(synthetic_result)
    try:
        axes = list(fig.axes)
        assert len(axes) >= 4, "面板数不足，四联图版被降级"

        for ax in axes:
            legend = ax.get_legend()
            assert legend is not None, f"{ax.get_title()!r} 缺图例"
            assert [text.get_text() for text in legend.get_texts()], f"{ax.get_title()!r} 图例为空"
            for label in (ax.get_xlabel(), ax.get_ylabel()):
                assert label, f"{ax.get_title()!r} 有坐标轴缺标签"
                assert "[" in label, f"坐标轴标签 {label!r} 未标注物理单位"

        positions = {tuple(np.round(ax.get_position().bounds, 3)) for ax in axes}
        assert len(positions) >= 4, "子图位置重叠，未真正分成四联"

        contour_ax, polar_ax = _panel(fig, "a"), _panel(fig, "d")
        outlines = [line for line in contour_ax.get_lines() if line.get_xydata().shape[0] >= 20]
        assert len(outlines) == N_SECTIONS, "截面叠合图未画满十个剖面"
        assert any("钝尾缘" in text.get_text() for text in contour_ax.get_legend().get_texts())
        assert any("工作点" in text.get_text() for text in polar_ax.get_legend().get_texts())
        assert "c_l" in polar_ax.get_ylabel()
        assert fig._suptitle is not None and "FM" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_spanwise_panels_carry_chord_twist_and_inflow(synthetic_result: RotorResult) -> None:
    """面板 (b)/(c) 的四个展向量来自结果契约，并被真正画进对应坐标轴。"""
    fig = build_summary_figure(synthetic_result)
    try:
        joined = "".join(ax.get_xlabel() + ax.get_ylabel() for ax in fig.axes)
        for token in ("弦长", "扭转角", "环量", "诱导速度", "r/R", "物理半径"):
            assert token in joined, f"缺少展向标注 {token}"

        chord_ax = _panel(fig, "b")
        values = _line_data(chord_ax, min_points=2)[:, 1]
        assert values.min() == pytest.approx(float(synthetic_result.chord.min()), rel=1e-9)
        assert values.max() == pytest.approx(float(synthetic_result.chord.max()), rel=1e-9)

        vi_ax = next(ax for ax in fig.axes if "诱导速度" in ax.get_ylabel())
        vi_values = _line_data(vi_ax, min_points=2)[:, 1]
        assert vi_values.min() == pytest.approx(float(synthetic_result.v_i.min()), rel=1e-9)
    finally:
        plt.close(fig)


def test_polar_panel_marks_every_section_design_point(synthetic_result: RotorResult) -> None:
    """面板 (d) 的十个设计工作点必须逐个落在各自的剖面极线上。"""
    fig = build_summary_figure(synthetic_result)
    try:
        polar_ax = _panel(fig, "d")
        curves = _line_data(polar_ax, min_points=20)
        for section in synthetic_result.sections:
            distance = np.hypot(curves[:, 0] - section.cd, curves[:, 1] - section.cl).min()
            # 极线按 0.5 deg 间距采样，设计迎角不落在格点上，容差取半步距内的曲线偏移量。
            assert distance < 0.06, "设计工作点落在剖面极线之外，工作点与 polars 不自洽"
        marked = [line for line in polar_ax.get_lines() if line.get_xydata().shape[0] == 1]
        assert len(marked) >= N_SECTIONS
    finally:
        plt.close(fig)


def test_dpi_below_gate_is_rejected(tmp_path: Path, synthetic_result: RotorResult) -> None:
    """低于 300 DPI 的请求被显式拒绝，而不是悄悄产出一张不合格图。"""
    with pytest.raises(ValueError, match="300"):
        generate_summary_plots(synthetic_result, tmp_path, dpi=150)


def test_empty_sections_are_reported_not_skipped(tmp_path: Path, synthetic_result: RotorResult) -> None:
    with pytest.raises(ValueError, match="sections"):
        generate_summary_plots(replace(synthetic_result, sections=[]), tmp_path)


def test_figure_survives_missing_polars(synthetic_result: RotorResult) -> None:
    """极线未落盘（快速路径常见）时面板 (d) 退化为工作点割线，标注依旧完备。"""
    fig = build_summary_figure(replace(synthetic_result, polars=[]))
    try:
        polar_ax = _panel(fig, "d")
        assert polar_ax.get_legend() is not None
        assert polar_ax.get_xlabel() and polar_ax.get_ylabel()
        assert len([line for line in polar_ax.get_lines() if line.get_xydata().shape[0] == 2]) >= N_SECTIONS
    finally:
        plt.close(fig)


def test_chord_frame_recovered_from_physical_mm_points(synthetic_result: RotorResult) -> None:
    """叠合图对 ``points_mm`` 的旋转/平移不变：主轴分解能还原单位弦长与钝尾缘厚度。"""
    section = synthetic_result.sections[0]
    blind = replace(section, points_normalized=np.zeros((0, 2)))
    psi, zeta = section_outline_in_chord_frame(blind)

    assert psi.min() == pytest.approx(0.0, abs=1e-9)
    assert psi.max() == pytest.approx(1.0, abs=1e-9)
    assert zeta[-1] - zeta[0] == pytest.approx(section.dz_te, rel=0.06)
    direct_psi, direct_zeta = section_outline_in_chord_frame(section)
    assert np.max(np.abs(direct_psi - psi)) < 0.02
    assert np.max(np.abs(direct_zeta - zeta)) < 0.02


# --------------------------------------------------------------------------------------
# Task 5.2 —— Plotly 交互三维
# --------------------------------------------------------------------------------------


def test_orient_convention_puts_leading_edge_above_trailing_edge() -> None:
    """正安装角必须让前缘高于后缘（相对风沿 $-Y$），否则整片桨叶是反扭转的。"""
    xi = np.array([[0.0, 1.0]])
    zeta = np.array([[0.0, 0.0]])
    placed = _orient(xi, zeta, np.array([0.5]), np.array([1.0]), np.array([np.radians(15.0)]))
    assert placed[0, 0, 0] == pytest.approx(0.5)
    assert placed[0, 1, 1] == pytest.approx(np.cos(np.radians(15.0)))
    assert placed[0, 1, 2] == pytest.approx(-np.sin(np.radians(15.0)))


def test_loft_surface_geometry(synthetic_result: RotorResult) -> None:
    """放样曲面：结构化网格、站位端点严格落在 r_hub 与 R、厚度与扭转沿展向递减。"""
    surface = build_blade_surface(synthetic_result, n_chord=24, n_span=31)

    assert surface.vertices.shape == (31, 24, 3)
    assert surface.loading.shape == (31, 24)
    assert surface.station_outlines.shape[0] == N_SECTIONS
    assert surface.r_over_R[0] == pytest.approx(HUB_OVER_R, rel=1e-6)
    assert surface.r_over_R[-1] == pytest.approx(1.0, rel=1e-6)

    root, tip = surface.vertices[0], surface.vertices[-1]
    assert root[0, 0] == pytest.approx(synthetic_result.sections[0].r, abs=1e-6)
    assert tip[0, 0] == pytest.approx(RADIUS, abs=1e-6)

    chord_root = synthetic_result.sections[0].chord
    assert 0.115 * chord_root < _minor_extent(root) < 0.145 * chord_root, "放样改掉了剖面厚度"

    # 弦线在 Y-Z 平面内的斜率即 -tan(theta)：这是扭转被真正画进三维的硬证据。
    slope_root = float(np.polyfit(root[:, 1], root[:, 2], 1)[0])
    slope_tip = float(np.polyfit(tip[:, 1], tip[:, 2], 1)[0])
    theta_root = np.radians(synthetic_result.sections[0].theta_deg)
    assert slope_root == pytest.approx(-np.tan(theta_root), abs=0.08)
    assert abs(slope_root) > abs(slope_tip), "扭转未沿展向递减，放样把 §4.1 的扭转规律画反了"
    assert np.ptp(surface.loading[:, 0]) > 0.0, "法向载荷沿展向无变化，色谱成了装饰而非数据"


def test_interactive_html_is_single_file_offline(tmp_path: Path, synthetic_result: RotorResult) -> None:
    """HTML 含 plotly-graph-div、至少一条 trace、体积 <5 MB，且没有任何外部资源引用。"""
    path = render_interactive_rotor_html(synthetic_result, tmp_path)
    assert path.name == "rotor_interactive_3d.html"
    size = path.stat().st_size
    assert size < GATE_MAX_HTML_BYTES, f"单文件 HTML {size:,} B 顶破 5 MB 门禁"

    html = path.read_text(encoding="utf-8")
    assert "plotly-graph-div" in html
    payload = _page_payload(html)
    assert "Plotly.newPlot" in payload
    assert payload.count('"type":"') >= 1
    assert payload.count('"type":"mesh3d"') >= 1

    blocks = _script_blocks(html)
    assert max(len(block) for block in blocks) > 1_000_000, "plotly.js 未被内联，产物离开 Python 环境打不开"
    skeleton = _skeleton(html)
    assert re.search(r"<(?:script|link|img|iframe)[^>]*\s(?:src|href)=", skeleton) is None, skeleton[:400]
    assert re.search(r"(?:src|href)\s*=\s*['\"]https?:", skeleton) is None
    assert "http" not in skeleton, "页面骨架引用网络地址，离线打开会白屏"
    assert "cdn.plot.ly" not in payload


def test_rotor_figure_traces_carry_mesh_and_loading(synthetic_result: RotorResult) -> None:
    """mesh3d trace 的顶点/索引/强度场长度自洽，色谱与悬停读数带物理量。"""
    fig = build_rotor_figure(synthetic_result)
    meshes = [trace for trace in fig.data if trace.type == "mesh3d"]
    assert len(fig.data) >= 1
    assert len(meshes) == 1

    mesh = meshes[0]
    n_vertices = len(mesh.x)
    assert n_vertices == DEFAULT_N_SPAN * DEFAULT_N_CHORD
    assert len(mesh.y) == len(mesh.z) == len(mesh.intensity) == len(mesh.customdata) == n_vertices
    assert len(mesh.i) == len(mesh.j) == len(mesh.k) == 2 * (DEFAULT_N_SPAN - 1) * DEFAULT_N_CHORD
    assert max(mesh.i) < n_vertices and min(mesh.i) >= 0
    assert mesh.colorbar.title.text.startswith("局部法向压力")
    assert "r/R=" in mesh.hovertemplate

    outlines = [trace for trace in fig.data if trace.type == "scatter3d"]
    assert outlines and any("截面轮廓" in str(trace.name) for trace in outlines)
    for axis in (fig.layout.scene.xaxis, fig.layout.scene.yaxis, fig.layout.scene.zaxis):
        assert "[" in axis.title.text, "三维坐标轴缺物理单位"


def test_blade_replication_stays_within_size_gate(tmp_path: Path, synthetic_result: RotorResult) -> None:
    """按 inputs.blade_count 复制整副旋翼时，自动降采样仍守住 5 MB 门禁。"""
    path = render_interactive_rotor_html(synthetic_result, tmp_path / "rotor", blade_count=None)
    payload = _page_payload(path.read_text(encoding="utf-8"))
    assert payload.count('"type":"mesh3d"') == synthetic_result.inputs.blade_count
    assert path.stat().st_size < GATE_MAX_HTML_BYTES


def test_mesh_points_can_be_rendered_directly(tmp_path: Path, synthetic_result: RotorResult) -> None:
    """Task 5.2 的备选输入：调用方自算好的 (n_span, n_chord, 3) 节点阵与压力场标量。"""
    surface: BladeSurface = build_blade_surface(synthetic_result, n_chord=18, n_span=24)
    path = render_interactive_rotor_html(
        surface.vertices, tmp_path, loading=surface.loading, blade_count=1, title="自算网格直通渲染"
    )
    html = path.read_text(encoding="utf-8")
    assert "plotly-graph-div" in html
    assert _json_literal("自算网格直通渲染") in html
    assert _page_payload(html).count('"type":"mesh3d"') == 1
    assert path.stat().st_size < GATE_MAX_HTML_BYTES


def test_mesh_points_shape_and_loading_are_validated(synthetic_result: RotorResult) -> None:
    surface = build_blade_surface(synthetic_result, n_chord=18, n_span=24)
    with pytest.raises(TypeError, match="n_span"):
        render_interactive_rotor_html(surface.vertices[..., :2], Path("unused"))
    with pytest.raises(ValueError, match="loading"):
        render_interactive_rotor_html(surface.vertices, Path("unused"), loading=np.zeros(3))


def test_loading_override_by_section_vector(tmp_path: Path, synthetic_result: RotorResult) -> None:
    """用剖面升力系数作色谱（Task 5.2 允许的另一份载荷场）也能落盘。"""
    cl = np.array([s.cl for s in synthetic_result.sections])
    path = render_interactive_rotor_html(
        synthetic_result, tmp_path, loading=cl, loading_label="剖面升力系数 cl  [−]"
    )
    assert _json_literal("剖面升力系数") in path.read_text(encoding="utf-8")
    assert path.stat().st_size < GATE_MAX_HTML_BYTES


def test_result_without_sections_cannot_be_lofted(synthetic_result: RotorResult) -> None:
    with pytest.raises(ValueError, match="sections"):
        build_blade_surface(replace(synthetic_result, sections=[]))


def test_blade_count_and_resolutions_are_validated(synthetic_result: RotorResult) -> None:
    with pytest.raises(ValueError, match="周向节点数"):
        build_blade_surface(synthetic_result, n_chord=6)
    with pytest.raises(ValueError, match="展向站点数"):
        build_blade_surface(synthetic_result, n_span=3)
    with pytest.raises(ValueError, match="叶片数"):
        render_interactive_rotor_html(synthetic_result, Path("unused"), blade_count=0)
