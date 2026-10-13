"""气动特性与截面型线四联静态图版（计划.md Task 5.1）。

优化器的终态是一堆数字：十截面的几何/气动档案加上整机 T、P、FM。数字能判定收敛，
却回答不了工程判读问题——反设计出的桨叶**在物理上是否成立**：

* 十个钝尾缘截面叠合在一起，型线是否光滑过渡、尾缘钝度是否真的全展恒定；
* 弦长与扭转沿展向是否单调收缩（非单调意味着 PCHIP 或优化边界出了问题）；
* 环量与下洗速度是否接近理想线形（下洗沿展向越均匀，诱导损失越低，FM 越接近上限）；
* 各剖面工作点离失速边界还有多远（极线 + 等升阻比射线直接读出裕度）。

本模块把这四项判读一次性落盘成单张 ``rotor_summary.png``（>=300 DPI 静态图版），
供 CLI 摘要、CI 产物与报告插图直接引用。

契约边界（计划.md §4.2）：只读 :class:`~hover_rotor.config.RotorResult` 与其
``SectionResult`` 列表的冻结字段，绝不依赖优化器内部结构，因此 M3 与 M5 可并行开发。
matplotlib 后端由调用方（或 CI 的 ``Agg``）决定，库内不强制切换。
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import NDArray

from hover_rotor.config import RotorResult, SectionResult

FIGURE_BASENAME = "rotor_summary.png"
"""默认落盘文件名，与计划.md Task 5.1 的输出契约一致。"""

MIN_DPI = 300
"""计划.md Task 5.1 的分辨率门禁：低于此值的报告用图版被拒绝。"""

_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Hiragino Sans GB",
    "WenQuanYi Micro Hei",
    "DengXian",
    "SimSun",
)
"""图版标注是中文，而 matplotlib 自带的 DejaVu Sans 没有 CJK 字形。按此顺序在系统里
找一个真实存在的中文字体并烘焙进每个文本对象，否则中文全部退化成方框。"""

FloatArray = NDArray[np.float64]

_POLAR_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "alpha": ("alpha_deg", "alpha", "aoa_deg", "a_deg"),
    "cl": ("cl", "c_l", "lift"),
    "cd": ("cd", "c_d", "drag"),
    "ld": ("ld", "l_d", "cl_cd", "lift_over_drag"),
}
"""``RotorResult.polars`` 条目的键名容错表：极线层与优化器的命名习惯不统一，
绘图层按别名顺序取值，避免把可视化绑死在某个字面量上。"""


# --------------------------------------------------------------------------------------
# 文本字体
# --------------------------------------------------------------------------------------


@cache
def _available_cjk_font() -> str | None:
    """返回系统里第一个可解析到的中文字体族名，全都没有时给 None。"""
    for family in _CJK_FONT_CANDIDATES:
        try:
            fontManager.findfont(FontProperties(family=family), fallback_to_default=False)
        except RuntimeError:
            continue
        return family
    return None


def _apply_text_font(fig: Figure) -> None:
    """把中文字体烘焙进图内所有文本对象（含图例、刻度、标注）。

    在对象层设置而不是改全局 ``rcParams``：可视化库不该污染进程的字体配置，且返回的
    Figure 之后被谁、在什么 rc 状态下二次绘制都不可控。mathtext（$\\psi$、$c_l$ 等）
    走独立的 ``mathtext.fontset``，不受本函数影响，因此中英混排不会出现断字。
    """
    family = _available_cjk_font()
    if family is None:
        warnings.warn(
            "系统未安装可用的中文字体，图版中的中文标注会显示为方框；"
            "请安装 fonts-noto-cjk（Debian/Ubuntu）或微软雅黑/黑体（Windows）后重新出图",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    for text in fig.findobj(Text):
        props = text.get_fontproperties()
        text.set_fontproperties(
            FontProperties(
                family=[family, "sans-serif"],
                size=props.get_size(),
                weight=props.get_weight(),
                style=props.get_style(),
            )
        )


# --------------------------------------------------------------------------------------
# 几何与展向数据读取
# --------------------------------------------------------------------------------------


def _coerce_points(values: Any) -> FloatArray:
    """把结果契约里的点阵字段规整为 ``(M, 2)`` 浮点数组，形状不合法时返回空数组。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        return np.zeros((0, 2))
    return arr


def _to_unit_chord(xi: FloatArray, zeta: FloatArray) -> tuple[FloatArray, FloatArray]:
    """弦向坐标平移至零起点并按弦向跨度归一，得到无量纲弦向坐标系。"""
    span = float(np.max(xi) - np.min(xi))
    if span <= 0.0:
        raise ValueError("截面型面在弦向退化成一个点，无法归一化")
    return (xi - float(np.min(xi))) / span, zeta / span


def section_outline_in_chord_frame(section: SectionResult) -> tuple[FloatArray, FloatArray]:
    """返回截面在**自身弦向坐标系**下的闭合型线 ``(psi, zeta)``，弦长归一为 1。

    Task 5.1 的截面叠合图与 Task 5.2 的放样蒙皮都需要"去掉安装角旋转与展向平移"的
    剖面形状，故在此一次完成提取，两个模块共用同一实现，避免各自解释 ``points_mm``
    的坐标约定而蒙出两份不一致的几何。

    取值优先级：

    1. ``points_normalized``：几何层原生输出 $(\\psi,\\zeta)$，直接采用；
    2. ``points_mm``：物理毫米点阵，已绕桨轴旋转 ``theta_deg`` 并平移至站位 r。
       其朝向与原点由上游导出层决定，本函数用主轴分解把弦线找回，故对旋转/平移不变。

    两者都不可用时抛 ``ValueError``——空型面意味着结果契约已被破坏，静默画空图比报错更糟。
    """
    pts = _coerce_points(section.points_normalized)
    if pts.shape[0] >= 12:
        return _to_unit_chord(pts[:, 0], pts[:, 1])

    phys = _coerce_points(section.points_mm)
    if phys.shape[0] < 12:
        raise ValueError(
            f"截面 {section.index} 无可用于绘图的型面点阵"
            f"（points_normalized={pts.shape[0]} 点、points_mm={phys.shape[0]} 点）"
        )
    shifted = phys - phys.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(shifted.T @ shifted)
    major = eigenvectors[:, -1]
    # 弦线有两个候选指向，靠型线自身的遍历方向定夺：CST 回路从尾缘下沿出发先走向前缘
    # （cst.py 的点序约定），故首段位移的反方向就是 LE->TE。取定 major 后 minor 必须用
    # 右手系跟随，不能再单独翻 zeta——翻分量等于把剖面做镜像，弯度会被上下颠倒。
    if float((shifted[1] - shifted[0]) @ major) > 0.0:
        major = -major
    xi = shifted @ major
    zeta = shifted @ np.array([-major[1], major[0]])
    return _to_unit_chord(xi, zeta)


def _trailing_edge_points(psi: FloatArray, zeta: FloatArray) -> FloatArray:
    """钝尾缘端面角点：弦向坐标落在 99.9% 弦长以外的采样点。"""
    tail = psi >= 0.999
    return np.column_stack((psi[tail], zeta[tail]))


def _spanwise(rotor_array: NDArray[np.float64], sections: Sequence[SectionResult], attr: str) -> FloatArray:
    """优先取整机展向数组，长度与截面数不符时回退到逐截面字段。

    优化器的中间迭代态可能只填了 ``SectionResult`` 而未汇总展向数组；绘图层既不空白
    也不因此中止整张图版。
    """
    arr = np.asarray(rotor_array, dtype=np.float64).ravel()
    if arr.size == len(sections):
        return arr
    return np.array([float(getattr(section, attr)) for section in sections], dtype=np.float64)


def _circulation_spanwise(result: RotorResult, sections: Sequence[SectionResult]) -> FloatArray:
    """展向环量 $\\Gamma=\\tfrac12 c_l W c$ [m$^2$/s]。

    ``RotorResult.circulation`` 缺席时由叶素量重建：合速度 $W=\\sqrt{(\\Omega r)^2+v_i^2}$
    忽略型阻切向分量，与 BEMT 的环量定义同阶。
    """
    arr = np.asarray(result.circulation, dtype=np.float64).ravel()
    if arr.size == len(sections):
        return arr
    r = _spanwise(result.r_over_R, sections, "r") * result.inputs.radius
    v_i = _spanwise(result.v_i, sections, "v_i")
    w = np.hypot(result.omega * r, v_i)
    cl = _spanwise(result.cl_distribution, sections, "cl")
    chord = _spanwise(result.chord, sections, "chord")
    return 0.5 * cl * w * chord


def _polar_series(polar: Mapping[str, Any] | None) -> dict[str, FloatArray] | None:
    """把 ``RotorResult.polars`` 的单条记录解成 ``{alpha, cl, cd, ld}`` 数组，不可用时返回 None。"""
    if not isinstance(polar, Mapping):
        return None

    def pick(quantity: str) -> FloatArray | None:
        for key in _POLAR_KEY_ALIASES[quantity]:
            if key in polar:
                arr = np.asarray(polar[key], dtype=np.float64).ravel()
                if arr.size >= 2:
                    return arr
        return None

    cl = pick("cl")
    cd = pick("cd")
    if cl is None or cd is None or cl.size != cd.size:
        return None
    alpha = pick("alpha")
    if alpha is None or alpha.size != cl.size:
        alpha = np.linspace(0.0, float(cl.size - 1), cl.size)
    ld = pick("ld")
    if ld is None or ld.size != cl.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            ld = np.divide(cl, cd, out=np.zeros_like(cl), where=cd > 0.0)
    return {"alpha": alpha, "cl": cl, "cd": cd, "ld": ld}


# --------------------------------------------------------------------------------------
# 四个面板
# --------------------------------------------------------------------------------------


def _panel_section_contours(ax: Axes, sections: Sequence[SectionResult], colors: NDArray[np.float64]) -> None:
    """面板 (a)：十钝尾缘截面按各自弦长归一后前缘对齐叠合。"""
    handles: list[Any] = []

    for section, color in zip(sections, colors, strict=True):
        psi, zeta = section_outline_in_chord_frame(section)
        (line,) = ax.plot(
            psi,
            zeta,
            color=color,
            lw=1.3,
            solid_capstyle="round",
            label=f"Sec {section.index:02d}  r/R={section.r_over_R:.3f}",
        )
        handles.append(line)
        te = _trailing_edge_points(psi, zeta)
        if te.shape[0] >= 2:
            ax.plot(te[:, 0], te[:, 1], ls="none", marker="s", ms=3.2, mew=0.9, mfc="none", color=color)

    (te_proxy,) = ax.plot(
        [], [], ls="none", marker="s", ms=3.2, mew=0.9, mfc="none", color="0.35", label="钝尾缘端面"
    )
    handles.append(te_proxy)

    # adjustable="datalim"：保持子图盒子不变、靠扩数据范围来满足等比例，
    # 于是图例自然落在上下留白里，不会压住型线。
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("弦向归一化坐标  $\\psi = x/c$  [−]")
    ax.set_ylabel("厚度坐标  $\\zeta = z/c$  [−]")
    ax.set_title(f"(a)  {len(sections)} 个钝尾缘截面对齐叠合型线", loc="left")
    ax.grid(alpha=0.28, lw=0.6)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.legend(handles=handles, fontsize=5.8, ncol=2, loc="upper right", framealpha=0.6)


def _panel_spanwise_pair(
    ax: Axes,
    r_over_R: FloatArray,
    radius: float,
    first: FloatArray,
    first_label: str,
    second: FloatArray,
    second_label: str,
    title: str,
    *,
    legend_locs: tuple[str, str] = ("lower left", "lower right"),
) -> None:
    """面板 (b)/(c)：同一站位轴上的双 y 轴展向分布，物理半径 r 由顶轴刻度直读。"""
    cmap = plt.get_cmap("viridis")
    xs = np.asarray(r_over_R, dtype=np.float64)

    (line1,) = ax.plot(xs, first, "-o", color=cmap(0.12), lw=1.9, ms=4.8)
    ax.set_xlabel("无量纲展向站位  $r/R$  [−]")
    ax.set_ylabel(first_label)
    ax.tick_params(axis="y", labelcolor=cmap(0.12))

    twin: Axes = ax.twinx()
    twin.plot(xs, second, "-s", color=cmap(0.82), lw=1.9, ms=4.8)
    twin.set_ylabel(second_label)
    twin.tick_params(axis="y", labelcolor=cmap(0.82))
    # 顶轴复用同一组站位刻度，只把刻度值换算成物理半径，读者不必再手乘 R。
    twin.xaxis.set_label_position("top")
    twin.xaxis.set_ticks_position("top")
    twin.set_xticks(xs, [f"{value * radius:.3f}" for value in xs], fontsize=6.2)
    twin.set_xlabel(f"物理半径  $r=(r/R)\\cdot R$  [m]（$R$={radius:g} m）")

    ax.set_title(title, loc="left")
    ax.grid(alpha=0.28, lw=0.6)
    ax.legend([line1], [first_label.split("[")[0].strip()], fontsize=6.6, loc=legend_locs[0], framealpha=0.7)
    twin.legend(
        [Line2D([], [], color=cmap(0.82), marker="s", lw=1.9, ms=4.8)],
        [second_label.split("[")[0].strip()],
        fontsize=6.6,
        loc=legend_locs[1],
        framealpha=0.7,
    )


def _panel_polars(
    ax: Axes,
    sections: Sequence[SectionResult],
    polars: Sequence[Any] | None,
    colors: NDArray[np.float64],
) -> None:
    """面板 (d)：剖面 $c_l$–$c_d$ 极线 + 等升阻比射线 + 各剖面设计工作点。"""
    curves: list[dict[str, FloatArray] | None] = []
    if isinstance(polars, Sequence) and not isinstance(polars, (str, bytes)):
        curves = [_polar_series(item) for item in polars]
    curves = (curves + [None] * len(sections))[: len(sections)]

    handles: list[Any] = []
    max_cd = 0.02
    max_ld = 2.0

    for index, (section, color, curve) in enumerate(zip(sections, colors, curves, strict=True)):
        name = f"Sec {section.index:02d}  r/R={section.r_over_R:.3f}"
        if curve is None:
            # 极线缺失（例如快速路径未落盘 polars）时退化为零阻到工作点的割线，
            # 保住剖面身份，不在图版上凭空留白。
            secant_cd = [max(section.cd * 0.05, 1.0e-4), section.cd]
            (line,) = ax.plot(secant_cd, [0.0, section.cl], lw=1.2, color=color, alpha=0.85, label=name)
        else:
            (line,) = ax.plot(curve["cd"], curve["cl"], lw=1.2, color=color, alpha=0.85, label=name)
            max_cd = max(max_cd, float(np.max(curve["cd"])))
            max_ld = max(max_ld, float(np.max(curve["ld"])))
        handles.append(line)

        ax.plot(section.cd, section.cl, ls="none", marker="o", ms=5.6, mfc=color, mec="black", mew=0.6)
        ax.annotate(
            f"{section.ld:.1f}",
            (section.cd, section.cl),
            fontsize=5.6,
            color="0.2",
            textcoords="offset points",
            xytext=(4, 4 if index % 2 == 0 else -10),
        )

    # 等升阻比射线 $c_l=(L/D)c_d$：极线上任一点到原点连线的斜率即该点 L/D，
    # 有了射线网格，升阻比与失速裕度都能直读而不必换算。
    for level in [round(float(v), 1) for v in np.linspace(0.35 * max_ld, max_ld, 4)]:
        ax.plot([0.0, max_cd], [0.0, level * max_cd], ls=":", lw=0.8, color="0.5")
    (ray_proxy,) = ax.plot([], [], ls=":", lw=0.9, color="0.5", label="等升阻比射线 $c_l=(L/D)\\,c_d$")
    design_label = "设计工作点（旁注 $L/D$）"
    (point_proxy,) = ax.plot(
        [], [], ls="none", marker="o", ms=5.6, mfc="0.7", mec="black", mew=0.6, label=design_label
    )
    handles += [ray_proxy, point_proxy]

    ax.set_xlabel("阻力系数  $c_d$  [−]")
    ax.set_ylabel("升力系数  $c_l$  [−]")
    ax.set_title("(d)  剖面升阻比极线与设计工作点", loc="left")
    ax.grid(alpha=0.28, lw=0.6)
    ax.legend(handles=handles, fontsize=5.7, ncol=3, loc="upper right", framealpha=0.6)


# --------------------------------------------------------------------------------------
# 公共入口
# --------------------------------------------------------------------------------------


def resolve_figure_path(output_path: str | Path, basename: str = FIGURE_BASENAME) -> Path:
    """把 ``output_path`` 解释成最终文件路径：目录或无后缀路径补齐默认文件名。

    CLI 传的是 ``--output-dir``，而本模块契约是一张具体图版文件，两种用法都必须成立，
    故在此统一收敛并保证父目录存在。
    """
    path = Path(output_path)
    if path.suffix == "" or path.is_dir():
        path = path / basename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_summary_figure(
    result: RotorResult,
    *,
    figsize: tuple[float, float] = (15.0, 10.5),
) -> Figure:
    """构造四联图版 :class:`~matplotlib.figure.Figure`，不落盘。

    与 :func:`generate_summary_plots` 拆开，是为了让测试与报告层能直接检视坐标轴、
    图例与单位标注——PNG 是只读产物，无法验证"标注完备"这条验收判据。

    面板定位约定：四个主面板的标题分别以 ``(a)``/``(b)``/``(c)``/``(d)`` 开头，用
    ``ax.get_title(loc="left")`` 检索。不要用 ``fig.axes`` 的下标——弦长/扭转与环量/下洗
    各带一个 ``twinx`` 副轴，它们被 matplotlib 追加在列表末尾，下标顺序与视觉顺序不一致。
    """
    sections = list(result.sections)
    if not sections:
        raise ValueError("结果中没有截面档案（sections 为空），四联图版无从绘制")

    count = len(sections)
    colors = plt.get_cmap("viridis")(np.linspace(0.02, 0.94, count))
    r_over_R = _spanwise(result.r_over_R, sections, "r_over_R")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(left=0.065, right=0.955, top=0.895, bottom=0.07, hspace=0.62, wspace=0.36)

    _panel_section_contours(axes[0, 0], sections, colors)
    _panel_spanwise_pair(
        axes[0, 1],
        r_over_R,
        result.inputs.radius,
        _spanwise(result.chord, sections, "chord"),
        "弦长  $c(r)$  [m]",
        _spanwise(result.theta_deg, sections, "theta_deg"),
        "安装扭转角  $\\theta(r)$  [deg]",
        "(b)  展向弦长–扭转演变",
    )
    _panel_spanwise_pair(
        axes[1, 0],
        r_over_R,
        result.inputs.radius,
        _circulation_spanwise(result, sections),
        "环量  $\\Gamma(r)$  [m$^2$/s]",
        _spanwise(result.v_i, sections, "v_i"),
        "诱导速度  $v_i(r)$  [m/s]",
        "(c)  展向环量与下洗速度分布",
        legend_locs=("upper right", "lower right"),
    )
    _panel_polars(axes[1, 1], sections, result.polars, colors)

    tip = sections[0]
    fig.suptitle(
        "悬停旋翼反设计终态气动总览    "
        f"$T$={result.thrust:.1f} N（目标 {result.inputs.thrust:.1f} N）    "
        f"$FM$={result.fm:.3f}    $C_T$={result.ct:.4f}    $C_P$={result.cp:.4f}    "
        f"$\\Omega$={result.rpm:.0f} rpm    $M_{{tip}}$={result.tip_mach:.3f}    "
        f"$t_{{te}}$={tip.te_thickness * 1000:.2f} mm（全展恒定）    "
        f"{count} 截面 / {result.inputs.blade_count} 桨叶",
        x=0.012,
        y=0.965,
        ha="left",
        fontsize=9.0,
    )
    _apply_text_font(fig)
    return fig


def generate_summary_plots(
    result: RotorResult,
    output_path: str | Path,
    *,
    dpi: int = MIN_DPI,
    figsize: tuple[float, float] = (15.0, 10.5),
    facecolor: str = "white",
) -> Path:
    """生成 ``rotor_summary.png`` 四联气动图版并返回落盘路径（计划.md Task 5.1）。

    面板构成：(a) 十钝尾缘截面对齐叠合型线；(b) 展向弦长与安装扭转演变（双 y 轴、
    物理单位，顶轴直读 r [m]）；(c) 展向环量与诱导速度 $v_i$（双 y 轴）；
    (d) 剖面升阻比极线，含等 $L/D$ 射线网格与每个剖面的设计工作点标注。

    Args:
        result: 反设计终态结果（只读冻结结果契约）。
        output_path: 目标文件路径；给目录或无后缀路径时自动补 ``rotor_summary.png``。
        dpi: 栅格分辨率，不得低于 :data:`MIN_DPI`（300）。
        figsize: 画布尺寸 [inches]。
        facecolor: 画布背景色。

    Returns:
        实际写出的文件 :class:`~pathlib.Path`。

    Raises:
        ValueError: ``dpi`` 低于验收门禁，或截面型面数据缺失。
    """
    if dpi < MIN_DPI:
        raise ValueError(
            f"输出分辨率 {dpi} DPI 低于计划.md Task 5.1 门禁 {MIN_DPI} DPI；请提高 dpi 而非放宽验收"
        )

    path = resolve_figure_path(output_path)
    fig = build_summary_figure(result, figsize=figsize)
    try:
        fig.savefig(path, dpi=int(dpi), facecolor=facecolor)
    finally:
        # 无头批处理（CLI 一次跑多工况）里图对象不释放会持续吃内存。
        plt.close(fig)
    return path


__all__ = [
    "FIGURE_BASENAME",
    "MIN_DPI",
    "build_summary_figure",
    "generate_summary_plots",
    "resolve_figure_path",
    "section_outline_in_chord_frame",
]
