"""可视化层：Matplotlib 多联气动图版与 Plotly 三维交互看板。"""

from hover_rotor.visualization.plot_rotor_3d import (
    HTML_BASENAME,
    BladeSurface,
    build_blade_surface,
    build_rotor_figure,
    render_interactive_rotor_html,
)
from hover_rotor.visualization.plot_sections import (
    FIGURE_BASENAME,
    build_summary_figure,
    generate_summary_plots,
    resolve_figure_path,
    section_outline_in_chord_frame,
)

__all__ = [
    "FIGURE_BASENAME",
    "HTML_BASENAME",
    "BladeSurface",
    "build_blade_surface",
    "build_rotor_figure",
    "build_summary_figure",
    "generate_summary_plots",
    "render_interactive_rotor_html",
    "resolve_figure_path",
    "section_outline_in_chord_frame",
]
