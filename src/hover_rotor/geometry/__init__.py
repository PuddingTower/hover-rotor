"""几何底层：钝尾缘 CST 型线发生器与展向非线性离散。"""

from hover_rotor.geometry.cst import (
    bernstein_basis,
    class_function,
    fit_shape_weights,
    generate_flatback_cst,
    relative_te_thickness,
    shape_function,
    trailing_edge_gap,
)
from hover_rotor.geometry.discretization import (
    SpanDistribution,
    build_span_distribution,
    cosine_span_positions,
    discretize_rotor_span,
    interpolate_span,
)

__all__ = [
    "bernstein_basis",
    "class_function",
    "fit_shape_weights",
    "generate_flatback_cst",
    "relative_te_thickness",
    "shape_function",
    "trailing_edge_gap",
    "SpanDistribution",
    "build_span_distribution",
    "cosine_span_positions",
    "discretize_rotor_span",
    "interpolate_span",
]
