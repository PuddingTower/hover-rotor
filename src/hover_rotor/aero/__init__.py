"""气动内核：基底阻力修正、可微极线代理与悬停 BEMT 求解器。"""

from hover_rotor.aero.base_drag import apply_base_drag, compute_base_drag_hoerner
from hover_rotor.aero.polar import (
    AeroPolarEvaluator,
    AirfoilShape,
    AnalyticPolarBackend,
    Polar,
    trailing_edge_drag_gain,
    trailing_edge_lift_gain,
)

__all__ = [
    "apply_base_drag",
    "compute_base_drag_hoerner",
    "AirfoilShape",
    "AnalyticPolarBackend",
    "AeroPolarEvaluator",
    "Polar",
    "trailing_edge_drag_gain",
    "trailing_edge_lift_gain",
]
