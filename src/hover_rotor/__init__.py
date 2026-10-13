"""hover_rotor —— 端到端悬停旋翼气动优化与钝尾缘截面制造系统。

三个工程指标进（目标推力 T、旋翼半径 R、钝尾缘绝对厚度 t_te），
十个可上架数控/线切割的钝尾缘截面出，中间由「拓展 CST 几何 -> 可微极线代理 ->
悬停 BEMT -> DE+SLSQP 两阶段反设计 -> DXF/STEP/STL/报表」一条可微链路打通。

本模块只暴露稳定 API；子模块内部符号不进 __all__。
"""

from __future__ import annotations

from hover_rotor.aero.base_drag import apply_base_drag, compute_base_drag_hoerner
from hover_rotor.aero.bemt import BEMTConfig, ConstantPolarTable, HoverBEMTSolver
from hover_rotor.aero.polar import AeroPolarEvaluator, AirfoilShape, Polar
from hover_rotor.config import (
    FeasibilityError,
    OptionalDependencyError,
    RotorInputs,
    RotorResult,
    SectionResult,
)
from hover_rotor.geometry.cst import generate_flatback_cst
from hover_rotor.geometry.discretization import build_span_distribution, cosine_span_positions
from hover_rotor.optimizer.objective import DesignBounds, HoverObjective
from hover_rotor.optimizer.solver import OptimizationSettings, build_result, run_rotor_optimization

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 输入与结果契约
    "RotorInputs",
    "RotorResult",
    "SectionResult",
    "OptionalDependencyError",
    "FeasibilityError",
    # 几何
    "generate_flatback_cst",
    "cosine_span_positions",
    "build_span_distribution",
    # 气动
    "compute_base_drag_hoerner",
    "apply_base_drag",
    "AirfoilShape",
    "AeroPolarEvaluator",
    "Polar",
    "HoverBEMTSolver",
    "BEMTConfig",
    "ConstantPolarTable",
    # 反设计
    "DesignBounds",
    "HoverObjective",
    "OptimizationSettings",
    "run_rotor_optimization",
    "build_result",
]
