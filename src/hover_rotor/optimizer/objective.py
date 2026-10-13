"""反设计目标函数与罚函数（计划.md Task 3.1）。

数学表述（§3）：

    min_X  J(X) = -FM(X) + w_T * ((T - T_target)/T_target)^2 + P_geom(X)
    s.t.   M_tip <= M_tip_max,  c_min <= c(r) <= c_max,  theta_min <= theta(r) <= theta_max,
           t_te(r) = t_te_input  (forall r)   <- 由几何层保证，不是自由变量

设计变量 X（25 维，顺序固定，编码即契约）：

    [0:4]   展向 4 控制节点弦长 c/R（PCHIP 重构全展分布）
    [4:8]   展向 4 控制节点安装角 theta [deg]
    [8]     叶尖马赫数 M_tip（转速由 M_tip*a_inf/R 反解，天然满足声学约束）
    [9:13]  叶根组上表面 CST 权重 A_u
    [13:17] 叶根组下表面 CST 权重 A_l
    [17:21] 叶尖组上表面 CST 权重 A_u
    [21:25] 叶尖组下表面 CST 权重 A_l

    中间 8 个站位的权重由「叶根 / 中值 / 叶尖」三组控制量经 PCHIP 展向光滑过渡得到，
    这是 §3 里「中间截面参数由展向光滑过渡生成」的直接实现，同时把变量数从
    10x2xn 压到 4xn，让差分进化在 D-07 的 3 分钟门禁内跑得完。

罚函数设计原则：**一切罚项都写成二次或 softplus 形式**，不用 if 分段返回大数——
差分进化无所谓，但第二阶段 SLSQP 依赖连续梯度，断点会让它停在假极值上。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hover_rotor.aero.bemt import BEMTConfig, HoverBEMTSolver, HoverSolution
from hover_rotor.aero.polar import AeroPolarEvaluator, AirfoilShape
from hover_rotor.config import RotorInputs
from hover_rotor.geometry.cst import generate_flatback_cst
from hover_rotor.geometry.discretization import SpanDistribution, build_span_distribution

FloatArray = NDArray[np.float64]

N_CHORD_CTRL = 4
N_TWIST_CTRL = 4
N_CST_WEIGHTS = 4
"""每组 CST 权重的个数（3 阶形函数）。与 §3 的「两组 CST 权重」共同决定设计空间维度。"""

DESIGN_DIM = N_CHORD_CTRL + N_TWIST_CTRL + 1 + 4 * N_CST_WEIGHTS

THRUST_WEIGHT = 100.0
"""§3 给定的推力二阶罚权重 w_T。"""

INTERSECTION_STIFFNESS = 4000.0
"""自相交罚的二次刚度。worst=-0.05 时给 ~12、-0.10 时给 ~44，
使穿模构型的 J 相对可行构型（约 -0.68）抬升两个数量级，
满足计划.md Task 3.1「至少抬升 10 倍」的判定准则。"""

TE_RATIO_LIMIT = 0.05
"""§Task 0.1 的 t_te/c_tip 工艺上限，越界以二次罚计入。"""

MAX_TWIST_GRADIENT = 12.0
"""相邻控制节点间允许的安装角变化 [deg/控制段]，超出即罚（防止锯齿扭转）。"""


@dataclass(frozen=True)
class DesignBounds:
    """设计变量箱式边界。缺省值覆盖微型穿越机到 eVTOL 量级，可由调用方收紧。"""

    chord_min_over_R: float = 0.045
    chord_max_over_R: float = 0.26
    theta_min_deg: float = 0.0
    theta_max_deg: float = 30.0
    tip_mach_min: float = 0.12
    tip_mach_max: float = 0.65
    au_min: float = 0.02
    au_max: float = 0.55
    al_min: float = -0.18
    al_max: float = 0.30

    def as_scipy_bounds(self) -> list[tuple[float, float]]:
        """按 DESIGN_DIM 顺序展开成 ``scipy.optimize`` 的 bounds 列表。"""
        bounds = [(self.chord_min_over_R, self.chord_max_over_R)] * N_CHORD_CTRL
        bounds += [(self.theta_min_deg, self.theta_max_deg)] * N_TWIST_CTRL
        bounds += [(self.tip_mach_min, self.tip_mach_max)]
        bounds += [(self.au_min, self.au_max)] * N_CST_WEIGHTS
        bounds += [(self.al_min, self.al_max)] * N_CST_WEIGHTS
        bounds += [(self.au_min, self.au_max)] * N_CST_WEIGHTS
        bounds += [(self.al_min, self.al_max)] * N_CST_WEIGHTS
        return bounds

    @classmethod
    def for_inputs(cls, inputs: RotorInputs) -> DesignBounds:
        """按输入工况收缩边界：叶尖弦长下限由 t_te/c_tip <= 5% 反推。"""
        c_tip_min = max(inputs.te_thickness / TE_RATIO_LIMIT, inputs.radius * 0.045)
        return DesignBounds(
            chord_min_over_R=min(c_tip_min / inputs.radius, 0.20),
            tip_mach_max=min(inputs.tip_mach_max, 0.65),
        )


@dataclass
class GeometryPayload:
    """解码后的几何：展向分布 + 十剖面 + 相对尾缘厚度。"""

    span: SpanDistribution
    shapes: list[AirfoilShape]
    dz_te: FloatArray
    r_ctrl: FloatArray
    violations: dict[str, float] = field(default_factory=dict)


def _softplus_sq(value: float, scale: float) -> float:
    """``scale*ln(1+exp(value/scale))`` 的平方，即光滑 max(0, value) 的二次式。"""
    z = max(min(value / scale, 40.0), -40.0)
    soft = scale * math.log1p(math.exp(z))
    return soft * soft


def control_span_positions(hub_ratio: float) -> FloatArray:
    """4 个展向控制节点的无量纲站位（含两端点，供 PCHIP 使用）。

    取 4 点而非等分是有意为之：把两个内节点压向 0.30/0.62 展向位置，使弦长与扭转
    在叶根承载过渡段与叶尖削缩段各有一个自由度，中段由单调样条自动光滑——这正是
    §3「4 点弦长 + 4 点扭转」的设计意图，等分节点会把自由度浪费在梯度很小的中段。
    """
    lo, hi = hub_ratio, 1.0
    return np.array([lo, lo + 0.30 * (hi - lo), lo + 0.62 * (hi - lo), hi])


def decode_design(x: Sequence[float] | FloatArray, inputs: RotorInputs) -> GeometryPayload:
    """设计向量 -> 展向几何 + 十剖面 CST 权重。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size != DESIGN_DIM:
        raise ValueError(f"设计向量维度应为 {DESIGN_DIM}，收到 {x.size}")

    hub = inputs.hub_ratio
    r_ctrl = control_span_positions(hub)
    chord_ctrl = x[0:N_CHORD_CTRL] * inputs.radius
    theta_ctrl = x[N_CHORD_CTRL : N_CHORD_CTRL + N_TWIST_CTRL]

    span = build_span_distribution(
        inputs.radius, r_ctrl, chord_ctrl, theta_ctrl, hub_ratio=hub, n_sections=inputs.n_sections
    )

    au_root = x[9 : 9 + N_CST_WEIGHTS]
    al_root = x[13 : 13 + N_CST_WEIGHTS]
    au_tip = x[17 : 17 + N_CST_WEIGHTS]
    al_tip = x[21 : 21 + N_CST_WEIGHTS]

    from scipy.interpolate import PchipInterpolator

    ctrl_r = np.array([hub, 0.5 * (hub + 1.0), 1.0])
    au_stations = np.column_stack(
        [
            PchipInterpolator(ctrl_r, np.array([g[0], g[1], g[2]]))(span.r_over_R)
            for g in _weight_cubes(au_root, au_tip)
        ]
    )
    al_stations = np.column_stack(
        [
            PchipInterpolator(ctrl_r, np.array([g[0], g[1], g[2]]))(span.r_over_R)
            for g in _weight_cubes(al_root, al_tip)
        ]
    )

    dz_te = inputs.te_thickness / span.chord
    shapes = [
        AirfoilShape.from_cst(au_stations[i], al_stations[i], float(dz_te[i])) for i in range(span.n_sections)
    ]
    return GeometryPayload(span=span, shapes=shapes, dz_te=dz_te, r_ctrl=r_ctrl)


def _weight_cubes(root: FloatArray, tip: FloatArray) -> list[FloatArray]:
    """把 (root, tip) 两组权重展开成逐 CST 分量的三控制点序列 [root, mid, tip]。"""
    mid = 0.5 * (root + tip)
    return [np.array([root[k], mid[k], tip[k]]) for k in range(root.size)]


def self_intersection_metric(au: FloatArray, al: FloatArray, dz_te: float, n_probe: int = 25) -> float:
    """剖面最小厚度（弦向采样），<= 0 即自相交或上下表面穿模。

    直接在 CST 解析式上探测而不生成完整点阵：厚度 zeta_u - zeta_l =
    C(psi)*(S_u - S_l) + psi*dz_te，类函数 C 在内部恒正，所以只需在加密网格上取最小值，
    代价是目标函数每次评估的 O(n_sections * n_probe)，比画 200 点闭合型线便宜一个量级。
    """
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_probe)))
    psi = psi[(psi > 1e-6) & (psi < 1.0 - 1e-6)]
    cls = np.power(psi, 0.5) * np.power(1.0 - psi, 1.0)
    from hover_rotor.geometry.cst import shape_function

    thickness = cls * (shape_function(psi, au) - shape_function(psi, al)) + psi * dz_te
    return float(np.min(thickness))


class HoverObjective:
    """悬停反设计的综合损失函数 J = -FM + w_T*推力残差^2 + P_geom。

    Args:
        inputs: 工程指标 T/R/t_te（含 B、轮毂比等）。
        bounds: 设计箱式边界。
        solver: BEMT 求解器；None 时用默认配置新建。
        config: BEMT 数值配置。
    """

    def __init__(
        self,
        inputs: RotorInputs,
        bounds: DesignBounds | None = None,
        solver: HoverBEMTSolver | None = None,
        config: BEMTConfig | None = None,
    ) -> None:
        self.inputs = inputs
        self.bounds = bounds or DesignBounds.for_inputs(inputs)
        self._evaluator = AeroPolarEvaluator("analytic")
        self.solver = solver or HoverBEMTSolver(self._evaluator, config)
        self._cache_key: bytes | None = None
        self._cache_value: tuple[float, dict[str, Any]] | None = None
        self.evaluations = 0

    @property
    def backend_name(self) -> str:
        return self._evaluator.backend_name

    @property
    def evaluator(self) -> AeroPolarEvaluator:
        """暴露极线评估器：报表层要按站位重算设计点气动量与极线表。"""
        return self._evaluator

    def omega_from(self, x: Sequence[float] | FloatArray) -> float:
        """设计变量第 9 位（叶尖马赫数）-> 转速 [rad/s]。"""
        return float(x[8]) * self.inputs.a_inf / self.inputs.radius

    def geometry(self, x: Sequence[float] | FloatArray) -> GeometryPayload:
        return decode_design(x, self.inputs)

    def evaluate(self, x: Sequence[float] | FloatArray) -> tuple[float, dict[str, Any]]:
        """返回 (J, diagnostics)。同一向量的 J 与诊断只算一次（``__call__`` 复用）。"""
        key = np.asarray(x, dtype=np.float64).tobytes()
        if self._cache_value is not None and self._cache_key == key:
            return self._cache_value

        x = np.asarray(x, dtype=np.float64)
        payload = self.geometry(x)
        omega = self.omega_from(x)
        solution = self.solver.solve(
            omega,
            payload.span,
            payload.shapes,
            radius=self.inputs.radius,
            blade_count=self.inputs.blade_count,
            hub_ratio=self.inputs.hub_ratio,
        )
        j, diag = self._score(x, payload, solution, omega)
        self._cache_key, self._cache_value = key, (j, diag)
        return j, diag

    def __call__(self, x: Sequence[float] | FloatArray) -> float:
        self.evaluations += 1
        return float(self.evaluate(x)[0])

    def _score(
        self, x: FloatArray, payload: GeometryPayload, solution: HoverSolution, omega: float
    ) -> tuple[float, dict[str, Any]]:
        inputs = self.inputs
        thrust_residual = (solution.thrust - inputs.thrust) / inputs.thrust
        j_aero = -solution.fm
        p_thrust = THRUST_WEIGHT * thrust_residual * thrust_residual
        p_geom, geom_parts = self._geometry_penalty(x, payload)
        p_mach = self._tip_mach_penalty(x)

        j = j_aero + p_thrust + p_geom + p_mach
        diagnostics = {
            "thrust": solution.thrust,
            "power": solution.power,
            "torque": solution.torque,
            "fm": solution.fm,
            "ct": solution.ct,
            "cp": solution.cp,
            "thrust_residual": thrust_residual,
            "penalty_thrust": p_thrust,
            "penalty_geometry": p_geom,
            "penalty_tip_mach": p_mach,
            "objective": j,
            "omega": omega,
            "tip_mach": solution.tip_mach,
            "converged": solution.converged,
            "iterations": solution.iterations,
            **geom_parts,
        }
        return j, diagnostics

    def _tip_mach_penalty(self, x: FloatArray) -> float:
        m = float(x[8])
        lo, hi = self.bounds.tip_mach_min, min(self.bounds.tip_mach_max, self.inputs.tip_mach_max)
        if m > hi:
            return 200.0 * (m - hi) ** 2
        if m < lo:
            return 200.0 * (lo - m) ** 2
        return 0.0

    def _geometry_penalty(self, x: FloatArray, payload: GeometryPayload) -> tuple[float, dict[str, float]]:
        """几何罚：弦长越界、扭转梯度超限、剖面自相交、叶尖尾缘厚度比越界。"""
        parts: dict[str, float] = {}

        chord_over_R = payload.span.chord / self.inputs.radius
        below = np.maximum(self.bounds.chord_min_over_R - chord_over_R, 0.0)
        above = np.maximum(chord_over_R - self.bounds.chord_max_over_R, 0.0)
        parts["penalty_chord"] = float(400.0 * np.sum(below**2 + above**2))

        theta = payload.span.theta_deg
        gradient = np.diff(theta)
        too_steep = np.maximum(gradient + 1e-9, 0.0)
        parts["penalty_twist"] = float(2.0 * np.sum((too_steep / max(MAX_TWIST_GRADIENT, 1e-9)) ** 2))
        out_of_range = np.maximum(self.bounds.theta_min_deg - theta, 0.0) + np.maximum(
            theta - self.bounds.theta_max_deg, 0.0
        )
        parts["penalty_twist"] += float(1.0 * np.sum(out_of_range**2))

        worst = min(self._thickness_from_shape(shape) for shape in payload.shapes)
        # 只对「真的穿模」计价：计划把 P_geom 限定为自相交/弦长越界/扭转梯度三项，
        # 早先额外加的"最小厚度 >= 0.025"软门槛会让可行构型也带 ~2e-3 的常驻罚，
        # 从而永远达不到 Task 3.1 判据要求的 |J-(-FM)| < 0.001。
        # softplus 形式保证在 worst=0 处连续可微（SLSQP 穿越可行边界时不被断点弹回）。
        parts["penalty_intersection"] = float(4000.0 * _softplus_sq(-worst, 0.005))

        dz_tip = float(payload.dz_te[-1])
        parts["penalty_te_ratio"] = float(150.0 * max(dz_tip - TE_RATIO_LIMIT, 0.0) ** 2)

        return float(sum(parts.values())), parts

    @staticmethod
    def _thickness_from_shape(shape: AirfoilShape) -> float:
        return self_intersection_metric(
            np.asarray(shape.au, dtype=np.float64), np.asarray(shape.al, dtype=np.float64), shape.dz_te
        )

    def section_contours(self, payload: GeometryPayload, n_points: int = 200) -> list[FloatArray]:
        """十截面的归一化闭合型线（制造层用），弦向坐标 psi、厚度坐标 zeta。"""
        contours = []
        for shape in payload.shapes:
            contours.append(
                generate_flatback_cst(
                    np.asarray(shape.au, dtype=np.float64),
                    np.asarray(shape.al, dtype=np.float64),
                    shape.dz_te,
                    n_points=n_points,
                )
            )
        return contours
