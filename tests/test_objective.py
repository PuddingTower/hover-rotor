"""Task 3.1 验收：优化目标函数与罚函数设计。

判定准则取自计划.md Task 3.1：
  * T 严格逼近 T_target 时惩罚项降至 1e-6 以下，J 与 -FM 绝对误差 < 0.001；
  * 自相交构型使 J 至少抬升 10 倍；
  * J 对每个设计变量的中心差分梯度连续存在。
"""

from __future__ import annotations

import numpy as np
import pytest

from hover_rotor.config import RotorInputs
from hover_rotor.optimizer.objective import (
    DESIGN_DIM,
    INTERSECTION_STIFFNESS,
    THRUST_WEIGHT,
    DesignBounds,
    HoverObjective,
    control_span_positions,
    decode_design,
    self_intersection_metric,
)

X_REF = np.concatenate(
    [
        [0.150, 0.128, 0.100, 0.078],  # c/R
        [21.0, 15.0, 9.0, 3.5],  # theta deg
        [0.32],  # M_tip
        [0.30, 0.28, 0.11, 0.03],
        [-0.05, -0.02, 0.02, 0.06],
        [0.27, 0.23, 0.09, 0.02],
        [-0.04, 0.00, 0.04, 0.08],
    ]
)


@pytest.fixture(scope="module")
def objective(baseline_inputs: RotorInputs) -> HoverObjective:
    return HoverObjective(baseline_inputs)


@pytest.fixture(scope="module")
def polished():
    """局部精调一次，供「罚项消退」与「自相交抬升」两条判据复用（快速路径约 10~20 s）。

    自建 RotorInputs 而非复用 baseline_inputs：后者是 function 作用域，
    module 作用域 fixture 依赖它会触发 pytest ScopeMismatch。
    """
    from hover_rotor.optimizer.solver import OptimizationSettings, run_rotor_optimization

    return run_rotor_optimization(
        RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.0025),
        settings=OptimizationSettings(use_global=False, seed=11, slsqp_maxiter=80),
    )


def test_design_vector_layout_is_the_contract(objective) -> None:
    """编码即契约：25 维、第 9 位是马赫数、控制节点含展向两端。"""
    assert DESIGN_DIM == 25
    assert X_REF.size == DESIGN_DIM
    assert objective.omega_from(X_REF) == pytest.approx(0.32 * baseline_mach_speed())
    r_ctrl = control_span_positions(objective.inputs.hub_ratio)
    assert r_ctrl[0] == pytest.approx(objective.inputs.hub_ratio)
    assert r_ctrl[-1] == pytest.approx(1.0)
    assert np.all(np.diff(r_ctrl) > 0.0)


def baseline_mach_speed() -> float:
    """a_inf / R：把 M_tip 反解成转速的换算因子基准（R=0.75 m）。"""
    from hover_rotor.config import A_INF

    return A_INF / 0.75


def test_decode_produces_ten_sections_with_constant_te(objective) -> None:
    """t_te 全展恒定不是自由变量：解码后每个站位的绝对尾缘厚度必须等于输入值。"""
    payload = decode_design(X_REF, objective.inputs)
    assert payload.span.n_sections == 10
    te = payload.dz_te * payload.span.chord
    assert np.max(np.abs(te - objective.inputs.te_thickness)) < 1e-15
    assert np.all(payload.dz_te > 0.0)
    assert np.all(payload.span.chord > 0.0)


def test_wrong_dimension_rejected(objective) -> None:
    with pytest.raises(ValueError, match="维度"):
        decode_design(np.zeros(DESIGN_DIM - 3), objective.inputs)


def test_penalty_weight_matches_spec_and_j_decomposition(objective) -> None:
    """J = -FM + 100*residual^2 + P_geom，逐项可核。"""
    j, d = objective.evaluate(X_REF)
    assert THRUST_WEIGHT == 100.0
    assert d["penalty_thrust"] == pytest.approx(THRUST_WEIGHT * d["thrust_residual"] ** 2, rel=1e-12)
    assert j == pytest.approx(-d["fm"] + d["penalty_thrust"] + d["penalty_geometry"] + d["penalty_tip_mach"])
    assert d["converged"] is True


def test_thrust_penalty_vanishes_at_target(objective, polished) -> None:
    """判定准则：推力达标时罚项 < 1e-6 且 |J - (-FM)| < 0.001。

    用局部精调（关 DE）把推力压到目标上——这也是 Task 3.2 快速路径的同一条代码，
    顺带验证「只给好初值就能达标」的设计意图。
    """
    # 沿扭转方向做一维割线微调，把推力精确压到目标上——
    # Task 3.1 这条判据问的是「罚函数在 T=T_target 处是否消退」，
    # 因此必须先把 T 真正打到 T_target，而不是拿优化器的收敛精度当判据。
    from scipy.optimize import brentq

    def residual(delta: float) -> float:
        x_trim = polished.x.copy()
        x_trim[4:8] = np.clip(x_trim[4:8] + delta, 0.0, 30.0)
        return objective.evaluate(x_trim)[1]["thrust"] - objective.inputs.thrust

    delta = brentq(residual, -4.0, 4.0, xtol=1e-10, rtol=1e-12)
    x_trim = polished.x.copy()
    x_trim[4:8] = np.clip(x_trim[4:8] + delta, 0.0, 30.0)
    j, d = objective.evaluate(x_trim)

    assert abs(d["thrust_residual"]) < 1.0e-4, f"微调后推力残差 {d['thrust_residual']:.2e}"
    assert d["penalty_thrust"] < 1e-6, f"推力罚 {d['penalty_thrust']:.2e} 未降到 1e-6 以下"
    assert abs(j + d["fm"]) < 0.001, f"|J-(-FM)|={abs(j + d['fm']):.5f}，罚项未按要求消退"
    assert d["penalty_geometry"] < 1e-3, f"精调后仍残留几何罚 {d['penalty_geometry']:.4f}"


def test_self_intersecting_geometry_raises_j_by_at_least_ten_times(objective, polished) -> None:
    """判定准则：自相交构型的 J 至少抬升 10 倍。"""
    _, d_ok = objective.evaluate(X_REF)
    j_feasible = polished.fun
    assert d_ok["penalty_intersection"] < 1.0

    x_bad = X_REF.copy()
    x_bad[13:17] = [0.75, 0.75, 0.75, 0.75]  # 下表面权重压倒上表面 -> 穿模
    payload = decode_design(x_bad, objective.inputs)
    worst = min(self_intersection_metric(np.asarray(s.au), np.asarray(s.al), s.dz_te) for s in payload.shapes)
    assert worst < 0.0, "构造的穿模构型未被几何指标识别"

    j_bad, d_bad = objective.evaluate(x_bad)
    assert d_bad["penalty_intersection"] > 1.0, "穿模未产生可见罚项"
    assert INTERSECTION_STIFFNESS > 0.0
    assert abs(j_bad) >= 10.0 * abs(j_feasible), f"J 只从 {j_feasible:.3f} 抬到 {j_bad:.3f}，不足 10 倍"


def test_central_difference_gradient_exists_for_every_variable(objective) -> None:
    """判定准则：J 对每个设计变量的中心差分梯度连续存在（SLSQP 阶段的前提）。"""
    steps = [1e-5, 1e-5, 1e-5, 1e-5, 2e-4, 2e-4, 2e-4, 2e-4, 1e-4] + [2e-4] * 16
    grads = np.empty(DESIGN_DIM)
    for i in range(DESIGN_DIM):
        xp, xm = X_REF.copy(), X_REF.copy()
        xp[i] += steps[i]
        xm[i] -= steps[i]
        jp = objective.evaluate(xp)[0]
        jm = objective.evaluate(xm)[0]
        assert np.isfinite(jp) and np.isfinite(jm), f"变量 {i} 的邻域评估出现非有限值"
        grads[i] = (jp - jm) / (2.0 * steps[i])
    assert np.all(np.isfinite(grads)), "存在缺失梯度的设计变量"
    assert np.any(np.abs(grads) > 1e-4), "所有梯度都为零，目标函数与设计变量脱钩"


def test_gradient_is_continuous_under_perturbation(objective) -> None:
    """同一位置重复求梯度必须逐位一致（确定性），且沿变量方向平滑移动时梯度不大跳。"""

    def central(i: int, x: np.ndarray, h: float = 1e-4) -> float:
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        return (objective.evaluate(xp)[0] - objective.evaluate(xm)[0]) / (2.0 * h)

    g1 = central(8, X_REF)
    g2 = central(8, X_REF)
    assert g1 == pytest.approx(g2, rel=1e-9, abs=1e-9), "目标函数非确定性，SLSQP 无法收敛"

    shifted = X_REF.copy()
    shifted[8] += 0.01
    g3 = central(8, shifted)
    assert abs(g3 - g1) < 5.0 * abs(g1) + 5.0, "M_tip 方向梯度不连续（存在硬分段）"


def test_tip_mach_violation_is_penalized(objective) -> None:
    """M_tip 越界必须付代价：越界构型的 J 不得低于同几何的合规构型。"""
    x = X_REF.copy()
    x[8] = objective.bounds.tip_mach_max + 0.2
    j_bad, d_bad = objective.evaluate(x)
    x_ok = X_REF.copy()
    x_ok[8] = objective.bounds.tip_mach_max
    j_ok, _ = objective.evaluate(x_ok)
    assert d_bad["penalty_tip_mach"] > 1.0
    assert j_bad > j_ok


def test_bounds_expand_to_design_dimension() -> None:
    bounds = DesignBounds()
    as_list = bounds.as_scipy_bounds()
    assert len(as_list) == DESIGN_DIM
    assert all(lo < hi for lo, hi in as_list)
    assert as_list[8] == (bounds.tip_mach_min, bounds.tip_mach_max)


def test_geometry_penalty_components_are_reported(objective) -> None:
    _, d = objective.evaluate(X_REF)
    for key in ("penalty_chord", "penalty_twist", "penalty_intersection", "penalty_te_ratio"):
        assert key in d, f"诊断字典缺少 {key}，报表层无法分解罚项"
        assert d[key] >= 0.0


def test_te_ratio_limit_enforced_on_tip_chord(baseline_inputs) -> None:
    """t_te/c_tip 越出工艺上限时必须抬升罚项（Task 0.1 校验的优化层对应物）。"""
    obj = HoverObjective(baseline_inputs)
    x = X_REF.copy()
    x[0:4] = 0.030  # 极小弦长 -> dz_te 远超 5%
    _, d = obj.evaluate(x)
    assert d["penalty_te_ratio"] > 0.5
    assert d["penalty_chord"] > 0.0
