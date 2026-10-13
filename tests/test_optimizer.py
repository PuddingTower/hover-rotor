"""Task 3.2 验收：多阶段混合寻优引擎。

基准案例 T=80 N、R=0.75 m、t_te=2.5 mm 的硬门禁（计划.md Task 3.2 + §5 + D-06/D-07）：
  * 含 DE 的完整流程 <= 3 min；解析快速路径 <= 30 s（性能基线目标值）；
  * 推力绝对误差 < 0.5%；
  * FM >= 0.68（硬门禁）；
  * M_tip <= 0.65 严格成立；
  * 10 截面尾缘厚度误差 < 0.001 mm；
  * 固定 seed 复跑结果一致。

FM 门禁由 `differential_evolution -> SLSQP -> 链式重启` 三段共同达成：
单次 SLSQP 停在 0.679（差 0.15%），重启线性化后才跨过 0.68（实测 0.6812）。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from hover_rotor.config import RotorInputs
from hover_rotor.geometry.cst import trailing_edge_gap
from hover_rotor.optimizer.objective import DESIGN_DIM, DesignBounds, HoverObjective
from hover_rotor.optimizer.solver import (
    OptimizationSettings,
    build_result,
    run_rotor_optimization,
)

BASELINE = dict(thrust=80.0, radius=0.75, te_thickness=0.0025)


def test_fast_path_is_deterministic_and_under_30s() -> None:
    """D-07 快速路径（关 DE、给好初值）<= 30 s。

    用 latency 配置（单轮 SLSQP、不链式重启）：链式重启是为跨过 FM 门禁加的，
    而快速路径的验收项是**耗时基线**，两者不该混在一个设置里互相拖累。
    """
    inputs = RotorInputs(**BASELINE)
    settings = OptimizationSettings(
        use_global=False, seed=5, polish_restarts=0, slsqp_maxiter=90, slsqp_ftol=1.0e-8
    )
    t0 = time.perf_counter()
    outcome = run_rotor_optimization(inputs, settings=settings)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"快速路径耗时 {elapsed:.1f}s 超过 30 s 基线"
    assert abs(outcome.diagnostics["thrust_residual"]) < 5e-3
    assert outcome.stages == ("slsqp",)


def test_seed_reproducibility_bitwise() -> None:
    """固定 seed 复跑必须逐位一致（用两轮 DE 验证随机源已被完全钉住）。"""
    inputs = RotorInputs(**BASELINE)
    settings = OptimizationSettings(
        seed=99, de_maxiter=2, de_popsize=3, polish_restarts=0, slsqp_maxiter=40, slsqp_ftol=1e-6
    )
    first = run_rotor_optimization(inputs, settings=settings)
    second = run_rotor_optimization(inputs, settings=settings)
    assert np.array_equal(first.x, second.x), "同种子得到不同设计向量"
    assert first.fun == pytest.approx(second.fun, rel=0, abs=0.0), "同种子目标值不一致"
    assert np.allclose(first.x, second.x)


def test_wrong_dimension_design_is_rejected() -> None:
    inputs = RotorInputs(**BASELINE)
    objective = HoverObjective(inputs)
    with pytest.raises(ValueError, match="维度"):
        objective.evaluate(np.zeros(DESIGN_DIM + 1))


def test_bounds_keep_tip_mach_hard_capped() -> None:
    """M_tip 作为设计变量并以箱式边界封顶，因此约束不可能被违反。"""
    bounds = DesignBounds.for_inputs(RotorInputs(**BASELINE))
    assert bounds.tip_mach_max <= 0.65
    assert bounds.as_scipy_bounds()[8] == (bounds.tip_mach_min, bounds.tip_mach_max)


@pytest.mark.slow
def test_baseline_case_meets_all_hard_gates() -> None:
    """§5 红线的端到端复验：耗时、推力吻合度、FM、压缩性、尾缘一致性。"""
    inputs = RotorInputs(**BASELINE)
    t0 = time.perf_counter()
    outcome = run_rotor_optimization(inputs, settings=OptimizationSettings(seed=2026))
    result = build_result(inputs, outcome)
    elapsed = time.perf_counter() - t0

    assert elapsed <= 180.0, f"含 DE 的完整流程耗时 {elapsed:.1f}s，超过 D-07 的 3 分钟硬门禁"
    assert result.thrust_error < 0.005, f"推力误差 {100 * result.thrust_error:.3f}% 超过 0.5%"
    assert result.fm >= 0.68, f"FM={result.fm:.4f} 未达 D-06 硬门禁 0.68"
    assert result.tip_mach <= 0.65 + 1e-12, f"M_tip={result.tip_mach:.4f} 违反压缩性约束"
    assert result.n_sections == 10
    assert result.ct > 0 and result.cp > 0 and result.power > result.power_ideal > 0

    gaps = [trailing_edge_gap(sec.points_normalized) * sec.chord for sec in result.sections]
    spread_mm = max(abs(g - inputs.te_thickness) for g in gaps) * 1000.0
    assert spread_mm < 1e-3, f"十截面尾缘厚度极差 {spread_mm:.2e} mm 超过 0.001 mm"

    assert result.solver_stage == "differential_evolution+slsqp"
    assert len(result.convergence_history) > 5, "收敛历史过短，报表层无法复盘搜索过程"
    assert result.seed == 2026


@pytest.mark.slow
def test_result_is_json_roundtrippable(baseline_inputs: RotorInputs) -> None:
    """Task 4.3 判定的前半：结果对象可无损往返（寻优侧）。"""
    import json

    outcome = run_rotor_optimization(
        baseline_inputs,
        settings=OptimizationSettings(
            use_global=False, seed=13, polish_restarts=0, slsqp_maxiter=60, slsqp_ftol=1e-7
        ),
    )
    result = build_result(baseline_inputs, outcome)
    restored = type(result).from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.fm == pytest.approx(result.fm, rel=0, abs=1e-12)
    assert len(restored.sections) == len(result.sections)
    assert np.allclose(restored.chord, result.chord)
    assert np.allclose(restored.sections[3].points_mm, result.sections[3].points_mm)
    assert restored.seed == result.seed


def test_build_result_fills_geometry_and_aero(baseline_inputs: RotorInputs) -> None:
    """装配层必须给出 CAD 需要的全部字段，否则 M4 只能拿到空点阵。"""
    outcome = run_rotor_optimization(
        baseline_inputs,
        settings=OptimizationSettings(
            use_global=False, seed=17, polish_restarts=0, slsqp_maxiter=60, slsqp_ftol=1e-7
        ),
    )
    result = build_result(baseline_inputs, outcome)
    for sec in result.sections:
        assert sec.points_mm.shape[0] >= 20
        assert sec.points_normalized.shape == sec.points_mm.shape
        assert sec.chord > 0 and sec.re > 0 and 0.0 <= sec.ma <= 0.75
        assert sec.te_thickness == pytest.approx(baseline_inputs.te_thickness)
        assert sec.dz_te == pytest.approx(baseline_inputs.te_thickness / sec.chord)
        assert sec.ld > 0
        assert len(sec.au) == len(sec.al) == 4
    assert len(result.polars) == 10
    assert set(result.polars[0]) >= {"alpha", "cl", "cd", "cm"}
    assert result.r_over_R[0] == pytest.approx(baseline_inputs.hub_ratio)
    assert result.r_over_R[-1] == pytest.approx(1.0)


def test_design_bounds_are_respected_by_optimum(baseline_inputs: RotorInputs) -> None:
    bounds = DesignBounds.for_inputs(baseline_inputs)
    outcome = run_rotor_optimization(
        baseline_inputs,
        settings=OptimizationSettings(
            use_global=False, seed=23, polish_restarts=0, slsqp_maxiter=60, slsqp_ftol=1e-7
        ),
    )
    x = outcome.x
    for lo, hi, value in zip(
        [b[0] for b in bounds.as_scipy_bounds()], [b[1] for b in bounds.as_scipy_bounds()], x, strict=True
    ):
        assert lo - 1e-9 <= value <= hi + 1e-9
    # 扭转单调性由几何罚项引导而非硬边界，故只检查它没有被推到边界外
    assert np.all(x[4:8] >= bounds.theta_min_deg - 1e-9) and np.all(x[4:8] <= bounds.theta_max_deg + 1e-9)
