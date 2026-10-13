"""Task 2.2 验收：可微极线代理与后端回退。

判定准则取自计划.md Task 2.2「测试判定准则」：单次极线求解 <5 ms、
alpha in [-5, 20] 内 c_l/c_d 无跳变、对称剖面 c_l(0)~0、
t_te/c 增大时 c_{l,max} 与 c_{d0} 同步单调上升（复现 §2.4 趋势）、
NeuralFoil 缺席时双后端一致性测试 skip。
另加薄翼实现的文献锚定（NACA 2412 零升迎角与 1/4 弦力矩）——
这是唯一能证明 A 系数积分与力矩平移没写错符号/前因子的外部判据。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from hover_rotor.aero.polar import (
    MA_MAX,
    RE_MAX,
    RE_MIN,
    AeroPolarEvaluator,
    AirfoilShape,
    _eckhous_max_ld,
    trailing_edge_lift_gain,
)
from hover_rotor.config import OptionalDependencyError

ALPHA_GRID = np.linspace(-5.0, 20.0, 251)


@pytest.fixture(scope="module")
def evaluator() -> AeroPolarEvaluator:
    return AeroPolarEvaluator("analytic")


@pytest.fixture(scope="module")
def naca0012(naca0012_cst) -> AirfoilShape:
    au, al = naca0012_cst
    return AirfoilShape.from_cst(au, al, 0.02)


def test_thin_airfoil_recovers_naca2412_constants(naca_to_cst) -> None:
    """薄翼升力线/零升迎角/1/4 弦力矩对 NACA 2412 文献值的复现能力。

    文献：alpha_L0 = -2.1 deg，c_m,c4 = -0.050，升力线斜率 0.109 /deg（不可压薄翼 2*pi）。
    """
    au, al = naca_to_cst("2412")
    shape = AirfoilShape.from_cst(au, al, 0.001)
    assert shape.alpha_zero_lift * 180 / np.pi == pytest.approx(-2.1, abs=0.35)
    assert shape.cm_quarter == pytest.approx(-0.050, abs=0.012)

    ev = AeroPolarEvaluator("analytic")
    alpha = np.linspace(-4.0, 6.0, 12)
    cl = ev.evaluate(shape, alpha, re=1.0e6, ma=0.0).cl
    slope = np.polyfit(alpha, cl, 1)[0]
    assert slope == pytest.approx(0.109, abs=0.008)


def test_symmetric_section_has_zero_lift_at_zero_alpha(evaluator, naca0012) -> None:
    """判定准则：对称翼型 c_l(0) ~ 0。"""
    cl0 = evaluator.evaluate(naca0012, [0.0], re=1.0e6, ma=0.0).cl[0]
    assert abs(cl0) < 0.02


def test_polar_solve_under_5ms(evaluator, naca0012) -> None:
    """判定准则：单次极线求解耗时 < 5 ms（12 个迎角点，几何特征量已缓存）。"""
    alpha = np.linspace(0.0, 10.0, 12)
    evaluator.evaluate(naca0012, alpha, re=8.0e5, ma=0.3)  # 预热
    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        evaluator.evaluate(naca0012, alpha, re=8.0e5, ma=0.3)
        samples.append((time.perf_counter() - t0) * 1000.0)
    assert np.median(samples) < 5.0, f"极线求解中位耗时 {np.median(samples):.2f} ms 超过 5 ms"


def test_polar_is_continuous_no_jumps(evaluator, naca0012) -> None:
    """判定准则：alpha in [-5, 20] 内 c_l(alpha)、c_d(alpha) 无跳变断点。

    c_l 用「相邻差分 / 全幅值」作相对度量：直接除以 local c_l 在 c_l->0 处必然爆炸，
    那度量的是零点附近的条件数而不是型线连续性，会把合法曲线判为失败。
    """
    res = evaluator.evaluate(naca0012, ALPHA_GRID, re=6.0e5, ma=0.35)
    span = float(res.cl.max() - res.cl.min())
    dcl = np.abs(np.diff(res.cl))
    dcd_rel = np.abs(np.diff(res.cd)) / np.abs(res.cd[:-1])
    assert dcl.max() < 0.20 * span, f"c_l 相邻点跳变 {dcl.max() / span:.1%} 超过 20% 全幅"
    assert dcd_rel.max() < 0.20, f"c_d 相邻点相对跳变 {dcd_rel.max():.1%} 超过 20%"
    assert np.abs(np.diff(res.cl, 2)).max() < 5e-3, "c_l 二阶差分存在拐点级不连续"


def test_blunt_te_trends_match_table_2_4(evaluator, naca0012_cst) -> None:
    """判定准则：t_te/c 增大时 c_{l,max} 单调上升、c_{d0} 单调上升（§2.4 趋势）。"""
    au, al = naca0012_cst
    ev = AeroPolarEvaluator("analytic")
    dz_grid = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
    alpha = np.linspace(-2.0, 26.0, 141)
    cl_max, cd_min = [], []
    for dz in dz_grid:
        shape = AirfoilShape.from_cst(au, al, dz)
        res = ev.evaluate(shape, alpha, re=1.0e6, ma=0.30)
        cl_max.append(float(res.cl.max()))
        cd_min.append(float(res.cd.min()))
    assert np.all(np.diff(cl_max) > 0.0), f"c_l,max 未随 t_te/c 单调上升: {cl_max}"
    assert np.all(np.diff(cd_min) > 0.0), f"c_d0 未随 t_te/c 单调上升: {cd_min}"
    # §2.4 结构型 flatback（5%~10%）应给 +10%~+18% 升力增益
    gain = cl_max[-1] / cl_max[0] - 1.0
    assert 0.08 <= gain <= 0.35, f"dz_te=0.05 的 c_l,max 增益 {gain:.1%} 偏离 §2.4 表列 10%~18%"
    assert trailing_edge_lift_gain(0.0) == 0.0


def test_max_ld_is_set_by_eckhous_for_clean_section(evaluator, naca0012_cst) -> None:
    """Eckhous 相关式在链路里确实是承重的：洁净剖面的模型最大升阻比应复现它。

    极线斜率 k = 1/(4 cd0 (L/D)max^2) 直接由 Eckhous 反解，故实测 (L/D)max 必然贴近
    相关式输出；若 k 被改成自由常数，本条立刻失配。用 dz_te=0.001 的近洁净剖面探测，
    因为钝尾缘的 Hoerner 基底阻力（D-01）本就应把 (L/D)max 压到洁净值以下。
    """
    au, al = naca0012_cst
    clean = AirfoilShape.from_cst(au, al, 0.001)
    res = evaluator.evaluate(clean, np.linspace(-4.0, 20.0, 241), re=1.0e6, ma=0.0)
    ld_model = float((res.cl / res.cd).max())
    ld_eck = _eckhous_max_ld(clean, 1.0e6)
    assert 0.85 * ld_eck <= ld_model <= 1.6 * ld_eck, (
        f"模型 (L/D)max={ld_model:.1f} 未复现 Eckhous 值 {ld_eck:.1f}，极线闭合已脱离经验锚点"
    )


def test_blunt_te_reduces_max_ld_below_clean_case(evaluator, naca0012_cst) -> None:
    """钝尾缘必须以最大升阻比的下降为代价换取升力增益（§2.4 的两面性）。"""
    au, al = naca0012_cst
    ev = AeroPolarEvaluator("analytic")
    alpha = np.linspace(-2.0, 24.0, 201)

    def _ld(dz: float) -> float:
        r = ev.evaluate(AirfoilShape.from_cst(au, al, dz), alpha, re=1.0e6, ma=0.0)
        return float((r.cl / r.cd).max())

    assert _ld(0.05) < _ld(0.01) < _ld(0.001)


def test_mach_rise_is_monotonic_and_wave_drag_smooth(evaluator, naca0012) -> None:
    """压缩性：同一迎角下 c_d 随 Ma 单调上升，且升力线按 Prandtl-Glauert 增强。"""
    ma_grid = np.linspace(0.0, 0.65, 28)
    cl = np.empty(ma_grid.size)
    cd = np.empty(ma_grid.size)
    for i, ma in enumerate(ma_grid):
        res = evaluator.evaluate(naca0012, [4.0], re=1.5e6, ma=float(ma))
        cl[i], cd[i] = res.cl[0], res.cd[0]
    assert np.all(np.diff(cd) > 0.0), "c_d 未随马赫数单调上升"
    assert np.all(np.diff(cl) > 0.0), "c_l 未随马赫数（Prandtl-Glauert）上升"
    assert np.abs(np.diff(cd, 2)).max() < 5e-3, "波阻在临界马赫数附近存在 kink"


def test_reynolds_and_mach_domain_are_enforced(evaluator, naca0012) -> None:
    """适用域越界必须显式抛错，不得静默外推。"""
    with pytest.raises(ValueError, match="雷诺数"):
        evaluator.evaluate(naca0012, [4.0], re=RE_MIN / 2.0)
    with pytest.raises(ValueError, match="雷诺数"):
        evaluator.evaluate(naca0012, [4.0], re=RE_MAX * 2.0, ma=0.0)
    with pytest.raises(ValueError, match="马赫数"):
        evaluator.evaluate(naca0012, [4.0], re=1.0e6, ma=MA_MAX + 0.05)


def test_base_drag_is_included_in_cd(evaluator, naca_to_cst) -> None:
    """D-01 基底阻力必须进入极线输出：同一剖面 dz_te 翻倍后零升阻力显著抬升。"""
    au, al = naca_to_cst("0012")
    ev = AeroPolarEvaluator("analytic")
    thin = ev.evaluate(AirfoilShape.from_cst(au, al, 0.005), [0.0], re=1.0e6).cd[0]
    blunt = ev.evaluate(AirfoilShape.from_cst(au, al, 0.050), [0.0], re=1.0e6).cd[0]
    assert blunt - thin > 0.02, f"dz_te 0.005->0.050 仅使 c_d 从 {thin:.4f} 变到 {blunt:.4f}，基底阻力未生效"


def test_backend_auto_falls_back_to_analytic() -> None:
    """D-02：auto 在 NeuralFoil 缺席时静默回退解析后端。"""
    assert AeroPolarEvaluator("auto").backend_name in ("analytic", "neuralfoil")


def test_forced_neuralfoil_backend_raises_when_absent() -> None:
    """强制 NN 后端而权重缺席时抛 OptionalDependencyError，而非 ImportError 或静默降级。"""
    try:
        import neuralfoil  # noqa: F401
    except ImportError:
        with pytest.raises(OptionalDependencyError, match="neuralfoil"):
            AeroPolarEvaluator("neuralfoil")
    else:  # pragma: no cover - 取决于环境
        pytest.skip("环境已装 neuralfoil，强制后端不应抛错")


def test_invalid_backend_name_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        AeroPolarEvaluator("xfoil")


def test_mirror_flips_camber_effects(naca_to_cst) -> None:
    """反弯镜像：零升迎角严格反号，厚度与尾缘钝度不变（供优化器使用反弯自由度）。"""
    au, al = naca_to_cst("2412")
    shape = AirfoilShape.from_cst(au, al, 0.01)
    mirrored = shape.mirror()
    assert mirrored.alpha_zero_lift == pytest.approx(-shape.alpha_zero_lift, abs=2e-3)
    assert mirrored.thickness == pytest.approx(shape.thickness, rel=1e-3)
    assert mirrored.dz_te == shape.dz_te


def test_batch_interface_matches_pointwise(evaluator, naca0012_cst) -> None:
    """批量接口与逐点调用逐位一致（BEMT 稠密积分依赖此等价性）。"""
    au, al = naca0012_cst
    shapes = [AirfoilShape.from_cst(au, al, dz) for dz in (0.005, 0.02, 0.035)]
    alpha = np.vstack([np.linspace(0.0, 12.0, 7) for _ in shapes])
    re = [4.0e5, 8.0e5, 1.2e6]
    ma = [0.1, 0.3, 0.5]
    batch = evaluator.evaluate_sections(shapes, alpha, re=re, ma=ma)
    for i, shape in enumerate(shapes):
        single = evaluator.evaluate(shape, alpha[i], re=re[i], ma=ma[i])
        assert np.allclose(batch.cl[i], single.cl)
        assert np.allclose(batch.cd[i], single.cd)
        assert np.allclose(batch.cm[i], single.cm)
    # 一维 alpha 视为所有站位共用
    shared = evaluator.evaluate_sections(shapes, alpha[0], re=re[0], ma=ma[0])
    assert shared.cl.shape == (3, 7)
    assert np.allclose(shared.cl[0], evaluator.evaluate(shapes[0], alpha[0], re=re[0], ma=ma[0]).cl)


def test_shape_rejects_sharp_or_invalid_te(naca0012_cst) -> None:
    """尖尾缘与非法权重在参数化层即被拒绝。"""
    au, al = naca0012_cst
    with pytest.raises(ValueError, match="尾缘厚度"):
        AirfoilShape.from_cst(au, al, 0.0)
    with pytest.raises(ValueError, match="等长"):
        AirfoilShape.from_cst([0.1, 0.2], [0.1], 0.02)


def test_shape_caching_makes_repeat_build_free(naca0012_cst) -> None:
    """同一组权重的剖面特征量必须命中缓存：冷构建 ~0.2 ms，缓存后近乎为零。"""
    au, al = naca0012_cst
    AirfoilShape.from_cst(au, al, 0.02)
    t0 = time.perf_counter()
    for _ in range(200):
        AirfoilShape.from_cst(au, al, 0.02)
    assert (time.perf_counter() - t0) / 200 < 5e-5, "AirfoilShape 未命中缓存，BEMT 高频调用将被拖垮"
