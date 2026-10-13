"""Task 0.1 验收：工程指标输入空间的边界校验。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hover_rotor.config import RotorInputs


def test_baseline_case_accepts() -> None:
    inputs = RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.0025)
    assert inputs.blade_count == 2
    assert inputs.hub_ratio == pytest.approx(0.15)
    assert inputs.tip_mach_max == pytest.approx(0.65)
    assert inputs.n_sections == 10


def test_derived_disk_area_and_speed_limit() -> None:
    inputs = RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.0025)
    assert inputs.disk_area == pytest.approx(3.14159 * 0.75**2, rel=1e-3)
    # omega_max = M_tip * a / R
    assert inputs.omega_max == pytest.approx(0.65 * 340.3 / 0.75, rel=1e-6)


@pytest.mark.parametrize(
    "kwargs, hint",
    [
        ({"thrust": -1.0, "radius": 0.75, "te_thickness": 0.0025}, "推力"),
        ({"thrust": 80.0, "radius": -0.2, "te_thickness": 0.0025}, "半径"),
        ({"thrust": 80.0, "radius": 0.75, "te_thickness": 0.0}, "钝尾缘厚度"),
        ({"thrust": 80.0, "radius": 0.75, "te_thickness": 0.008}, "钝尾缘厚度"),
        ({"thrust": 50000.0, "radius": 0.75, "te_thickness": 0.0025}, "推力"),
    ],
)
def test_illegal_inputs_raise_chinese_hint(kwargs: dict, hint: str) -> None:
    """负数/越界物理指标必须被拦截，且提示可读。"""
    with pytest.raises(ValidationError) as excinfo:
        RotorInputs(**kwargs)
    assert hint in str(excinfo.value)


def test_sharp_trailing_edge_rejected() -> None:
    """尖尾缘在工艺上不可接受：0.3 mm 下限即为其拦截点。"""
    with pytest.raises(ValidationError, match="尖尾缘"):
        RotorInputs(thrust=80.0, radius=0.75, te_thickness=1e-5)


def test_te_over_tip_chord_ratio_rejected() -> None:
    """t_te/c_tip = 0.06 超过 5% 工艺上限。"""
    with pytest.raises(ValidationError, match="工艺上限"):
        RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.003, chord_tip=0.05)


def test_te_within_tip_chord_ratio_accepted() -> None:
    inputs = RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.0025, chord_tip=0.06)
    assert inputs.chord_tip == pytest.approx(0.06)


def test_unknown_field_is_forbidden() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.0025, chord=0.1)


@pytest.mark.parametrize("hub", [0.0, 0.02, 0.35])
def test_hub_ratio_out_of_range(hub: float) -> None:
    with pytest.raises(ValidationError):
        RotorInputs(thrust=80.0, radius=0.75, te_thickness=0.0025, hub_ratio=hub)


def test_optional_dependency_error_is_runtime_error() -> None:
    """CAD/NN 后端缺席应抛 OptionalDependencyError，供调用方降级而非中止。"""
    from hover_rotor.config import OptionalDependencyError

    assert issubclass(OptionalDependencyError, RuntimeError)
