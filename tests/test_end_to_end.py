"""Task 5.3 / 6.2 验收：单命令端到端与三套算例。

判定准则（Task 5.3）：
  * CliRunner 正常调用返回 exit_code == 0；
  * `--thrust -1`、`--te-thickness 0` 返回非零码并给出清晰的中/英文越界提示；
  * 输出目录文件清单完整：DXF×10、DAT×10、CSV×10、STEP、STL、manifest、JSON、PNG、HTML。

Task 6.2 的三套算例（微型穿越机 / 工业级重载 / eVTOL 载人级）在 slow 作业里
用快速路径各跑一遍，验证同一套代码从 5 N 外推到 4500 N 不需切换任何模型常数。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from hover_rotor.cli import main
from hover_rotor.config import RotorInputs

EXPECTED_SUFFIX_COUNTS = {
    ".dxf": 10,
    ".dat": 10,
    ".csv": 11,
    ".step": 1,
    ".stl": 1,
    ".png": 1,
    ".html": 1,
    ".json": 1,
    ".md": 1,
}


def _run(args: list[str]) -> tuple[int, str]:
    """调用 CLI 并把 rich 渲染出的中文提示一并返回。

    ``--quick`` 走降级预算，单次约 25 s；不传该开关就是完整两阶段流程（留给 slow 作业）。
    """
    result = CliRunner().invoke(main, args)
    text = result.output
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        text += "\n".join(str(part) for part in result.exception.args)
    return result.exit_code, text


@pytest.fixture(scope="module")
def e2e_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """跑一次端到端命令，把输出目录交给多条断言复用（一次寻优 ~25 s）。"""
    out = tmp_path_factory.mktemp("e2e")
    code, output = _run(
        [
            "--thrust",
            "80",
            "--radius",
            "0.75",
            "--te-thickness",
            "0.0025",
            "--output-dir",
            str(out),
            "--quick",
            "--quiet",
            "--seed",
            "2026",
        ]
    )
    assert code == 0, f"CLI 退出码 {code}\n{output[-2000:]}"
    return out


def test_cli_exit_code_zero_and_files_complete(e2e_dir: Path) -> None:
    counts: dict[str, int] = {}
    for path in e2e_dir.iterdir():
        counts[path.suffix] = counts.get(path.suffix, 0) + 1
    for suffix, expected in EXPECTED_SUFFIX_COUNTS.items():
        assert counts.get(suffix, 0) == expected, f"{suffix} 文件数 {counts.get(suffix, 0)} != {expected}"
    assert (e2e_dir / "sec01_contour.dxf").exists()
    assert (e2e_dir / "sec10_contour.dxf").exists()
    assert (e2e_dir / "rotor_blade_solid.step").stat().st_size > 10_000
    assert (e2e_dir / "rotor_blade_solid.stl").stat().st_size > 10_000
    assert (e2e_dir / "manufacturing_manifest.csv").stat().st_size > 500


def test_cli_json_is_recoverable_to_equivalent_result(e2e_dir: Path) -> None:
    payload = json.loads((e2e_dir / "rotor_report.json").read_text(encoding="utf-8"))
    restored = RotorInputs.model_validate(payload["inputs"])
    assert restored.thrust == pytest.approx(80.0)
    assert restored.te_thickness == pytest.approx(0.0025)
    assert len(payload["sections"]) == 10

    from hover_rotor.config import RotorResult

    result = RotorResult.from_dict(payload)
    assert result.fm == pytest.approx(payload["fm"], rel=0, abs=1e-12)
    assert result.sections[0].points_mm.shape[0] >= 20


def test_cli_reports_aerodynamics_summary_and_te_consistency(e2e_dir: Path) -> None:
    from hover_rotor.config import RotorResult

    result = RotorResult.from_dict(json.loads((e2e_dir / "rotor_report.json").read_text(encoding="utf-8")))
    assert 0.0 < result.fm < 1.0
    assert result.thrust > 0 and result.power > 0
    assert result.tip_mach <= 0.65 + 1e-9
    assert all(sec.te_thickness == pytest.approx(0.0025) for sec in result.sections)
    assert result.r_over_R[-1] == pytest.approx(1.0)
    assert result.backend in ("analytic", "neuralfoil")


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        (["--thrust", "-1", "--radius", "0.75", "--te-thickness", "0.0025"], "推力必须为正值"),
        (["--thrust", "80", "--radius", "0.75", "--te-thickness", "0"], "钝尾缘厚度必须为正值"),
        (["--thrust", "80", "--radius", "-0.2", "--te-thickness", "0.0025"], "旋翼半径必须为正值"),
        (["--thrust", "80", "--radius", "0.75", "--te-thickness", "0.0025", "--blade-count", "9"], "叶片数"),
        (["--thrust", "80", "--radius", "0.75", "--te-thickness", "0.0025", "--hub-ratio", "0.9"], "轮毂比"),
    ],
)
def test_cli_rejects_out_of_domain_inputs_with_nonzero_code(args: list[str], needle: str) -> None:
    """判定准则：越界输入必须非零退出，并把工艺限值原样转述（中文消息不被翻译成英文）。"""
    code, output = _run([*args, "--output-dir", "./_should_not_be_written"])
    assert code != 0, f"越界输入被接受了：{args}"
    assert needle in output, f"输出未包含预期提示 {needle!r}：{output[:400]}"


def test_cli_help_documents_the_spec_signature() -> None:
    code, output = _run(["--help"])
    assert code == 0
    for option in ("--thrust", "--radius", "--te-thickness", "--output-dir"):
        assert option in output


@pytest.mark.parametrize(
    ("thrust", "radius", "te", "hub"),
    [
        (5.0, 0.127, 0.0005, 0.10),  # 微型穿越机
        (120.0, 0.75, 0.0012, 0.15),  # 工业级重载
        (4500.0, 1.75, 0.0025, 0.18),  # eVTOL 载人级
    ],
)
@pytest.mark.slow
def test_three_example_cases_run_through_the_same_stack(
    thrust: float, radius: float, te: float, hub: float
) -> None:
    """§Task 6.2 三套算例各跑通一次，并满足 §5 的可核验子集。

    用局部精调快速路径：算例验收关心的是**同一套代码能否外推三个量级**，
    而不是每个算例都榨到最优 FM（完整 DE 流程只在基准工况上验硬门禁）。
    """
    from hover_rotor.optimizer.solver import OptimizationSettings, build_result, run_rotor_optimization

    inputs = RotorInputs(thrust=thrust, radius=radius, te_thickness=te, blade_count=2, hub_ratio=hub)
    outcome = run_rotor_optimization(
        inputs,
        settings=OptimizationSettings(
            use_global=False, seed=3, polish_restarts=1, slsqp_maxiter=90, slsqp_ftol=1e-8
        ),
    )
    result = build_result(inputs, outcome)
    assert result.n_sections == 10
    assert result.thrust_error < 0.005, f"{thrust} N 工况推力误差 {100 * result.thrust_error:.2f}%"
    assert result.tip_mach <= 0.65 + 1e-9
    assert 0.0 < result.fm < 1.0
    assert result.power > result.power_ideal > 0.0
    assert all(sec.te_thickness == pytest.approx(te) for sec in result.sections)
    assert result.sections[0].r_over_R == pytest.approx(hub)


@pytest.mark.parametrize(
    "script", ["01_micro_drone_rotor.py", "02_industrial_uav_rotor.py", "03_evtol_scale_rotor.py"]
)
def test_example_scripts_are_importable_and_declare_main(script: str) -> None:
    """算例脚本必须能被编译并暴露 main()（CI 不跑完整算例，但至少保证不是死文档）。"""
    import py_compile

    path = Path(__file__).resolve().parents[1] / "examples" / script
    assert path.exists(), f"缺少算例 {script}"
    py_compile.compile(str(path), doraise=True)
    source = path.read_text(encoding="utf-8")
    assert "def main()" in source
    assert 'if __name__ == "__main__"' in source
