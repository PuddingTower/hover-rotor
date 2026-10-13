"""现代化 CLI 终端交互入口（计划.md Task 5.3）。

命令签名（§Task 5.3）::

    hover-rotor-opt --thrust 100.0 --radius 0.8 --te-thickness 0.003 --output-dir ./results

一次命令走完整条反设计链路：输入校验 -> 两阶段寻优 -> 十截面 DXF/DAT/CSV ->
STEP/STL 放样 -> 制造报表 -> 二维图版与三维交互 HTML，并在终端用 rich
实时显示 DE 代数、推力残差与 FM。

三个刻意的设计选择：

1. **进度回调放在优化器里而不是靠轮询**：``OptimizationSettings.progress`` 每轮
   收到一条诊断记录，终端据此刷新表格。这样库调用者与 CLI 看到的是同一份诊断字典，
   不会出现"CLI 显示收敛但结果对象里没有"的双轨状态。
2. **CAD 导出失败不影响气动结论的交付**：build123d/OCC 缺席时（D-09）打印告警并继续，
   报表/图纸/图版仍全部落盘——它们只依赖 ezdxf 与 matplotlib。
3. **输入越界返回非零退出码并原样转述 Pydantic 的中文校验消息**，不重写成英文：
   操作者面对的是工艺限值（"钝尾缘厚度必须为正值…"），翻译反而增加误读风险。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hover_rotor import __version__
from hover_rotor.config import OptionalDependencyError, RotorInputs
from hover_rotor.geometry.cst import trailing_edge_gap
from hover_rotor.optimizer.objective import DesignBounds
from hover_rotor.optimizer.solver import OptimizationSettings, build_result, run_rotor_optimization

console = Console()


def _emit(entry: dict[str, Any]) -> None:
    """优化过程逐条打印。stage 名带 de_gen_* 时按代计数，否则视作阶段汇总。"""
    stage = str(entry.get("stage", ""))
    if stage.startswith("de_gen"):
        console.log(
            f"[green]{stage}[/]  FM={entry['fm']:.4f}  T={entry['thrust']:.2f} N  "
            f"残差={abs(entry['thrust_residual']):.3%}  M_tip={entry['tip_mach']:.3f}"
        )
    else:
        console.log(
            f"[bold cyan]{stage}[/]  J={entry['objective']:.5f}  FM={entry['fm']:.4f}  "
            f"T={entry['thrust']:.3f} N  残差={abs(entry['thrust_residual']):.3%}  "
            f"用时={entry['elapsed_s']:.1f}s"
        )


def _summary_table(inputs: RotorInputs, result: Any) -> Table:
    table = Table(title="悬停气动核心参数", show_lines=False, header_style="bold magenta")
    table.add_column("量", style="cyan", no_wrap=True)
    table.add_column("数值", justify="right")
    table.add_column("单位", style="dim")
    rows = [
        ("目标推力 T_target", f"{inputs.thrust:.3f}", "N"),
        ("实际推力 T", f"{result.thrust:.3f}", "N"),
        ("推力误差", f"{100 * result.thrust_error:.4f}", "%"),
        ("品质因数 FM", f"{result.fm:.4f}", "-"),
        ("推力系数 C_T", f"{result.ct:.6f}", "-"),
        ("功率系数 C_P", f"{result.cp:.7f}", "-"),
        ("轴功率 P", f"{result.power:.2f}", "W"),
        ("理想功率 P_ideal", f"{result.power_ideal:.2f}", "W"),
        ("扭矩 Q", f"{result.torque:.4f}", "N·m"),
        ("转速 Ω", f"{result.rpm:.1f}", "rpm"),
        ("叶尖马赫数", f"{result.tip_mach:.4f}", "-"),
        ("极线后端", result.backend, "-"),
        ("求解阶段", result.solver_stage, "-"),
        ("耗时", f"{result.elapsed_s:.1f}", "s"),
    ]
    for name, value, unit in rows:
        table.add_row(name, value, unit)
    return table


def _section_table(result: Any, radius: float) -> Table:
    table = Table(title="十截面档案（制造站位）", header_style="bold blue")
    table.add_column("Sec", justify="right")
    table.add_column("r/R", justify="right")
    table.add_column("c/R", justify="right")
    table.add_column("theta", justify="right")
    table.add_column("alpha", justify="right")
    table.add_column("t_te/c", justify="right")
    table.add_column("Re", justify="right")
    table.add_column("Ma", justify="right")
    table.add_column("cl", justify="right")
    table.add_column("cd", justify="right")
    for sec in result.sections:
        table.add_row(
            f"{sec.index:02d}",
            f"{sec.r_over_R:.4f}",
            f"{sec.chord / radius:.4f}",
            f"{sec.theta_deg:.2f}",
            f"{sec.alpha_deg:.2f}",
            f"{sec.dz_te:.4f}",
            f"{sec.re:.2e}",
            f"{sec.ma:.3f}",
            f"{sec.cl:.3f}",
            f"{sec.cd:.5f}",
        )
    return table


def _check_te_consistency(result: Any, inputs: RotorInputs) -> float:
    """十截面绝对尾缘厚度极差 [mm]。几何层的恒等式应当在数值精度内为 0。"""
    gaps = [
        abs(trailing_edge_gap(sec.points_normalized) * sec.chord - inputs.te_thickness)
        for sec in result.sections
    ]
    return 1000.0 * float(max(gaps)) if gaps else float("inf")


@click.command()
@click.option("--thrust", type=float, required=True, help="目标悬停推力 T [N]")
@click.option("--radius", type=float, required=True, help="旋翼外半径 R [m]")
@click.option("--te-thickness", type=float, required=True, help="钝尾缘绝对厚度 t_te [m]，全展恒定")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("./results"), show_default=True)
@click.option("--blade-count", type=int, default=2, show_default=True, help="叶片数 B，取值 2~6")
@click.option(
    "--hub-ratio", type=float, default=0.15, show_default=True, help="轮毂比 r_hub/R，取值 0.05~0.20"
)
@click.option("--seed", type=int, default=2026, show_default=True, help="随机种子，固定后结果逐位可复现")
@click.option("--fast", is_flag=True, help="跳过差分进化全局粗搜，只做局部精调（性能基线路径）")
@click.option("--de-maxiter", type=int, default=None, help="差分进化代数上限，缺省按预设预算")
@click.option("--no-step", is_flag=True, help="跳过 STEP/STL 放样（无 OCC 内核的环境或只要二维数据时）")
@click.option("--quick", is_flag=True, help="缩短精调预算，只用于验证链路贯通与出图，不作为设计结果")
@click.option("--quiet", is_flag=True, help="只打印最终摘要，不打印迭代过程")
@click.version_option(__version__, prog_name="hover-rotor")
def main(
    thrust: float,
    radius: float,
    te_thickness: float,
    output_dir: Path,
    blade_count: int,
    hub_ratio: float,
    seed: int,
    fast: bool,
    de_maxiter: int | None,
    no_step: bool,
    quick: bool,
    quiet: bool,
) -> None:
    """端到端悬停旋翼气动优化：给定 T / R / t_te，反设计 FM 最大的十截面钝尾缘桨叶并导出制造文件。"""
    try:
        inputs = RotorInputs(
            thrust=thrust,
            radius=radius,
            te_thickness=te_thickness,
            blade_count=blade_count,
            hub_ratio=hub_ratio,
        )
    except ValidationError as exc:
        console.print(
            f"[bold red]输入越界：[/] {exc.errors()[0]['type']} 于字段 "
            f"[bold yellow]{exc.errors()[0]['loc'][0]}[/]"
        )
        for err in exc.errors():
            console.print(f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}")
        console.print(
            Panel(
                "约束：T>0 且落在 [1, 20000] N；R>0 且落在 [0.05, 5] m；"
                "t_te>0 且落在 [0.3, 5] mm（尖尾缘 t_te=0 不被接受）；B∈[2,6]；r_hub/R∈[0.05,0.20]。",
                title="参数可行域 / feasible domain",
                border_style="yellow",
            )
        )
        raise SystemExit(2) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = DesignBounds.for_inputs(inputs)
    settings = OptimizationSettings(seed=seed, use_global=not fast, progress=None if quiet else _emit)
    if de_maxiter:
        settings.de_maxiter = de_maxiter
    if quick:
        # 演示/CI 用的降级预算：一次 SLSQP、无链式重启。FM 会低于 D-06 门禁，
        # 但十截面、STEP/STL、图版与报表的产出结构完全一致，足以验证链路。
        settings.slsqp_maxiter = 40
        settings.slsqp_ftol = 1.0e-6
        settings.polish_restarts = 0
        settings.de_maxiter = 2
        settings.de_popsize = 3

    console.print(
        Panel(
            f"T = {inputs.thrust} N   R = {inputs.radius} m   t_te = {inputs.te_thickness * 1000:.3f} mm   "
            f"B = {inputs.blade_count}   r_hub/R = {inputs.hub_ratio}   seed = {seed}\n"
            f"路径：{'局部精调 fast path' if fast else 'DE 全局粗搜 -> SLSQP 精调'}",
            title="hover-rotor 反设计",
            border_style="green",
        )
    )

    outcome = run_rotor_optimization(inputs, bounds=bounds, settings=settings)
    result = build_result(inputs, outcome, bounds)

    if not quiet:
        console.print(_summary_table(inputs, result))
        console.print(_section_table(result, inputs.radius))

    te_error_mm = _check_te_consistency(result, inputs)
    if inputs.tip_mach_max is not None and result.tip_mach > inputs.tip_mach_max + 1e-9:
        console.print(f"[bold red]违反压缩性约束：M_tip={result.tip_mach:.4f} > {inputs.tip_mach_max}[/]")
        raise SystemExit(3)

    from hover_rotor.cad.dxf_exporter import export_sections_dxf
    from hover_rotor.cad.table_export import export_manufacturing_tables

    written: list[Path] = []
    written += export_sections_dxf(result.sections, output_dir)
    enriched = export_manufacturing_tables(result, output_dir)
    written += [
        output_dir / "manufacturing_manifest.csv",
        output_dir / "rotor_report.json",
        output_dir / "manufacturing_report.md",
    ]

    if not no_step:
        try:
            from hover_rotor.cad.step_builder import build_blade_step

            step_path, stl_path = build_blade_step(result.sections, output_dir)
            written += [step_path, stl_path]
        except OptionalDependencyError as exc:
            console.print(f"[yellow]跳过 STEP/STL：[/] {exc}")

    try:
        from hover_rotor.visualization.plot_rotor_3d import render_interactive_rotor_html
        from hover_rotor.visualization.plot_sections import generate_summary_plots

        written.append(generate_summary_plots(enriched, output_dir / "rotor_summary.png"))
        written.append(render_interactive_rotor_html(enriched, output_dir / "rotor_interactive_3d.html"))
    except Exception as exc:  # noqa: BLE001 - 可视化不得吞掉已生成的制造数据交付
        console.print(f"[yellow]可视化降级：[/] {type(exc).__name__}: {exc}")

    console.print(
        f"[bold green]完成[/]  FM={result.fm:.4f}  T={result.thrust:.3f} N "
        f"(误差 {100 * result.thrust_error:.4f}%)  M_tip={result.tip_mach:.3f}  "
        f"截面尾缘厚度极差={te_error_mm:.2e} mm"
    )
    console.print(
        f"输出目录 [cyan]{output_dir}[/]  共 {len(written)} 个文件："
        f"DXF×{inputs.n_sections}、DAT×{inputs.n_sections}、CSV×{inputs.n_sections}、"
        "STEP、STL、manifest、report.json、report.md、summary.png、interactive.html"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
