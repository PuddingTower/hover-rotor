"""算例 2：工业级重载植保 / 巡检无人机桨叶反设计。

T = 120 N、R = 0.75 m、t_te = 1.2 mm —— 与基准验收工况同一量级，
差别在尾缘更薄（1.2 mm vs 2.5 mm），用于观察基底阻力对 FM 的边际影响。

    python examples/02_industrial_uav_rotor.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from hover_rotor.cad.dxf_exporter import export_sections_dxf
from hover_rotor.cad.step_builder import build_blade_step
from hover_rotor.cad.table_export import export_manufacturing_tables
from hover_rotor.config import OptionalDependencyError, RotorInputs
from hover_rotor.optimizer.solver import OptimizationSettings, build_result, run_rotor_optimization
from hover_rotor.visualization.plot_sections import generate_summary_plots

OUTPUT_DIR = Path("./results_industrial")


def main() -> None:
    inputs = RotorInputs(thrust=120.0, radius=0.75, te_thickness=0.0012, blade_count=2, hub_ratio=0.15)
    started = time.perf_counter()
    outcome = run_rotor_optimization(inputs, settings=OptimizationSettings(seed=2026))
    result = build_result(inputs, outcome)
    elapsed = time.perf_counter() - started

    print(f"工业级重载桨叶  T={result.thrust:.3f} N (误差 {100 * result.thrust_error:.3f}%)")
    print(f"FM={result.fm:.4f}  C_T={result.ct:.5f}  C_P={result.cp:.6f}  M_tip={result.tip_mach:.3f}")
    print(f"转速={result.rpm:.0f} rpm  轴功率={result.power:.1f} W")
    print(f"耗时={elapsed:.1f}s  评估={outcome.evaluations} 次  极线后端={result.backend}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_sections_dxf(result.sections, OUTPUT_DIR)
    enriched = export_manufacturing_tables(result, OUTPUT_DIR)
    generate_summary_plots(enriched, OUTPUT_DIR / "rotor_summary.png")
    (OUTPUT_DIR / "rotor_report.json").write_text(
        json.dumps(enriched.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    try:
        build_blade_step(result.sections, OUTPUT_DIR)
    except OptionalDependencyError as exc:
        print(f"跳过三维导出：{exc}")


if __name__ == "__main__":
    main()
