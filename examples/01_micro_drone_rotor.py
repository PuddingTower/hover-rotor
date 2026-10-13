"""算例 1：微型穿越机 / 小型四旋翼桨叶反设计。

T = 5 N、R = 0.127 m、t_te = 0.5 mm —— 盘面载荷极低、雷诺数处在分离泡敏感区
（Re ≈ 5e4~1e5），是检验「钝尾缘工艺约束能否在小尺寸上兑现」的典型工况。

    python examples/01_micro_drone_rotor.py

产出与 CLI 相同的一套制造文件（默认写入 ./results_micro）。
"""

from __future__ import annotations

import time
from pathlib import Path

from hover_rotor.cad.dxf_exporter import export_sections_dxf
from hover_rotor.cad.step_builder import build_blade_step
from hover_rotor.cad.table_export import export_manufacturing_tables
from hover_rotor.config import OptionalDependencyError, RotorInputs
from hover_rotor.optimizer.solver import OptimizationSettings, build_result, run_rotor_optimization

OUTPUT_DIR = Path("./results_micro")


def main() -> None:
    inputs = RotorInputs(
        thrust=5.0,
        radius=0.127,
        te_thickness=0.0005,
        blade_count=2,
        hub_ratio=0.10,
    )
    started = time.perf_counter()
    # 小直径旋翼的最优转速很高（M_tip 上限先到），DE 的搜索体积比工业级算例小，
    # 因此给一个更短的预算即可收敛。
    outcome = run_rotor_optimization(inputs, settings=OptimizationSettings(seed=1, de_maxiter=18))
    result = build_result(inputs, outcome)
    elapsed = time.perf_counter() - started

    print(
        f"微型穿越机桨叶  T={result.thrust:.3f} N (目标 {inputs.thrust} N, "
        f"误差 {100 * result.thrust_error:.3f}%)"
    )
    print(f"FM={result.fm:.4f}  C_T={result.ct:.5f}  C_P={result.cp:.6f}  M_tip={result.tip_mach:.3f}")
    print(f"转速={result.rpm:.0f} rpm  轴功率={result.power:.2f} W  理想功率={result.power_ideal:.2f} W")
    print(f"耗时={elapsed:.1f}s  目标函数评估={outcome.evaluations} 次")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_sections_dxf(result.sections, OUTPUT_DIR)
    export_manufacturing_tables(result, OUTPUT_DIR)
    try:
        step_path, stl_path = build_blade_step(result.sections, OUTPUT_DIR)
        print(f"STEP -> {step_path.name}, STL -> {stl_path.name}")
    except OptionalDependencyError as exc:
        print(f"跳过三维导出：{exc}")


if __name__ == "__main__":
    main()
