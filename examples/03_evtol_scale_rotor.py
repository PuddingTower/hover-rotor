"""算例 3：eVTOL / 载人级垂直起降旋翼反设计。

T = 4500 N、R = 1.75 m、t_te = 2.5 mm —— 盘面载荷 ~468 Pa，属升力巡航构型的
垂起旋翼量级。此算例的意义主要在**外推检验**：气动链路的无量纲形式与半径无关，
故同一套代码从 5 N 穿越机一路推到载人级，不切换任何模型常数。

    python examples/03_evtol_scale_rotor.py

注意：载人级旋翼的真实性价比还受桨尖速度、结构模态与噪声约束，本算例只按
计划.md §3 的目标函数（FM 极大 + 推力等式 + 几何可行性）求最优，
产出应作为**气动参考构型**而非定型方案。
"""

from __future__ import annotations

import time
from pathlib import Path

from hover_rotor.cad.dxf_exporter import export_sections_dxf
from hover_rotor.cad.step_builder import build_blade_step
from hover_rotor.cad.table_export import export_manufacturing_tables
from hover_rotor.config import OptionalDependencyError, RotorInputs
from hover_rotor.optimizer.objective import DesignBounds
from hover_rotor.optimizer.solver import OptimizationSettings, build_result, run_rotor_optimization
from hover_rotor.visualization.plot_rotor_3d import render_interactive_rotor_html

OUTPUT_DIR = Path("./results_evtol")


def main() -> None:
    inputs = RotorInputs(
        thrust=4500.0,
        radius=1.75,
        te_thickness=0.0025,
        blade_count=2,
        hub_ratio=0.18,
        tip_mach_max=0.65,
    )
    # 载人级垂起的弦长/扭转范围与无人机不同：叶尖弦长下限由 t_te/c_tip<=5% 自动抬到 0.05 m，
    # 这里额外把展向弦长上界放宽到 0.30R，避免低转速侧被边界卡住而虚报 FM。
    bounds = DesignBounds.for_inputs(inputs)
    bounds = DesignBounds(
        chord_min_over_R=bounds.chord_min_over_R,
        chord_max_over_R=0.30,
        theta_min_deg=bounds.theta_min_deg,
        theta_max_deg=bounds.theta_max_deg,
        tip_mach_min=bounds.tip_mach_min,
        tip_mach_max=bounds.tip_mach_max,
    )

    started = time.perf_counter()
    outcome = run_rotor_optimization(inputs, bounds=bounds, settings=OptimizationSettings(seed=3))
    result = build_result(inputs, outcome, bounds)
    elapsed = time.perf_counter() - started

    print(f"eVTOL 级旋翼  T={result.thrust:.2f} N (误差 {100 * result.thrust_error:.3f}%)")
    print(f"FM={result.fm:.4f}  C_T={result.ct:.5f}  C_P={result.cp:.6f}  M_tip={result.tip_mach:.3f}")
    print(f"转速={result.rpm:.1f} rpm  轴功率={result.power / 1000:.2f} kW  扭矩={result.torque:.2f} N·m")
    print(f"耗时={elapsed:.1f}s  评估={outcome.evaluations} 次")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_sections_dxf(result.sections, OUTPUT_DIR)
    export_manufacturing_tables(result, OUTPUT_DIR)
    render_interactive_rotor_html(result, OUTPUT_DIR / "rotor_interactive_3d.html")
    try:
        build_blade_step(result.sections, OUTPUT_DIR)
    except OptionalDependencyError as exc:
        print(f"跳过三维导出：{exc}")


if __name__ == "__main__":
    main()
