# 验收复现记录（§5 红线逐条）

本文件把 README「验收状态」表里每个数字的**复现命令与实测值**固定下来，
使验收结论可被第三方审计，而不是文档里的散文断言。

- 复现环境：Windows 10 / Python 3.14.4 / numpy 2.5.1 / scipy 1.18.1 / ezdxf 1.4.4 /
  build123d 0.13.0（cadquery-ocp 8.0.1.1）。
- 计划.md 的验收数值是**唯一判据**（D-06/D-07 裁定），本文件不另立标准。
- 每行注明"已实机验证"或"未验证"，未验证的不得被当作已通过。

## 1 复现命令

```bash
pip install -e ".[dev,cad]"
pip install -e ".[step]"        # OCC 内核；装不上则 STEP/STL 断言自动 skip

# 门禁三连（与 ci.yml 完全一致）
ruff check src/ tests/ examples/
ruff format --check src/ tests/ examples/
mypy src/
pytest -m "not slow" --cov=hover_rotor --cov-fail-under=85
pytest -m slow --durations=10

# 基准工况端到端（含 DE 全局粗搜），对应 D-07 的 3 分钟硬门禁
hover-rotor-opt --thrust 80 --radius 0.75 --te-thickness 0.0025 --seed 2026 --output-dir ./results
```

## 2 逐条实测

| # | 判据 | 阈值 | 实测 | 判定来源 | 状态 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | Caradonna-Tung 推力系数误差 | ≤5% | +26.7%（θ=8°，C_T=0.005815 vs 实测 0.00459；定义经 Ferlisi 明文确认为 ρ 基 $C_T=T/\rho A(\Omega R)^2$） | `tests/test_bemt_caradonna.py::test_red_line_5pct_vs_experiment_is_not_met_by_low_order_method` | ❌ 未达成（xfail 记录） |
| 2 | 同上，方法学对齐 | 与同方法公开值偏差 ≤6% | +3.8%（公开 BEMT 复现 0.0056） | `test_method_tracks_published_bemt_value_within_6_percent` | ✅ |
| 3 | 三点趋势复现 | 偏差在低阶方法已知带内 | 5°→+44.5%、8°→+26.7%、12°→+22.4%（基准取图题原印 0.0021/0.00459/0.00796） | `test_measured_collective_trend_is_reproduced` | ✅（记录用） |
| 4 | 目标推力吻合度 | ≤0.5% | 0.0114%（基准）；0.021%/0.078%/0.088%（三算例） | `test_baseline_case_meets_all_hard_gates`、`test_three_example_cases_run_through_the_same_stack` | ✅ |
| 5 | 悬停品质因数 FM | ≥0.68 硬门禁 | 0.7620（基准，完整两阶段） | 同上 | ✅ 受设计边界支配，见 §4 |
| 6 | 叶尖马赫数 | ≤0.65 | 0.123（变量边界保证） | 同上 | ✅ |
| 7 | 十截面 t_te 一致性 | <0.001 mm | 4.34e-16 mm | 同上 + CLI `_check_te_consistency` | ✅ |
| 8 | 含 DE 端到端耗时 | ≤3 min | 147 s / 154 s（seed 7 / 2026，DE 代数 24→18，FM 两者同为 0.7620；调优前 165~167 s） | `test_baseline_case_meets_all_hard_gates` | ✅ 余量 14~18% |
| 9 | 快速路径耗时 | ≤30 s | <30 s（`OptimizationSettings(use_global=False, polish_restarts=0, slsqp_maxiter=90)`） | `test_fast_path_is_deterministic_and_under_30s` | ✅ |
| 10 | 单次极线求解 | <5 ms | 0.26 ms（12 个迎角点，几何已缓存） | `test_polar_solve_under_5ms` | ✅ |
| 11 | 覆盖率 | ≥85% | 94%（161 测试全通过，含 5 个 slow） | `--cov-fail-under=85` | ✅ |
| 12 | BEMT vs Betz 均匀下洗解析解 | <0.1% | 0.0000%（独立 4001 点梯形求积 + λ 定点迭代对拍） | `test_betz_uniform_wake_matches_independent_implementation` | ✅ |
| 13 | 零载荷闭式动量极限 | <0.1% | 0.022%，FM→1.0000 | `test_momentum_theory_closed_form_in_vanishing_loading` | ✅ |
| 14 | 薄翼实现（NACA 2412） | α_L0、c_m,c4 复现文献值 | −2.04°（文献 −2.1°）、−0.0497（文献 −0.050） | `test_thin_airfoil_recovers_naca2412_constants` | ✅ |
| 15 | Hoerner 基底阻力基准点 | [0.0125, 0.0130] | 0.01253 | `tests/test_base_drag.py` | ✅ |
| 16 | DXF 回读 | 闭合 LWPOLYLINE、200 顶点、尾缘竖直无缝隙 | 通过 | `test_cad_export.py` | ✅ |
| 17 | STEP 实体合法性 | `BRepCheck_Analyzer.IsValid()`、闭合壳 | 通过（1 solid/1 shell，600 边各属 2 面） | `test_cad_export.py` | ✅ |
| 18 | STL 水密流形 | 每条边被 2 三角形反向共享、0 边界边、Euler χ=2 | 通过（8102 三角形，χ=2） | 同上（纯 Python 网格拓扑自检） | ✅ |
| 19 | 截面力学量 vs NACA0012 解析 | A、I_xx 偏差 <1% | ≈0.2% | `test_table_export.py` | ✅ |
| 20 | 图版 | ≥300 DPI、>100 KB、PIL 可读 | 4500×3150、299.99 DPI、794 KB | `test_visualization.py` | ✅ |
| 21 | 交互 HTML | plotly.js 内联、无外链、<5 MB | 4.94 MB，无 `<script src=http>`（文件内仅存 in-bundle 默认 URL 字符串，不触发请求） | 同上 | ✅ |
| 22 | CLI 端到端 | exit 0 + 产物齐全 | 37 文件（DXF×10、DAT×10、CSV×11、STEP、STL、JSON、MD、PNG、HTML） | `test_end_to_end.py` | ✅ |
| 23 | CLI 越界拒绝 | 非零码 + 中文提示 | 5 组越界输入全部非零退出 | 同上 | ✅ |
| 24 | 固定种子可复现 | 逐位一致 | 两个不同 seed 收敛到同一设计向量；同 seed 两次 `np.array_equal` 为真 | `test_seed_reproducibility_bitwise` | ✅ |
| 25 | Docker `build`/`run` 端到端 | 平稳出全套文件 | — | 无 | ⚠️ **未验证**（开发机无 Docker 守护进程，也无 podman/nerdctl） |
| 25b | `python -m build` 产出 wheel/sdist（Task 6.1 发布链） | 构建成功且内容完整 | wheel 27 项含 `hover_rotor/py.typed`、20 个模块、`entry_points.txt`（`hover-rotor-opt`）；sdist 54 项 | 本机执行 `python -m build --outdir ...` 后逐项目录核对 | ✅ |
| 26 | 三维视图浏览器交互 | 离线平移/旋转/高亮 | 结构断言通过（trace 存在、无网络依赖） | 同上 | ⚠️ 手势交互未做实机浏览器验证 |

## 3 §5 唯一未达成项的完整证据链

判据本身：计划.md §5「气动保真度 / Caradonna-Tung 推力系数误差 ≤5%」。

1. **实测基准值取自原始报告**，非二手转述。Caradonna & Tung, NASA-TM-81232
   （NTRS 19820004169）图题原文印刷：θ_c=5°→C_T=0.0021、8°→0.00459/0.00460、12°→0.00796/0.0079。
   试验件几何由三篇独立文献一致给出：R=1.143 m、c=0.1905 m、根切 0.1905 m、B=2、
   NACA 0012、未扭转、1250 rpm、M_tip=0.439；`MU_AIR=1.81e-5` 下 ρΩRc/μ=1.926e6
   与文献桨尖弦向雷诺数 1.92e6 吻合，确认介质参数同源。
2. **同方法公开复现也是这个量级**：公开 BEMT 实现（2D 极线 + Prandtl 叶尖损失）报告
   +37.0%（5°）/+21.6%（8°）/+21.7%（12°），对应 C_T≈0.0029/0.0056/0.0096。
   本实现 0.00582（含切向诱导）与该值偏差 3.8%，说明数值链路无误。
3. **定义已核实，不存在 ρ / ½ρ 约定歧义**：Ferlisi（PolyMTL, 2018）明文
   `C_T = T/ρA(ΩR)²`、`C_P = P/ρA(ΩR)³`，其表 4.1 给出 θ=5/8/12° 实测量 0.0021/0.0046/0.0079，
   与 TM-81232 图题原文逐位一致。曾被怀疑的 `C_T=0.0063` 经核对是 Son & Kim 文中**另一架
   四叶前飞旋翼**（µ=0.15，源自 NASA-TM-100541）的工况，与本悬停基准无关。
4. **原始报告作者自己的结论同向**：TM-81232 正文印刷 "the resulting computed thrust
   coefficient was too high by about 20%"（用估计尾迹几何的升力面解），
   只有把**实测尾迹几何**代入后误差才降到 5% 以内（8°→+4.3%、12°→+5.1%）。
4. **闭式极限交叉验证**：无叶尖损失的均匀下洗 + 二维升力线闭式解给出 C_T=0.00588，
   与本实现 0.00582 一致到 1%，进一步排除实现错误。

结论：≤5% 需要"实测尾迹几何 / 高保真极线"层级的模型，而计划 D-02 明确要求默认使用
毫秒级解析极线代理、§1 把 BEMT 指派为主循环求解器。两条要求在同一个验收数值上冲突，
**这不是可以在 BEMT 层内"修 bug"解决的问题**。

要真正达成，需要下列之一（都超出 Task 2.3 的契约范围，需产品决策后再排期）：
- 用实测尾迹几何的刚性尾迹修正（原文路线），或
- 离散螺旋涡尾迹的 Biot-Savart 诱导计算（计划 §1 表格里的"自由尾迹法"一档，
  单次评估 1–10 s，会顶穿 D-07 的 3 分钟门禁，只能用作优化后的校核层），或
- 接入 `[neural]` extras 的 NeuralFoil/XFOIL 级极线 + 非均匀入流修正。

## 3.1 为达标尝试并否决的两条路（数据留档）

- **联立切向诱导 a′（已实现，作为默认）**：8° 的 C_T 0.005902 → 0.005815（−1.5%）。方向正确、量级远不足。
- **非均匀入流全局放大 κ（未采纳为默认，`nonuniform_inflow_ratio` 保留为显式校核开关）**：
  用三个独立总距点最小二乘标定得 κ=1.33 → 三点误差 +4.70% / −4.02% / −4.10%（max 4.70%，
  **形式上满足 ≤5%**）。否决理由：同一 κ 下基准设计工况寻优**收敛失败**（推力残差 27.4%，
  弦长/转速双顶界），即全局乘子在低实度试验件上有效、在高实度设计问题上失效；
  且文献中不存在该量级的低阶修正（有限桨叶效应是叶尖局部的，typical-vs-infinity 在悬停符号相反）。
  完整论证与数据表见 `docs/theory.md` §4.1.1。

## 4 关于 FM 数字的正确读法

FM 的绝对值是**设计边界**（`DesignBounds`）的函数，不是物理常数：基准工况最优解把展向弦长
顶到 `chord_max_over_R`、把转速压到 `tip_mach_min` 附近。原因是计划 §3 给的目标函数只含
FM、推力等式与几何可行性三项惩罚，不含失速裕度/刚度/噪声约束，而型阻功率 ∝ Ω³ 使
"低速大弦长"必然占优。因此：

- 门禁判定（FM≥0.68）以固定边界与固定种子为准，可复现；
- 不得把 FM 数值当作跨工况可比的选型结论；收紧弦长上界会显著拉低 FM。
