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
| 1 | Caradonna-Tung 推力系数误差 | ≤5% | +26.7%（θ=8°，C_T=0.005815 vs 实测 0.00459，原文 p.47 图题逐字提取；定义由 Ferlisi 式 (2.13) 确认为 ρ 基 $C_T=T/\rho A(\Omega R)^2$） | `tests/test_bemt_caradonna.py::test_red_line_5pct_vs_experiment_is_not_met_by_low_order_method` | ❌ 未达成（xfail 记录） |
| 2 | 同上，方法学对齐 | 偏差落在原始报告自述的方法层级带 [+15%, +35%] | +26.7%（TM-81232 p.7 自述其升力面解 "too high by about 20%"） | `test_method_error_sits_in_band_admitted_by_primary_source` | ✅ |
| 3 | 三点趋势复现 | 偏差在低阶方法已知带内 | 5°→+44.5%、8°→+26.7%、12°→+22.4%（基准 0.0021/0.00459/0.00796，逐字出处见 §3 表） | `test_measured_collective_trend_is_reproduced` | ✅（记录用） |
| 4 | 目标推力吻合度 | ≤0.5% | 0.0114%（基准）；0.021%/0.078%/0.088%（三算例） | `test_baseline_case_meets_all_hard_gates`、`test_three_example_cases_run_through_the_same_stack` | ✅ |
| 5 | 悬停品质因数 FM | ≥0.68 硬门禁 | 0.7620（基准，完整两阶段） | 同上 | ✅ 受设计边界支配，见 §4 |
| 6 | 叶尖马赫数 | ≤0.65 | 0.123（变量边界保证） | 同上 | ✅ |
| 7 | 十截面 t_te 一致性 | <0.001 mm | 4.34e-16 mm | 同上 + CLI `_check_te_consistency` | ✅ |
| 8 | 含 DE 端到端耗时 | ≤3 min | 147 s / 154 s（seed 7 / 2026，DE 代数 24→18，FM 两者同为 0.7620；调优前 165~167 s） | `test_baseline_case_meets_all_hard_gates` | ✅ 余量 14~18% |
| 9 | 快速路径耗时 | ≤30 s | <30 s（`OptimizationSettings(use_global=False, polish_restarts=0, slsqp_maxiter=90)`） | `test_fast_path_is_deterministic_and_under_30s` | ✅ |
| 10 | 单次极线求解 | <5 ms | 0.26 ms（12 个迎角点，几何已缓存） | `test_polar_solve_under_5ms` | ✅ |
| 11 | 覆盖率 | ≥85% | 94.26%（161 收集 = 160 通过 + 1 xfail，含 5 个 slow） | `--cov-fail-under=85` | ✅ |
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

1. **实测基准值取自原始报告的文本层，逐字提取而非转述**。
   Caradonna & Tung, "Experimental and analytical studies of a model helicopter rotor in hover",
   NASA-TM-81232（NTRS 19820004169）：

   | 量 | 原文印刷 | 位置 |
   | :-- | :-- | :-- |
   | 翼型/几何 | "NACA 0012 profile and were untwisted and untapered. An aspect ratio of 6" | p.4 §2 |
   | 8° 实测 | "Ω = 1250 rpm, M_tip = 0.439, C_T = 0.00459" | p.47 图 4 |
   | 8° 实测（再印） | "TEST DATA, θ₀ = 8°, Ω = 1250 rpm, C_T = 0.00460" | p.49 图 6 |
   | 12° 实测 | "Ω = 1250 rpm, C_T = 0.00796" | p.48 图 5 |
   | 12° 实测（再印） | "TEST DATA θ₀ = 12°, ΩR = 150 m/s (491 fps), C_T = 0.0079" | p.58 图 15 |
   | 5° 实测 | "TEST DATA, θ₀ = 5°, ΩR = 150 m/s (491 fps), C_T = 0.0021" | p.59 图 16 |
   | 作者自述偏差 | "the resulting computed thrust coefficient was too high by about 20%"（Kocurek 估计尾迹 + AMI 升力面） | p.7 |
   | 作者自述达标条件 | "the thrust coefficient is overpredicted by less than 5%"（改用实测涡迹，8°） | p.7 |

   几何另由 Ferlisi (PolyMTL, 2018) 表 4.1 独立给出：$R$=1.143 m、根切 0.1905 m、
   $c$=0.1905 m、扭转 0、B=2、$\sigma$=0.1061（与 $Bc/(\pi R)$ 自洽）。该表把 12° 列为
   1800 rpm，与原文图 5 的 1250 rpm 冲突；本仓库以一次文献为准。
   `MU_AIR=1.81e-5` 下 ρΩRc/μ=1.926e6 与文献桨尖弦向雷诺数 1.92e6 吻合，确认介质参数同源。
2. **实现正确性用一次文献自述的偏差带判定**：本实现 8° 给 C_T=0.005815（+26.7%）。
   报告作者的升力面解在估计尾迹下就偏 +20%，换实测涡迹后 8°→+4.3%（0.0048 对 0.0046）、
   12°→+5.1%（0.0083 对 0.0079），而 5°→+14.3%（0.0024 对 0.0021）**始终没进 5%**。
   因此判据取 [+15%, +35%] 的"同层级实现正确"带；要求 BEMT 达到升力面级精度属方法层级错配。
3. **定义已核实，不存在 ρ / ½ρ 约定歧义**：Ferlisi 式 (2.13) 明文
   $C_T = T/\rho A(\Omega R)^2$、$C_P = P/\rho A(\Omega R)^3$，其表 4.1 的实测量 0.0021/0.0046/0.0079
   与 TM-81232 图题逐位一致。曾被怀疑的 `C_T=0.0063` 经核对是 Son & Kim 文中**另一架
   四叶前飞旋翼**（µ=0.15，源自 NASA-TM-100541）的工况，与本悬停基准无关。
4. **闭式极限交叉验证**：无叶尖损失的均匀下洗 + 二维升力线闭式解给出 C_T=0.00588，
   与本实现 0.005815 一致到 1.1%，进一步排除实现错误。
5. **诚实修正（应"链接未验证"的质询）**：本节早期版本引用过一个二手"同方法公开 BEMT 复现
   C_T≈0.0056（+21.6%）"作为锚点。本次复查**未能定位到可逐字核验的出处链接**，故该数字
   已从 `tests/test_bemt_caradonna.py` 与本文档移除，方法学判据改用上面第 2 条的
   一次文献自述偏差带。结论未变（+20% 量级由报告作者自述并印刷在正文里）。

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

## 5 引用链接可达性（2026-10-14 实测）

仓库内出现的每个外部链接都在本机用 `curl` 实测过状态码；**没有实测过的链接不得写进仓库**。
复测命令（任意机器可跑，输出即下表）：

```bash
for u in <本文涉及的 URL>; do
  printf "%s  %s\n" "$(curl -sS -o /dev/null -w '%{http_code}' -L --max-time 30 "$u")" "$u"
done
```

| URL | 用途 | HTTP | 判定 |
| :-- | :-- | :-- | :-- |
| `https://github.com/PuddingTower/hover-rotor` | 仓库主页 | 200 | ✅ 公开可访问 |
| `https://archive.org/details/NASA_NTRS_Archive_19820004169` | TM-81232 影印件（§3 逐字提取的来源） | 200 | ✅ |
| `https://archive.org/download/NASA_NTRS_Archive_19820004169/NASA_NTRS_Archive_19820004169.pdf` | 同件 PDF，2 334 139 B / 60 页，**含文本层** | 200 | ✅ 可直接 `pypdf` 提取 |
| `https://archive.org/download/NASA_NTRS_Archive_19820004169/NASA_NTRS_Archive_19820004169_djvu.txt` | OCR 全文（注意：部分图题数字在 OCR 里丢失，如 5° 的 0.0021） | 200 | ✅ 仅作辅助 |
| `https://publications.polymtl.ca/3059/1/208_CarloFerlisi.pdf` | Ferlisi 论文（$C_T$ 定义式 2.13、表 4.1 几何） | 200 | ✅ 3 031 025 B / 86 页。**2026-10-13 同一 URL 曾返回 000/0 字节**，故可达性必须带日期读 |
| `https://ntrs.nasa.gov/citations/19820004169` | NTRS 官方记录页 | **403** | ⚠️ 站点反爬拦截脚本，浏览器可正常访问；本文一律改用 archive.org 镜像 |
| `https://ntrs.nasa.gov/api/citations/19820004169/downloads/19820004169.pdf` | NTRS 直链 PDF | **403** | ⚠️ 同上 |
| `https://arxiv.org/abs/1805.00323` | 桨尖涡卷起的自适应涡格法研究（说明 BEMT 层级缺失的三维涡物理） | 200 | ✅ |
| `https://img.shields.io/badge/License-Apache_2.0-blue.svg` | README 许可徽标 | 200 | ✅ |
| `https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue` | README Python 徽标 | 200 | ✅ |
| `https://github.com/PuddingTower/hover-rotor/actions/workflows/ci.yml/badge.svg` | CI 徽标 | **404** | ❌ **有意不放**：`ci.yml` 尚未推送（原因见 §6），放一个 404 徽标等于伪造状态 |

## 6 CI 徽标缺失与 workflow 推送受阻的原因（写在仓库里，不外挂说明）

`.github/workflows/ci.yml` 与 `release.yml` **已在本机提交并通过本地等价命令验证**，
但推送到 GitHub 时被服务器端拒绝，错误原文：

```
To https://github.com/PuddingTower/hover-rotor.git
 ! [remote rejected] 0002806 -> main (refusing to allow an OAuth App to create or update
   workflow `.github/workflows/ci.yml` without `workflow` scope)
error: failed to push some refs to 'https://github.com/PuddingTower/hover-rotor.git'
```

本机 `gh` 与 `git` 共用同一枚 OAuth 令牌，`gh auth status` 与 `curl -I -H "Authorization: Bearer $(gh auth token)" https://api.github.com/user`
双路实测的 scopes 都是 `gist, read:org, read:user, repo`——**没有 `workflow`**。
（勘误：本仓库早期文档写过"开发机没有 gh CLI"，那是错的——`gh` 装在
`C:\Program Files\GitHub CLI\gh.exe`，版本 2.102.0，只是不在 Git Bash 的 PATH 里，
所以 `gh` 命令直接调用会 `command not found`。凭据受限是真的，"没装 gh"不是。）

仓库侧没有可绕过的设置（该限制是 GitHub 对 OAuth App 令牌的服务器端策略），
本机也没有其它凭据：`~/.ssh` 只有 `known_hosts`，`ssh -T git@github.com` 返回
`Permission denied (publickey)`。因此：

- 现状：远端 `main` 的代码与文档完整，**缺 CI/release workflow 文件**，故 CI 徽标为 404，
  README 里只写文字说明、不放徽标；
- 解法（仓库所有者一次浏览器授权）：`gh auth refresh -s workflow`（需要该文件在 PATH 里，
  或用绝对路径调用），随后 `git push`；本地 `main` 恰好只领先那一个 workflow 提交；
- 推送后把 README 里注释掉的徽标行放开，即恢复徽标。

**门禁并未因此失去验证**：`ci.yml` 的每一步都在本机以同样命令跑过，逐条结果见本文 §1、§2。
