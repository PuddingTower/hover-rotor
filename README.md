# hover-rotor —— 端到端悬停旋翼气动优化与钝尾缘截面制造系统

> 输入三个工程指标：**目标推力 T**、**旋翼半径 R**、**钝尾缘绝对厚度 t_te**；
> 输出使悬停品质因数 FM 最大化的 **10 个钝尾缘截面**，以及可直接上架
> 数控 / 慢走丝线切割 / 3D 打印的 **DXF、STEP、STL** 与制造报表。

一条链路打通气动计算与数控车间：**拓展钝尾缘 CST 几何 → 可微极线代理 → 悬停 BEMT →
DE + SLSQP 两阶段反设计 → 制造数据导出**，全程无尖尾缘、无手工建模、无数据接口壁垒。

Apache-2.0 | Python ≥ 3.10 | 161 个测试：160 通过 + 1 个 `xfail(strict=True)`（如实记录 §5 唯一未达成项）；覆盖率 94.3%（门禁 ≥85%）

仓库：<https://github.com/PuddingTower/hover-rotor>

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

> **CI 徽标为什么是空的（原因写在仓库里，不靠外部说明）**：workflow 文件已在本地提交，
> 但 GitHub 服务器端拒绝本机的 OAuth 令牌推送它们——错误原文
> `refusing to allow an OAuth App to create or update workflow '.github/workflows/ci.yml' without 'workflow' scope`，
> 实测令牌 scopes 为 `gist, read:org, read:user, repo`。授权后（`gh auth refresh -s workflow` → `git push`）
> 放开下面这行即可；在此之前放一个跑不起来的徽标等于伪造状态——该 URL 实测返回 **404**：
> `![CI](https://github.com/PuddingTower/hover-rotor/actions/workflows/ci.yml/badge.svg)`
> 完整复测记录见 [`docs/verification.md`](docs/verification.md) §6；CI 的每一步都在本机跑过同样的命令，逐条结果见 §1、§2。

---

## 1 为什么是钝尾缘

尖后缘在复材模压、树脂充模、数控加工中工艺脆性极高：纤维富集不良、脱模破损、后缘颤振开裂。
钝尾缘（flatback）用一点气动代价换取实打实的结构与工艺收益：

| 尾缘构型 | t_te/c | c_l,max | c_d0 | 剖面惯性矩 I_xx | 加工良率 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 标准尖后缘 | ≤0.1% | 基准 | 基准 | 极低（易扭转颤振） | 边缘极易碎裂 |
| 微钝尾缘 | 0.5%~1.0% | +1.5~3.0% | +3~6% | +25% | 脱模破损显著降低 |
| 中度 Flatback | 1.5%~3.0% | +5.0~8.0% | +12~25% | +120% | 适合 CNC 一体成型 |
| 结构型 Flatback | 5.0%~10.0% | +10~18% | +40~90% | 3~5 倍 | 根部抗疲劳极佳 |

本系统把 **t_te 作为输入而不是事后校核**：参数化在数学底层杜绝尖后缘
（类函数在 ψ=1 归零，尾缘厚度项独占），所以任何优化迭代产生的几何都是**可制造的**。

## 2 快速启动

```bash
git clone https://github.com/PuddingTower/hover-rotor.git && cd hover-rotor
pip install -e ".[dev,cad,step]"          # step 拉 OpenCASCADE，只要气动链路则 .[dev]

hover-rotor-opt --thrust 80 --radius 0.75 --te-thickness 0.0025 --output-dir ./results
```

约 2 分钟（含差分进化全局粗搜）后，`./results/` 得到 37 个文件：
`sec01_contour.dxf … sec10_contour.dxf`、对应 `.dat` / `.csv`、
`rotor_blade_solid.step`、`rotor_blade_solid.stl`、`manufacturing_manifest.csv`、
`rotor_report.json`、`manufacturing_report.md`、`rotor_summary.png`、
`rotor_interactive_3d.html`。

常用开关：`--fast`（跳过全局搜索，只做局部精调）、`--quick`（缩短预算，仅验证链路）、
`--no-step`（无 OCC 内核环境）、`--seed`（固定后逐位可复现）、`--blade-count`、`--hub-ratio`。

作为库使用：

```python
from hover_rotor import RotorInputs, OptimizationSettings, build_result, run_rotor_optimization

inputs = RotorInputs(thrust=120.0, radius=0.75, te_thickness=0.0012, blade_count=2)
outcome = run_rotor_optimization(inputs, settings=OptimizationSettings(seed=2026))
result = build_result(inputs, outcome)
print(f"FM={result.fm:.4f}  T={result.thrust:.2f} N  M_tip={result.tip_mach:.3f}")
```

容器（`docker build -t hover-rotor .`）见文末说明——**本镜像未经实机 docker build 验证**。

## 3 数学内核（摘要，完整推导见 [`docs/theory.md`](docs/theory.md)）

**几何｜拓展 CST**

$$\zeta_{u,l}(\psi)=C_{0.5,1.0}(\psi)S_{u,l}(\psi)\pm\psi\frac{\Delta\bar z_{te}}{2},\qquad
C_{0.5,1.0}=\psi^{0.5}(1-\psi),\qquad
S=\textstyle\sum_i A_i\binom ni\psi^i(1-\psi)^{n-i}$$

恒有 $\zeta_u(1)-\zeta_l(1)=\Delta\bar z_{te}=t_{te}/c$。楔形项在取中弧线时相消，
故尾缘钝度不污染薄翼升力解，只经厚度（失速）与基底阻力（压差）起作用。

**气动｜悬停 BEMT**

$$FM=\frac{C_T^{3/2}}{\sqrt2\,C_P},\qquad
a^2=\frac{\sigma F c_n}{4\cos^2\phi-\sigma F c_n},\qquad
F=F_{tip}F_{hub}\ \text{(Prandtl)}$$

大诱导区用 Buhl–Glauert 型 $C^1$ 有理饱和，$a\le0.4$ 区间**恒等**（测试以 `atol=1e-15` 锁定）。

**气动｜基底阻力（Hoerner / Standish–van Dam）**

$$\Delta C_{d,base}=0.135\,\frac{t_{te}/c}{\left(C_{d,clean}\right)^{1/3}}$$

公式分母取**除**：验收区间 $t_{te}/c=0.02,\,C_d=0.01\Rightarrow[0.0125,0.0130]$ 只有除法能命中。

**离散｜余弦聚点**

$$\bar r_i=\bar r_{hub}+(1-\bar r_{hub})\cdot\tfrac12\big[1-\cos\tfrac{\pi(i-1)}{N-1}\big]$$

给出 0.150 / 0.1756 / 0.2494 / … / 1.0000，叶根与叶尖**两端同时**加密，端点严格闭合。

**优化**

$$\min_X\ J=-FM+100\Big(\frac{T-T_{target}}{T_{target}}\Big)^2+P_{geom},\quad
X=[c(4),\theta(4),M_{tip},\text{CST}_{root}(8),\text{CST}_{tip}(8)]$$

## 4 制造接口（详见 [`docs/manufacturing.md`](docs/manufacturing.md)）

| 产物 | 用途 | 单位与坐标框 |
| :-- | :-- | :-- |
| `secNN_contour.dxf` | 线切割 / 激光下料 | mm，弦向框，单一闭合 LWPOLYLINE 200 顶点 |
| `secNN_contour.dat` | 气动型线库 / 复材铺层 | Selig 格式（上表面尾缘→前缘，下表面前缘→尾缘） |
| `secNN.csv` | 校验与追溯 | 归一化 + 弦向 mm + 空间 mm + 扭角 + 气动工作点 |
| `rotor_blade_solid.step` | 五轴 CAM / 模具 | AP214，叶片坐标（z=径向，已含扭转） |
| `rotor_blade_solid.stl` | 3D 打印切片 | 水密流形（Euler χ=2，法向一致） |
| `manufacturing_manifest.csv` | 强度校核 | 含 A、x_cg、y_cg、I_xx、J 等结构力学量 |
| `rotor_report.json` | 复现 | 全配置 + 种子 + 收敛历史，可反序列化回 `RotorResult` |

车间要点：**DXF/DAT 是未旋转的弦向坐标**（下料件无方位），**STEP/STL 才带扭转与径向位置**，
两套数字不可互相覆盖；十截面绝对尾缘厚度极差实测 4e-16 mm（约束由参数化保证）。

## 5 验收状态（如实区分"已验证"与"未达成"）

| §5 红线 | 阈值 | 实测 | 状态 |
| :-- | :-- | :-- | :-- |
| 目标推力吻合度 | ≤0.5% | 0.0114%（四算例 0.011%~0.088%） | ✅ 已验证 |
| 悬停品质因数 FM | ≥0.68（硬门禁） | **0.7620** | ✅ 已验证，但**受设计边界支配**（见 §5 下方说明与算例表） |
| 叶尖马赫数 | ≤0.65 | 0.123（约束由变量边界保证，不可能违反） | ✅ 已验证 |
| 十截面 t_te 一致性 | <0.001 mm | 4e-16 mm | ✅ 已验证（由参数化恒等保证） |
| 含 DE 端到端耗时 | ≤3 min | 147 s / 154 s（seed 7 与 2026，FM 同为 0.7620） | ✅ 已验证，余量 14~18% |
| 快速路径耗时 | ≤30 s | 15–29 s（`polish_restarts=0` 配置） | ✅ 已验证 |
| 单次极线求解 | <5 ms | 0.26 ms（12 迎角点） | ✅ 已验证 |
| 覆盖率 | ≥85% | 94.3%（160 通过 + 1 个 xfail 记录，含 5 个 slow 门禁） | ✅ 已验证 |
| Caradonna–Tung 推力系数 | ≤5% | **+26.7%**（C_T=0.005815 vs 实测 0.00459，TM-81232 p.47 图题逐字提取） | ❌ **未达成**（原因与证据链见下方与 `docs/verification.md` §3） |
| STEP/STL 实体合法性 | 水密流形 | BRepCheck 有效、χ=2、0 边界边 | ✅ 已验证 |
| Docker 端到端 | build+run 平稳 | — | ⚠️ **未验证**（开发机无 Docker 守护进程） |
| CI 在 GitHub 上实际运行 | 工作流被接受并跑绿 | 服务器端拒绝推送 workflow（令牌无 `workflow` scope），徽标 URL 实测 404 | ⚠️ **受阻**；`ci.yml` 的每一步已在本机同命令跑过，见 `docs/verification.md` §1–§2、§6 |
| 三维视图离线交互 | 无网络依赖 | plotly.js 内联、4.94 MB、无外链 script | ✅ 已验证 |

**关于未达成的 5% 红线**（完整证据链见 `docs/verification.md` §3，推导见 `docs/theory.md` §4）：
判据锚点全部换成**可从文本层逐字提取的一次文献**（archive.org 影印件，HTTP 200，页码已标注）。
TM-81232 p.7 作者自述：用估计尾迹几何的 AMI **升力面**解 "too high by about 20%"；
改用实测涡迹后 8°→+4.3%、12°→+5.1%，而 5° 即使实测尾迹仍 +14.3%（0.0024 对 0.0021），从未进过 5%。
本实现（解析 2D 极线 + 环元 BEMT + Prandtl 损失 + 联立切向诱导）给 +26.7%，落在报告自述的方法层级
偏差带 [+15%, +35%] 内，测试以此判"实现正确"；而 ≤5% 属升力面/CFD 级保真度，与 D-02 为毫秒级
主循环指定的低阶层级冲突——**这不是 BEMT 层内可以"修 bug"抹平的差距**。
两条能在 BEMT 层内刷过 5% 的捷径（联立 a′ 只值 1.4%；全局入流放大 κ=1.33 让三点全进 ±5%
却使高实度设计工况寻优失败）都已被量化并**否决**，数据留在 `docs/theory.md` §4.1.1。
本实现在该判据上以 `xfail(strict=True)` 保留可执行记录：若将来接入高保真极线后端使其真的达标，
测试会以 XPASS 形式强制更新本表。

> 一处**诚实修正**：本节早期版本引用过一个二手"同方法公开 BEMT 复现 +21.6%（C_T≈0.0056）"，
> 本次复查未能定位到可逐字核验的出处链接，已从 README、`docs/theory.md` 与测试常量中删除，
> 锚点改用上面的原文页码。结论未变，但不再依赖不可复现的转述。

算例对照（同一套代码、同一组模型常数，不做量级切换）：

| 算例 | T / R / t_te | 求解路径 | 耗时 | FM | 推力误差 | M_tip | 转速 | 轴功率 | 说明 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 基准验收工况 | 80 N / 0.75 m / 2.5 mm | DE → SLSQP（完整两阶段） | 147~154 s | **0.7620** | 0.0114% | 0.128 | 553 rpm | 451 W | §5 各红线的评定工况 |
| 微型穿越机 | 5 N / 0.127 m / 0.5 mm | 局部精调 | 27 s | 0.6989 | 0.021% | 0.187 | 4791 rpm | 45 W | 内段 Re 低于代理下界，3 条外推告警见 `RotorResult.warnings` |
| 工业级重载 | 120 N / 0.75 m / 1.2 mm | 局部精调 | 25 s | 0.7715 | 0.078% | 0.174 | 756 rpm | 820 W | 尾缘更薄 → 基底阻力下降 → FM 与基准同档 |
| eVTOL 载人级 | 4500 N / 1.75 m / 2.5 mm | 局部精调 | 23 s | 0.8045 | 0.088% | 0.381 | 707 rpm | 77.4 kW | 同一套模型常数直接外推两个量级 |

> **读这张表的正确方式**：FM 的绝对值是**设计边界的函数**，不是物理结论。四个算例的最优解都把
> 展向弦长顶到 `chord_max_over_R`（0.26R / 0.30R）并把转速压到 `tip_mach_min` 附近——
> 因为目标函数只含 FM、推力罚与几何可行性罚（计划 §3 原文如此），不含失速裕度与刚度约束，
> 而型阻功率 ∝ Ω³ 天然偏疼"低速大弦长"。收紧弦长上界会显著拉低 FM。
> 上表用于验证**链路跨量级可用**，不可用于选型决策。

## 6 已知局限（不粉饰）

1. **FM 门禁的达成依赖设计边界**。基准工况最优解把弦长顶到 `chord_max_over_R=0.26R`、
   转速压到 `tip_mach_min=0.12`，展向工作迎角 10°~22°（越过 c_l,max）。这是给定目标函数
   （FM 极大 + 无失速裕度约束）的合法最优——型阻功率 ∝ Ω³ 使低速大弦长必然占优，
   与真实选型准则（先定转速/桨尖声速裕度，再定弦长）相反。
   要把它当设计工具用，须另加失速裕度、刚度/模态与噪声约束；本仓库未实现这些约束，因为
   计划.md 的目标函数里没有。`DesignBounds` 的默认值是本项目的工程选择，报告数值时不可当作物理结论。
2. **极线代理是工程解析模型**，不是 CFD/XFOIL 级精度；适用域 Re∈[3e4,5e6]、Ma≤0.75、
   t/c∈[0.04,0.24]、dz_te≤0.10，越界显式抛错。装好 `neuralfoil` 权重后接口可切换，
   但发布链与测试不下载权重（避免联网卡死主循环）。
3. **§1 与 Task 2.3 的入流公式不一致，已按 §1 补齐**：计划 §1 的
   `W=sqrt((Omega r - u_t)^2 + v_i^2)` 含切向诱导，Task 2.3 的迭代契约却写
   `phi=arctan(v_i/Omega r)`。本实现联立求解 a′（`bemt.py` §1.2 推导），
   但它只解释 1.5% 的 C_T 偏差，不是 §5 缺口的来源。
   另一个被**量化后否决**的路线（单一非均匀入流放大系数 κ=1.33，可把三个实测点压进 ±5%
   但会让高实度设计工况寻优失败）连同数据表记在 `docs/theory.md` §4.1.1，
   开关 `BEMTConfig.nonuniform_inflow_ratio` 保留但默认关闭。
4. **微型旋翼会被解析极线的雷诺数下界截断**。R≈0.13 m 级算例的内段 Re 落到 3e4 以下，
   BEMT 建表与报表层都钳位到边界评估，并把每一处写进 `RotorResult.warnings`；
   这类工况的绝对精度不应采信，链路可用性与告警是已验证的。
5. **扭转单调性由罚项引导而非硬约束**，极端初值下可能给出非单调扭转。
6. `rotor_interactive_3d.html` 约 4.9 MB（plotly.js 内联所致），逼近 5 MB 上限。
7. **CI 与 Release 工作流尚未在 GitHub 上运行**（令牌缺 `workflow` scope，服务器端拒绝推送），
   Dockerfile 也未做过实机 `docker build`。这两项在验收表里如实标为受阻/未验证，
   不靠"应该能跑"补齐；已在本地用同命令逐条跑过，见 `docs/verification.md` §1–§2、§6。

## 7 开发与验证

```bash
pip install -e ".[dev,cad,step]"
ruff check src tests && ruff format --check src tests
mypy src/
pytest -m "not slow" --cov=hover_rotor --cov-fail-under=85   # 核心门禁，几分钟内
pytest -m slow                                               # 含 DE 的端到端基准工况
python -m build                                              # 发布链自检：wheel + sdist
python examples/01_micro_drone_rotor.py                      # 三套算例见 examples/
```

仓库结构：`src/hover_rotor/{geometry,aero,optimizer,cad,visualization}` + `tests/` 12 个测试文件
+ `Dockerfile` + `docs/{theory,manufacturing,verification}.md`；
`.github/workflows/{ci,release}.yml` 已在本地提交、**尚未推送成功**（原因见 §5 表末行与
`docs/verification.md` §6）。

## 8 参考文献（每个链接都实测过 HTTP 状态）

规则：**未实测的链接不写进仓库**。下表状态为 2026-10-14 用 `curl` 在本机取得，
完整复测命令与逐字提取的页码见 `docs/verification.md` §5、§3。

| 文献 | 链接 | 状态 |
| :-- | :-- | :-- |
| Caradonna & Tung, *Experimental and analytical studies of a model helicopter rotor in hover*, NASA-TM-81232 (1981) | [archive.org 影印件（含文本层）](https://archive.org/details/NASA_NTRS_Archive_19820004169) | 200 |
| 同上，NTRS 官方记录页 | [ntrs.nasa.gov/citations/19820004169](https://ntrs.nasa.gov/citations/19820004169) | 403（反爬拦截脚本，浏览器可达；故正文一律引用上面的镜像） |
| C. Ferlisi, *Rotor Wake Modelling Using the Vortex-Lattice Method*, École Polytechnique de Montréal, 2018（$C_T$ 定义式 2.13、表 4.1 几何） | [publications.polymtl.ca/3059/1/208_CarloFerlisi.pdf](https://publications.polymtl.ca/3059/1/208_CarloFerlisi.pdf) | 200（前一日同 URL 曾不可达，可达性须带日期读） |
| 桨尖涡卷起的自适应涡格法研究（说明 BEMT 层级所缺的三维涡物理） | [arXiv:1805.00323](https://arxiv.org/abs/1805.00323) | 200 |

## 9 许可

Apache-2.0。见 [`LICENSE`](LICENSE)。
