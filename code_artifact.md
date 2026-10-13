# 端到端悬停旋翼智能生成与气动优化系统 (AeroHover-Gen) 研发与开源全流程计划

> **项目代号**：`AeroHover-Gen` (Open-Source End-to-End Hover Rotor Generation & Optimization System)  
> **核心定位**：基于气动力学第一性原理与前沿优化算法，输入推力 ($T$)、半径 ($R$)、钝尾缘厚度 ($t_{te}$)，端到端全自动反设计并导出悬停品质因数 ($FM$) 最大化、十截面参数化、直接面向数控/复材模具生产的工业级旋翼系统。  
> **开发模式**：全流程模块化、原子化拆解，专为自主编码 Agent 与开发者协同研发、CI/CD 自动化验证与 GitHub 开源打造。

---

## 目录
1. [项目愿景与关键工程指标](#1-项目愿景与关键工程指标)
2. [前沿文献调研与核心理论基础](#2-前沿文献调研与核心理论基础)
3. [系统顶层架构与输入输出规范](#3-系统顶层架构与输入输出规范)
4. [Agent 原子化研发任务规划 (M0 ~ M6)](#4-agent-原子化研发任务规划-m0--m6)
   - [M0: 仓库工程脚手架与 CI/CD 基础设施](#m0-仓库工程脚手架与-cicd-基础设施)
   - [M1: 钝尾缘专用 CST 翼型参数化与几何生成引擎](#m1-钝尾缘专用-cst-翼型参数化与几何生成引擎)
   - [M2: 悬停叶素动量理论 (BEMT) 高保真气动求解器](#m2-悬停叶素动量理论-bemt-高保真气动求解器)
   - [M3: 钝尾缘翼型气动极线轻量级神经网络/代理模型](#m3-钝尾缘翼型气动极线轻量级神经网络代理模型)
   - [M4: FM 最大化双层分层逆向优化算法引擎](#m4-fm-最大化双层分层逆向优化算法引擎)
   - [M5: 十截面提取、NURBS 3D 放样与生产级 CAD (STEP/IGES/DAT) 导出](#m5-十截面提取nurbs-3d-放样与生产级-cad-stepigesdat-导出)
   - [M6: 交互式三维可视化看板、端到端验证与开源发布](#m6-交互式三维可视化看板端到端验证与开源发布)
5. [GitHub 开源仓库规范与文件目录结构](#5-github-开源仓库规范与文件目录结构)
6. [质量保障与工业级验收基准 (Benchmarking)](#6-质量保障与工业级验收基准-benchmarking)

---

## 1. 项目愿景与关键工程指标

在低空无人机、eVTOL（电动垂直起降飞行器）与传统直升机设计中，旋翼在纯悬停状态下的气动效率直接决定了航时与有效载荷能力。传统旋翼气动设计流程长、依赖专家经验，且容易忽略工业生产制造底线（如尖尾缘在数控切削、复材合模、注塑或3D打印中极易碎裂、缺胶或变形）。

本项目旨在研发一套**端到端（End-to-End）纯悬停旋翼智能生成开源框架**：
1. **纯参数化输入**：用户仅需给定目标推力 $T$ (N)、旋翼半径 $R$ (m) 以及工艺强制要求的**钝尾缘物理厚度 $t_{te}$ (mm)**（严格禁止尖尾缘）；
2. **全局性能最优**：以悬停品质因数（Figure of Merit, $FM$）最大化为单一主目标，自动寻优最优转速、扭转角沿展向分布、弦长沿展向分布及沿展向各特征截面的翼型型线；
3. **直接可生产输出**：程序输出沿展向精确计算的 **10 个关键特征截面（Airfoil Stations）**，生成平滑闭合的钝尾缘二维坐标云、三维 NURBS 实体曲面放样（STEP/IGES/STL）以及全套生产加工工艺数据表；
4. **全生命周期 Agent 驱动**：全套任务以原子化结构组织，具备前置依赖、代码规格、输入输出定义与自动化测试断言，可由智能代码 Agent 独立执行、调试并推送到 GitHub 仓库。

---

## 2. 前沿文献调研与核心理论基础

### 2.1 钝尾缘（Blunt Trailing Edge / Flatback）气动与制造机理
- **气动收益与失速延迟**：文献（*Standish & van Dam, OSTI 2003; Ram et al., MDPI 2023*）表明，在中高升力工况下，合理设计的钝尾缘（厚度一般为弦长的 $0.5\% \sim 2.5\%$）能够有效推迟上表面吸力面后部的附面层分离，提高最大升力系数 $C_{l,max}$，并缓和失速特性；
- **基础阻力控制与平衡**：钝尾缘带来的基底压差阻力（Base Drag）可通过优化尾缘前部的逆压梯度恢复区进行抵消。在悬停状态下，内侧及中段叶素运行于较高有效迎角，钝尾缘翼型能大幅提升该区域截面的升阻比 ($L/D$)；
- **生产制造必要性**：尖尾缘（Sharp TE, 理论厚度 0）在实际碳纤维复合材料成型中无法铺层、热压罐合模时容易夹伤纤维，CNC 刀具无法切削零半径边缘。钝尾缘（Flatback 或 Rounded Blunt）是航空旋转叶片的工业标准结构形式。

### 2.2 几何参数化：修正型函数/形状函数变换 (Modified CST)
采用 Kulfan 提出的 CST (Class Shape Transformation) 方法，并针对钝尾缘引入显式厚度边界修正项（*AIAA Journal, 2024*）：
$$y_u(x) = C(x) \cdot S_u(x) + x \cdot rac{\Delta z_{te}}{2}$$
$$y_l(x) = C(x) \cdot S_l(x) - x \cdot rac{\Delta z_{te}}{2}$$
其中：
- 类别函数（Class Function）：圆头平尾亚音速翼型取 $C(x) = x^{0.5}(1-x)^{1.0}$；
- 形状函数（Shape Function）：$S(x) = \sum_{i=0}^{n} A_i B_{i,n}(x)$，$B_{i,n}(x)$ 为 $n$ 阶 Bernstein 多项式基底；
- 钝尾缘厚度项：$\Delta z_{te} = rac{t_{te}}{c(r)}$，无量纲尾缘钝度根据当地展向截面弦长 $c(r)$ 严格精确匹配，确保无论截面弦长如何变化，物理尾缘厚度恒定等于用户输入的 $t_{te}$；
- 尾缘封闭方式：提供平面切平（Flatback）与切向圆弧（Round Blunt）两种标准化工业构造。

### 2.3 悬停气动力学理论与品质因数 (FM) 最大化
根据叶素动量理论（Blade Element Momentum Theory, BEMT，*Leishman: Principles of Helicopter Aerodynamics*）：
- **悬停品质因数定义**：
  $$FM = rac{P_{ideal}}{P_{actual}} = rac{T \cdot \sqrt{rac{T}{2ho A}}}{P} = rac{C_T^{3/2}}{\sqrt{2} C_P}$$
  式中 $A = \pi R^2$，推力系数 $C_T = rac{T}{ho A (\Omega R)^2}$，功率系数 $C_P = rac{P}{ho A (\Omega R)^3}$。
- **最佳下洗与环量分布（Betz-Goldstein 准则）**：纯悬停工况下诱导功率最小化的充要条件是整个旋翼盘面上的诱导下洗速度分布趋于均匀（$v_i(r) pprox 	ext{const}$）。
- **损失修正**：
  - 叶尖损失采用 Prandtl 叶尖损失因子 $F_{tip}(r) = rac{2}{\pi} rccos \left( \exp \left( -rac{B(R-r)}{2 r \sin \phi} ight) ight)$；
  - 叶根损失因子 $F_{root}(r) = rac{2}{\pi} rccos \left( \exp \left( -rac{B(r-R_{hub})}{2 r \sin \phi} ight) ight)$；
  - 总损失系数 $F = F_{tip} \cdot F_{root}$。

---

## 3. 系统顶层架构与输入输出规范

```
               +------------------------------------------------------+
               |                      用户输入                        |
               |   目标推力 T (N) | 旋翼半径 R (m) | 尾缘厚度 t_te (mm) |
               +------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           AeroHover-Gen 核心引擎                                  |
|                                                                                   |
|  +---------------------+    +----------------------+    +-----------------------+ |
|  |  工况解析与转速寻优 | -> | 全局环量分配逆向设计 | -> | 截面翼型气动智能反设计| |
|  |  (Tip Mach/Solidity)|    | (弦长 c(r) & 扭转 θ) |    | (CST钝尾缘参数寻优)   | |
|  +---------------------+    +----------------------+    +-----------------------+ |
|                                         |                                         |
|                                         v                                         |
|                    +------------------------------------------+                   |
|                    |     高保真 BEMT 气动闭环校验与微调       |                   |
|                    |   (FM 目标函数收敛 / 保证 T = T_target)  |                   |
|                    +------------------------------------------+                   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                  综合工业输出                                     |
|                                                                                   |
|  1. 十个特征截面型线数据库 (DAT/CSV, 严格保留 t_te 钝尾缘, 弦长, 扭转角, 安装角) |
|  2. 生产级 3D CAD 模型 (STEP / IGES 实体放样, STL 3D打印面网格)                   |
|  3. 性能报表 (FM, 功率消耗 P, 展向升力/下洗分布, 翼型极线图)                      |
|  4. 交互式可视化看板 (Web/HTML 3D 渲染器与 Plotly 性能图表)                       |
+-----------------------------------------------------------------------------------+
```

### 3.1 输入参数规格 (Inputs)
| 参数名 | 物理意义 | 典型范围 | 缺省处理/约束校验 |
| :--- | :--- | :--- | :--- |
| `thrust` ($T$) | 目标悬停推力 | $1.0 	ext{ N} \sim 20000.0 	ext{ N}$ | 必需参数，严格正数 |
| `radius` ($R$) | 旋翼外半径 | $0.05 	ext{ m} \sim 5.0 	ext{ m}$ | 必需参数，严格正数 |
| `te_thickness` ($t_{te}$) | 尾缘绝对钝度厚度 | $0.3 	ext{ mm} \sim 5.0 	ext{ mm}$ | 必需参数，强制检查 $t_{te} > 0$ 且 $t_{te}/c_{tip} \le 0.05$ |
| `blade_count` ($B$) | 旋翼叶片数 | $2 \sim 6$ 片 | 可选，默认 2 片 |
| `hub_radius` ($R_{hub}$) | 桨毂内径 | $0.05R \sim 0.2R$ | 可选，默认 $0.1R$ |
| `rho` ($ho$) | 空气密度 | 默认 $1.225 	ext{ kg/m}^3$ | 标准海平面大气 |
| `tip_mach_max` | 叶尖最大马赫数 | 默认 $0.65$ | 避免叶尖严重激波与气动噪声 |

### 3.2 十个特征截面径向位置分布 (10 Airfoil Stations)
为精确捕捉叶根涡脱落与叶尖损失急剧变化的梯度，10个截面采用**余弦/切比雪夫加权非均匀离散**：
$$ar{r}_i = rac{r_i - R_{hub}}{R - R_{hub}} = rac{1}{2} \left[ 1 - \cos\left( rac{i - 0.5}{10} \pi ight) ight], \quad i = 1, 2, \dots, 10$$
典型相对展向无量纲位置 $r_i/R$ 分布如下：
- Station 1: $r/R pprox 0.12$（叶根转接段）
- Station 2: $r/R pprox 0.18$（主要承力大厚度区）
- Station 3: $r/R pprox 0.28$（过渡段）
- Station 4: $r/R pprox 0.40$（主推力贡献起始段）
- Station 5: $r/R pprox 0.53$（高升阻比主翼段）
- Station 6: $r/R pprox 0.67$（高升阻比主翼段）
- Station 7: $r/R pprox 0.79$（峰值动力贡献段）
- Station 8: $r/R pprox 0.89$（叶尖过渡段）
- Station 9: $r/R pprox 0.95$（防激波防分离区）
- Station 10: $r/R pprox 0.985$（叶尖损失极线控制截面）

---

## 4. Agent 原子化研发任务规划 (M0 ~ M6)

每个任务均具备明确的 **任务目标**、**前置依赖**、**代码规格**、**输入输出**、**验证断言** 与 **Agent 运行命令**。

---

### M0: 仓库工程脚手架与 CI/CD 基础设施

#### 任务 TASK-001: 仓库目录初始化与 Python 包工程骨架
- **目标**：搭建符合现代 Python 规范的开源项目脚手架（基于 `pyproject.toml` 与 `src/` 布局），配置环境依赖。
- **前置依赖**：无
- **技术规范**：
  - 项目包命名：`aerohover`
  - Python 要求：`>=3.10`
  - 核心依赖：`numpy`, `scipy`, `pandas`, `tabulate`, `matplotlib`, `plotly`, `pydantic`
  - 可选 CAD 导出依赖：`cadquery` 或 `pythonocc-core` / 自研高保真 B-spline STEP 格式生成器
- **交付物**：
  - `pyproject.toml`
  - `src/aerohover/__init__.py`
  - `tests/conftest.py`
  - `.gitignore`, `README.md`, `LICENSE` (Apache-2.0 或 MIT)
- **验证断言**：运行 `pytest tests/` 正常返回退出码 0，`pip install -e .` 安装成功。
- **Agent 执行指令**：
  ```bash
  mkdir -p src/aerohover/{core,airfoil,optimizer,geometry,visualization,cli} tests
  git init
  pytest tests/
  ```

#### 任务 TASK-002: 自动化 CI 流程配置 (GitHub Actions)
- **目标**：编写 GitHub Actions 持续集成工作流，覆盖格式校验、类型检查、单元测试与覆盖率统计。
- **前置依赖**：TASK-001
- **交付物**：`.github/workflows/ci.yml`（包含 `ruff` 检查、`mypy` 静态类型检查、`pytest --cov`）。
- **验证断言**：CI 配置文件通过语法检查，本地 `pytest` 测试覆盖率门禁预设为 85%。

---

### M1: 钝尾缘专用 CST 翼型参数化与几何生成引擎

#### 任务 TASK-101: 钝尾缘 CST (Class Shape Transformation) 数学模块
- **目标**：实现带显式绝对钝尾缘厚度的 6~8 阶 Bernstein 多项式 CST 翼型表达与正反算算法。
- **前置依赖**：TASK-001
- **代码规格** (`src/aerohover/airfoil/cst.py`)：
  - 定义类 `BluntTECSTAirfoil`
  - 方法 `compute_coordinates(chord: float, te_thickness: float, n_points: int = 200, te_style: str = "flat") -> np.ndarray`
  - 输入：上表面系数向量 $A_u$（维数 $n$）、下表面系数向量 $A_l$（维数 $n$）、弦长 $c$、绝对尾缘厚度 $t_{te}$
  - 输出：形状为 $(N, 2)$ 的二维闭合型线点云，上表面自尾缘向头部、下表面自头部向尾缘顺序排列，尾缘两点垂直间距严格为 $t_{te}$。
- **验证断言**：
  - 断言 `np.abs((y_upper[-1] - y_lower[-1]) - te_thickness) < 1e-6`
  - 检查型线无自相交，前缘平滑曲率导数连续，前缘半径 $R_{le} > 0$。

#### 任务 TASK-102: 钝尾缘几何质量审计与修形算子
- **目标**：实现针对数控铣削与复材铺层的几何审计算子（最大厚度位置、厚度单调减小判定、脱模倾角审计）。
- **前置依赖**：TASK-101
- **代码规格** (`src/aerohover/airfoil/quality.py`)：
  - 计算各截面型线几何属性：最大厚度比 $(t/c)_{max}$、最大厚度纵向位置 $x_{tmax}$、前缘曲率半径 $r_{le}$、尾缘楔角 $eta_{te}$；
  - 判定算子：若上/下表面出现局部反弯凹坑或过小曲率（导致刀具干涉），自动产生惩罚项或执行局部样条光滑滤波。
- **验证断言**：对标准 NACA0012 附加 $1	ext{ mm}$ 钝尾缘进行测试，计算厚度与理论几何偏差 $< 0.1\%$。

---

### M2: 悬停叶素动量理论 (BEMT) 高保真气动求解器

#### 任务 TASK-201: 非线性诱导速度残差方程与高健壮性求解器
- **目标**：开发纯悬停工况专用的 BEMT 单微元流动平衡非线性方程求解器。
- **前置依赖**：TASK-001
- **数学方程** (`src/aerohover/core/bemt_element.py`)：
  - 局部入流角 $\phi = rctan\left(rac{v_i}{\Omega r}ight)$，有效迎角 $lpha = 	heta - \phi$
  - 叶素微元推力与转矩：
    $$dT = rac{1}{2} ho B c (V_{rel})^2 (C_l \cos\phi - C_d \sin\phi) dr$$
    $$dQ = rac{1}{2} ho B c (V_{rel})^2 (C_l \sin\phi + C_d \cos\phi) r dr$$
  - 动量理论推力平衡方程：
    $$dT = 4 \pi ho r F v_i^2 dr$$
  - 结合修正因子构建残差函数 $R(v_i) = 0$，采用带有自适应松弛因子的 Brent 方法或牛顿-拉夫逊方法求解，确保在失速区或大诱导下 $100\%$ 收敛。
- **验证断言**：在已知 $C_l, C_d$ 常数假设下，与解析 Betz 理论解对比，误差 $< 0.1\%$。

#### 任务 TASK-202: 旋翼整体悬停性能积分器与品质因数 (FM) 评估器
- **目标**：实现沿展向全叶片流动积分，输出总推力 $T$、总扭矩 $Q$、悬停需用功率 $P = Q \Omega$、诱导损失比、型阻损失比与 $FM$。
- **前置依赖**：TASK-201
- **代码规格** (`src/aerohover/core/rotor_evaluator.py`)：
  - 函数 `evaluate_rotor(rotor_geom, operating_condition) -> RotorHoverResult`
  - 内部实现对 10 个及高密度插值（$N=50$）展向站位的辛普森复合积分；
  - 计算品质因数：$FM = rac{T^{1.5}}{\sqrt{2 ho \pi R^2} \cdot P}$。
- **验证断言**：运行经典直升机测试算例（如 Harrington Rotor 1 或 Caradonna-Tung 旋翼实验数据），计算悬停推力与功率曲线吻合度在工程许可范围（误差 $< 3.5\%$）。

---

### M3: 钝尾缘翼型气动极线轻量级神经网络/代理模型

#### 任务 TASK-301: 钝尾缘翼型气动数据集生成与标准库搭建
- **目标**：建立钝尾缘翼型在低马赫数、中低雷诺数（$Re \in [5	imes 10^4, 2	imes 10^6]$）下的高精度气动极线预计算数据集与插值查询模块。
- **前置依赖**：TASK-101
- **代码规格** (`src/aerohover/airfoil/polar_db.py`)：
  - 预先收录并参数化典型钝尾缘基准翼型库（包括改进型 NACA 系列、S809 钝化型、VR-7/VR-8 钝尾缘旋翼翼型及超临界钝尾缘系列）；
  - 提供数据插值器：给定迎角 $lpha \in [-5^\circ, 20^\circ]$、雷诺数 $Re$、马赫数 $M$ 和尾缘钝度 $\Delta z_{te}$，极速返回 $(C_l, C_d, C_m)$。
- **验证断言**：在查询未录入网格点时，双三次样条插值光滑连续，升阻比极值连续性良好。

#### 任务 TASK-302: 轻量级嵌入式神经网络极线代理模型 (Neural Airfoil Surrogate)
- **目标**：构建不依赖外部复杂 CFD 软件的纯 Python 嵌入式微型 MLP 神经网络推理器，可直接输入 CST 参数预测极线。
- **前置依赖**：TASK-301
- **代码规格** (`src/aerohover/airfoil/surrogate.py`)：
  - 基于纯 NumPy 实现全连接层前向推理（权重固化为轻量 npz 格式，仅数MB大小，零网络依赖）；
  - 输入：CST 上下表面 12 个控制参数 + $\Delta z_{te}$ + $lpha$ + $Re$ + $M$；
  - 输出：$C_l, C_d$；
  - 单次评估时间 $< 0.1	ext{ ms}$，支持向量化批处理。
- **验证断言**：代理模型输出与测试集验证点相比，$C_l$ 相对误差 $< 2.0\%$，$C_d$ 均方根误差 $< 0.0015$。

---

### M4: FM 最大化双层分层逆向优化算法引擎

#### 任务 TASK-401: 顶层参数化与全局环量最优分布逆向解析
- **目标**：根据输入 $T$ 与 $R$，求解理想均匀下洗条件下的目标环量分布 $\Gamma(r)$，确定基准弦长分布 $c(r)$ 与扭转角分布 $	heta(r)$。
- **前置依赖**：TASK-202
- **算法细节** (`src/aerohover/optimizer/global_planform.py`)：
  - 求解最优转速 $\Omega$：在叶尖马赫数 $M_{tip} \le M_{max}$ 约束下，扫描旋翼实度 $\sigma = rac{B ar{c}}{\pi R} \in [0.04, 0.12]$，使叶片主要工作区处于翼型最高升阻比工况；
  - 逆向设计：根据 Betz-Goldstein 条件计算理想叶片展向平面形状，并在叶根处施加最小结构厚度约束，在叶尖施加防激波截面削薄约束。
- **验证断言**：针对输入 $T=20	ext{ N}, R=0.25	ext{ m}$，自动给出的最优转速与实度合理落在典型四旋翼最优悬停工作区间（$\Omega pprox 4000 \sim 6000	ext{ rpm}$）。

#### 任务 TASK-402: 截面翼型与扭转/弦长协同优化器 (SLSQP / Differential Evolution)
- **目标**：以 $FM$ 最大化为目标函数，在满足目标推力 $T \pm 0.1\%$ 约束以及物理钝尾缘 $t_{te}$ 约束下，协同寻优 10 个截面的气动外形。
- **前置依赖**：TASK-101, TASK-302, TASK-401
- **优化问题数学表述** (`src/aerohover/optimizer/rotor_opt.py`)：
  $$\max_{\mathbf{X}} FM(\mathbf{X})$$
  $$	ext{s.t.} \quad |T(\mathbf{X}) - T_{target}| \le \epsilon$$
  $$\quad t_{te, i} = t_{te} \quad (orall i = 1, \dots, 10)$$
  $$\quad c_{min} \le c_i \le c_{max}, \quad |	heta'_r| \le \Delta	heta_{max}$$
  设计变量向量 $\mathbf{X}$ 包含：
  - 旋翼转速 $\Omega$
  - 10 个截面的弦长 $c_i$、扭转角 $	heta_i$
  - 10 个截面的 CST 翼型形状参数（或截面厚度分布与中弧线参数）
- **验证断言**：运行端到端优化，100 次迭代内收敛，$FM$ 指标较常规矩形无扭转平板/常规薄翼旋翼提高 $15\% \sim 30\%$ 以上，达到 $0.72 \sim 0.79$ 优质悬停区间。

---

### M5: 十截面提取、NURBS 3D 放样与生产级 CAD (STEP/IGES/DAT) 导出

#### 任务 TASK-501: 工业级十截面离散数据标准结构生成器
- **目标**：规范化提取并导出最优状态下的 10 个特征截面的完整工程数据集。
- **前置依赖**：TASK-402
- **数据结构与格式** (`src/aerohover/geometry/sections.py`)：
  - 为每个截面生成：
    1. 标准 Selig 格式 `.dat` 文件（钝尾缘闭合坐标，上表面自尾缘向头部、下表面自头部向尾缘，严格标注物理尾缘切平尺寸）；
    2. 截面工程参数字典：径向位置 $r_i$、无量纲位置 $r_i/R$、真实弦长 $c_i$ (mm)、安装扭转角 $	heta_i$ (deg)、尾缘厚度 $t_{te}$ (mm)、当地雷诺数 $Re_i$、马赫数 $M_i$、设计工况升力系数 $C_{l,i}$ 与升阻比 $(L/D)_i$；
    3. 结构截面力学参数：截面截面积 $A$、重心偏移量 $(x_{cg}, y_{cg})$、扭转惯性矩 $J$。
  - 导出汇总表格为结构化 `csv` 与人类易读的 Markdown 报告。
- **验证断言**：十个截面的尾缘两点欧式距离在浮点公差内严格等于 $t_{te}$。

#### 任务 TASK-502: 三维空间空间叶片 NURBS 实体曲面放样与 CAD (STEP/IGES/STL) 导出
- **目标**：将 10 个离散截面通过三维刚体变换（缩放、扭转旋转、展向平移、前缘对齐）映射至三维空间，利用 B-spline 平滑插值生成符合工业标准的 3D 实体模型。
- **前置依赖**：TASK-501
- **代码规格** (`src/aerohover/geometry/cad_export.py`)：
  - 构建三维截面闭合空间多段线；
  - 钝尾缘侧面生成平整封闭垂直基底面（Blunt TE Face）；
  - 导出支持数控铣削与注塑模具设计的标准 `STEP` (ISO 10303-21) 文件；
  - 导出直接兼容 3D 打印切片机（如 Cura, PrusaSlicer, BambuStudio）的闭合流形高精度 `STL` 格式（无缝闭合实体，无非流形边缘）。
- **验证断言**：生成的 STL 文件使用三角网格流形检查函数检验，法向量朝外、孔洞数为 0、水密性（watertight）检验通过。

---

### M6: 交互式三维可视化看板、端到端验证与开源发布

#### 任务 TASK-601: 交互式 WebGL 3D 旋翼与气动仪表盘看板 (Plotly/HTML)
- **目标**：开发单文件、免额外服务器运行的交互式 HTML 可视化看板。
- **前置依赖**：TASK-501, TASK-502
- **代码规格** (`src/aerohover/visualization/dashboard.py`)：
  - 3D 旋翼叶片交互渲染视图：支持鼠标拖拽旋转、缩放、剖面线高亮；
  - 2D 翼型截面叠放与展开视图（标明钝尾缘特征与厚度对比）；
  - 气动参数展向分布折线图：下洗速度 $v_i(r)$、环量 $\Gamma(r)$、迎角 $lpha(r)$、弦长 $c(r)$、扭转角 $	heta(r)$；
  - 悬停品质因数仪表盘与功率损耗组成柱状图（诱导功率 vs 型阻功率）。
- **验证断言**：运行生成单文件 `rotor_report.html`，文件大小适中（$< 5	ext{MB}$），可在任意现代浏览器中离线直接秒级打开。

#### 任务 TASK-602: 端到端 CLI 命令行界面与一键生成主入口
- **目标**：封装极简、健壮的命令行交互界面。
- **前置依赖**：TASK-402, TASK-502, TASK-601
- **使用体验** (`src/aerohover/cli/main.py`)：
  ```bash
  # 最小化单行执行命令：
  aerohover generate --thrust 25.0 --radius 0.28 --te-thickness 0.8 --output-dir ./output/
  ```
  终端附带丰富进度条显示优化代数、当前推力残差与实时 $FM$。
- **验证断言**：CLI 参数校验完备，输入负数推力或尖尾缘（$t_{te} \le 0$）立即输出清晰友好的错误拦截与提示信息。

#### 任务 TASK-603: 完整开源文档体系、示例教程与 GitHub 发布
- **目标**：产出顶级的开源工程文档，包括高颜值 README、数学公式推导文档、理论白皮书与 3 个典型工业算例。
- **前置依赖**：所有前置任务
- **交付内容**：
  - `README.md`：包含核心特性动图、安装指南、快速开始、API 说明、算例对比表；
  - `docs/theory.md`：详细记录 BEMT、钝尾缘气动力学、CST 公式与文献引用；
  - `examples/` 目录下提供 3 套完整运行脚本：
    1. 小型穿越机/四旋翼无刷电机旋翼（$T=5	ext{ N}, R=0.127	ext{ m}, t_{te}=0.5	ext{ mm}$）；
    2. 工业级重载植保巡检无人机旋翼（$T=120	ext{ N}, R=0.75	ext{ m}, t_{te}=1.2	ext{ mm}$）；
    3. eVTOL / 载人级垂直起降悬停旋翼（$T=4500	ext{ N}, R=1.75	ext{ m}, t_{te}=2.5	ext{ mm}$）。
- **验证断言**：在全新虚拟环境中运行完整示例，3 分钟内完整输出全部 10 截面、CAD 模型与可视化报告。

---

## 5. GitHub 开源仓库规范与文件目录结构

开源代码仓库目录树结构严格遵循如下工程化组织标准：

```
aerohover-gen/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # 自动化测试与代码静态扫描
│       └── release.yml            # 自动化发版与构建 Wheel 包
├── docs/
│   ├── assets/                    # 架构图、3D 渲染图、性能对比图
│   ├── theory.md                  # 理论推导手册 (BEMT, 钝尾缘 CST, Betz 准则)
│   └── manufacturing.md           # 模具设计、CNC 加工与 3D 打印工艺指南
├── examples/
│   ├── 01_micro_drone_rotor.py    # 小型多旋翼旋翼生成示例 (5N, 5寸)
│   ├── 02_industrial_uav_rotor.py # 工业级无人机旋翼生成示例 (120N, 30寸)
│   └── 03_evtol_scale_rotor.py    # eVTOL 级大推力旋翼生成示例 (4500N)
├── src/
│   └── aerohover/
│       ├── __init__.py
│       ├── cli/                   # CLI 命令行入口
│       │   ├── __init__.py
│       │   └── main.py
│       ├── core/                  # 气动核心算法
│       │   ├── __init__.py
│       │   ├── bemt_element.py    # 单叶素动量守恒方程求解
│       │   ├── inflow_models.py   # 下洗速度与损失因子修正 (Prandtl/Glauert)
│       │   └── rotor_evaluator.py # 沿展向全叶片气动性能积分器
│       ├── airfoil/               # 钝尾缘翼型建模与代理
│       │   ├── __init__.py
│       │   ├── cst.py             # 钝尾缘修改型 CST 翼型生成器
│       │   ├── quality.py         # 翼型几何曲率与制造工艺性审计
│       │   ├── polar_db.py        # 极线数据库与基准翼型库
│       │   └── surrogate.py       # 嵌入式轻量神经网络极线代理
│       ├── optimizer/             # FM 最大化逆向优化引擎
│       │   ├── __init__.py
│       │   ├── global_planform.py # 最优转速与环量分配初始估计
│       │   └── rotor_opt.py       # 双层逆向优化求解器 (SLSQP / GA)
│       ├── geometry/              # 截面与 CAD 建模
│       │   ├── __init__.py
│       │   ├── sections.py        # 十截面提取、参数化与 DAT/CSV 生成
│       │   └── cad_export.py      # 3D NURBS 实体放样与 STEP/STL 导出
│       └── visualization/         # 可视化报表与交互式图形
│           ├── __init__.py
│           ├── plots.py           # 极线、下洗、弦长分布 2D 渲染
│           └── dashboard.py       # 单文件 WebGL 交互式 3D 看板
├── tests/                         # 单元测试与集成测试套件
│   ├── test_cst.py
│   ├── test_bemt.py
│   ├── test_surrogate.py
│   ├── test_optimizer.py
│   ├── test_cad_export.py
│   └── test_end_to_end.py
├── LICENSE                        # Apache 2.0 开源协议
├── pyproject.toml                 # 现代打包配置与依赖管理
├── README.md                      # 开源项目主说明文档
└── 计划.md                        # 本原子化项目实施任务计划书
```

---

## 6. 质量保障与工业级验收基准 (Benchmarking)

为保证 Agent 开发产出具备严格的工业可用性与学术可信度，设立以下四重硬性验收红线：

| 校验维度 | 评价指标 | 验收达标阈值 | 验证手段 |
| :--- | :--- | :--- | :--- |
| **气动保真度** | 推力吻合度 | $|T_{calc} - T_{target}| / T_{target} \le 0.5\%$ | 高密度 BEMT 积分复验 |
| **效率指标** | 悬停品质因数 ($FM$) | 达到当前推力与尺寸下的理论极限值区间（典型 $0.70 \sim 0.78$） | 与传统矩形平面叶片对照基准提升 $\ge 15\%$ |
| **尾缘物理约束** | 尾缘厚度一致性 | 所有 10 个截面尾缘绝对厚度严格等于输入 $t_{te}$（误差 $< 0.001	ext{ mm}$） | 坐标几何绝对间距扫描断言 |
| **几何与加工性** | CAD 模型实体水密性 | 导出的 STEP/STL 模型为合法实体（Watertight Manifold），法向量一致，无破面 | 几何拓扑检查工具自动断言 |
| **代码与运行性能** | 端到端生成耗时 | 单核 CPU 运行端到端优化至模型导出全流程耗时 $\le 30	ext{ 秒}$ | 基准机型压测计时 |
| **测试完备度** | 测试覆盖率 (Coverage) | 核心算法与工具链单测覆盖率 $\ge 85\%$ | GitHub Actions 自动化报表 |

---

## 7. Agent 执行行动指南 (Runbook for Developer / Agent)

当自动化智能 Agent 接管本项目时，请按照以下步调依次推进：

1. **Phase 1 (环境与基础类)**：优先实现 TASK-001、TASK-101 与 TASK-201，打通从 CST 翼型生成到 BEMT 单微元平衡求解的数学链路，执行单元测试确保无数值发散；
2. **Phase 2 (数据与代理)**：实现 TASK-301/302，完成轻量极线求解机制，提供纳秒级极线响应，为优化循环扫清速度瓶颈；
3. **Phase 3 (反设计与优化)**：串联 TASK-401 与 TASK-402，完成在给定 $T, R, t_{te}$ 约束下的 $FM$ 极大化寻优，输出最优收敛历史；
4. **Phase 4 (制造数据与输出)**：实现 TASK-501 与 TASK-502，导出十截面 `.dat`、参数表格以及三维实体 `STEP/STL`，运行水密性与几何自检断言；
5. **Phase 5 (交互与开源文档)**：组装 TASK-601、TASK-602 与 TASK-603，运行端到端测试，生成开源发布包与演示文档。

本计划兼备前沿学术理论严谨度与现代工业级软件工程可落地性，可即刻启动 Agent 研发流程。
