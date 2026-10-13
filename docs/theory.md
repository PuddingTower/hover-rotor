# 理论推导（hover-rotor）

本文收录计划.md §1–§4 的三处核心推导，并给出**模型保真度的实测结论**。
正文用 LaTeX 记号，代码里的实现符号在每节末尾对照列出。

---

## 1 悬停动量理论与品质因数

### 1.1 FM 的定义与理想功率

作动盘在悬停中把静止来流加速到尾迹诱导速度 $2v_i$，盘处轴向速度为 $v_i$。
穿越 annulus（半径 $r$、宽度 $dr$）的质量流量为 $\mathrm{d}\dot m = \rho\,(2\pi r\,\mathrm{d}r)\,v_i$，
动量定理给出

$$
\mathrm{d}T = \mathrm{d}\dot m \cdot (2v_i) = 4\pi \rho\, r\, v_i^{2}\,\mathrm{d}r
\quad\Longrightarrow\quad
T = 2\rho A v_i^{2},\qquad A=\pi R^{2}
$$

（注意：含前行速度 $V_h$ 的通用形式 $\mathrm{d}T = 4\pi\rho r (\Omega r)^2 a(1+a)\mathrm{d}r$ 在悬停
$V_h\to 0$ 时退化为上式。把 $(1+a)$ 因子错误地留在悬停式里是低阶程序最常见的一处推导失误，
本实现按 $V_h=0$ 严格化简。）

理想功率即单位时间赋予气流的动能通量下限 $P_{\text{ideal}} = T v_i = T\sqrt{T/(2\rho A)}$。
实际轴功率 $P$ 还包含型阻消耗，于是

$$
FM = \frac{P_{\text{ideal}}}{P} = \frac{T v_i}{P}
    = \frac{C_T^{3/2}}{\sqrt{2}\,C_P},\qquad
C_T=\frac{T}{\rho A(\Omega R)^2},\quad
C_P=\frac{P}{\rho A(\Omega R)^3}
$$

$FM=1$ 当且仅当无型阻、无叶尖损失、下洗严格均匀——这正是 §1.4 用来对拍数值格式的解析极限。

### 1.2 叶素-动量耦合

叶素局部几何与运动学（$u_t$ 为切向诱导速度）：

$$
W=\sqrt{(\Omega r-u_t)^2+v_i^2},\qquad
\tan\phi=\frac{v_i}{\Omega r-u_t},\qquad
\alpha=\theta(r)-\phi
$$

$$
\mathrm{d}T = B\cdot\tfrac12 \rho W^2 c\,\big[c_l\cos\phi-c_d\sin\phi\big]\mathrm{d}r,\qquad
\mathrm{d}Q = B\cdot\tfrac12 \rho W^2 c\,\big[c_l\sin\phi+c_d\cos\phi\big]r\,\mathrm{d}r
$$

与 $\mathrm{d}T_{\text{mom}}$ 联立消去 $v_i$，得局部轴向诱导因子的闭式（$u_t$ 略去，见 §1.5）：

$$
a^2 \equiv \Big(\frac{v_i}{\Omega r}\Big)^2
     = \frac{\sigma F\,c_n}{4\cos^2\phi-\sigma F\,c_n},
\qquad \sigma=\frac{Bc}{2\pi r},\quad c_n=c_l\cos\phi-c_d\sin\phi
$$

分母即 $1-k$ 形式（$k=\sigma Fc_n/(4\cos^2\phi)$），$k\to1$ 时 $a$ 发散——湍流尾迹态的数学征候。

**关于 $u_t$（切向诱导）的取舍**：计划 §1 的 $W$ 表达式含 $u_t$，而 Task 2.3 的实现契约写的是
$\phi=\arctan(v_i/\Omega r)$，两处不一致。本实现取后者，因为悬停下 $a'$ 的量级可由
$aa' = \sigma(1-a')^2c_t/(4\cos^2\phi)$ 估计：典型 $a\approx0.08$、$\sigma c_t\approx3.5\times10^{-3}$
给 $a'\approx1.1\%$，对 $\phi$ 的影响 $\sim0.05^\circ$，远小于极线代理自身的不确定度。
代价是 $\mathrm{d}Q$ 里少了一份涡流损失，落在 §5 FM 结论的保守侧。

### 1.3 Prandtl 叶尖/叶根损失

有限叶片数使附着涡沿展向重新分布，端部环量归零：

$$
F_{\text{tip}}=\frac{2}{\pi}\arccos\!\Big[\exp\!\Big(-\frac{B(R-r)}{2r\sin\phi}\Big)\Big],\qquad
F_{\text{hub}}=\frac{2}{\pi}\arccos\!\Big[\exp\!\Big(-\frac{B(r-R_{\text{hub}})}{2R_{\text{hub}}\sin\phi}\Big)\Big]
$$

$F$ 折减的是**环量**，因此它必须同时进入诱导方程与载荷积分；只在一处使用会造成
「入流按三维、积分按二维」的系统性推力高估（本项目开发中实测该不一致值达 20%）。
叶根项分母取 $R_{\text{hub}}$ 而非局部 $r$：叶根涡系的特征尺度是桨毂半径。

### 1.4 Betz 均匀下洗解析解（数值格式的判据）

刚性均匀尾迹、无限叶片、常数 $c_l,c_d$ 时，整张盘只有一个未知量
$\lambda=v_i/(\Omega R)$，力平衡为标量方程

$$
2\rho A(\lambda\Omega R)^2 = \int_{R_{\text{hub}}}^{R}
\tfrac12 \rho \frac{(\Omega r)^2}{\cos^2\phi(\lambda)}\,Bc\,
\big[c_l\cos\phi(\lambda)-c_d\sin\phi(\lambda)\big]\,\mathrm{d}r,
\qquad \phi(r;\lambda)=\arctan\frac{\lambda}{r/R}
$$

求解器的 `wake="uniform"` 分支用 Brent 求根 + Simpson 求积复现该解；测试里用
**另一套独立实现**（4001 点梯形求积 + $\lambda$ 定点迭代）对拍，推力偏差 $<0.1\%$。
另有纯闭式极限判据：$c_d=0,\ c_l=0.02$ 时
$T=\tfrac12\rho\Omega^2 B c\,c_l (R^3-R_{\text{hub}}^3)/3$、$v_i=\sqrt{T/(2\rho A)}$、$FM\to1$，
实测吻合到 $0.02\%$。这两条把「积分/迭代格式」与「气动模型」的误差彻底分离。

### 1.5 Buhl–Glauert 连续过渡

$a>a_c=0.4$ 后动量理论多值。本实现以 $C^1$ 有理式饱和，且在 $a\le a_c$ 区间**恒等**
（测试用 `atol=1e-15` 锁定"弱载荷工况不得留下任何修正痕迹"）：

$$
a = a_c + \Delta\frac{x}{1+x},\qquad x=\frac{|a_{\text{mom}}|-a_c}{\Delta},\ \Delta=0.25
$$

在 $x=0$ 处 $\mathrm{d}a/\mathrm{d}a_{\text{mom}}=1$，故衔接光滑；$a$ 有界于 $a_c+\Delta=0.65$，
使深度失速的叶内段不再让迭代发散——这是第二阶段 SLSQP 能穿过该区的前提。
诱导因子还保留符号：深度负载荷对应**上洗**（$v_i<0$）；把 $k$ 钳到非负会造成
「零载荷→零入流→大迎角→大载荷」的二周期振荡。

---

## 2 钝尾缘拓展 CST

### 2.1 参数化

弦向与厚度归一化坐标 $\psi=x/c,\ \zeta=z/c$：

$$
\zeta_u=C_{N_1,N_2}(\psi)S_u(\psi)+\psi\frac{\Delta\bar z_{te}}{2},\qquad
\zeta_l=C_{N_1,N_2}(\psi)S_l(\psi)-\psi\frac{\Delta\bar z_{te}}{2}
$$

$$
C_{0.5,1.0}(\psi)=\psi^{0.5}(1-\psi)^{1.0},\qquad
S(\psi)=\sum_{i=0}^{n}A_i K_{i,n}\psi^i(1-\psi)^{n-i},\quad K_{i,n}=\binom{n}{i}
$$

类函数在 $\psi=1$ 归零，故

$$
\zeta_u(1)-\zeta_l(1)=\Delta\bar z_{te}=t_{te}/c
$$

**恒等成立**：尾缘厚度由参数化结构保证，不是后处理修补，尖尾缘在数学上不可表示。
绝对厚度 $t_{te}$ 全展恒定而 $\Delta\bar z_{te}=t_{te}/c(r)$ 逐站位变化，这是工艺约束的核心。

`round` 封头模式以尾缘中点为心、半径 $\Delta\bar z_{te}/2$ 的半圆接在上下两角点之间，
两种模式下角点距严格为 $t_{te}$。

### 2.2 中弧线与钝尾缘的解耦

上下表面取平均时楔形项相消：$\zeta_c=(\zeta_u+\zeta_l)/2=C\cdot(S_u+S_l)/2$。
因此薄翼理论在中弧线上求得的 $\alpha_{L0}$、$\mathrm{d}c_l/\mathrm{d}\alpha$、$c_{m,c/4}$
**不受尾缘钝度污染**，钝度只经由厚度（失速）与基底阻力（压差）起作用。
这正是 §2.2 所述流动机理在参数化层面的对应物，也解释了为何
`c_{l,max}` 与 `c_{d0}` 随 $t_{te}/c$ **同时**上升而升力线斜率几乎不变。

实现校验（`tests/test_polar.py`）：把 NACA 2412 拟合进 CST 空间后，薄翼积分复现
$\alpha_{L0}=-2.04^\circ$（文献 $-2.1^\circ$）、$c_{m,c/4}=-0.0497$（文献 $-0.050$）。

### 2.3 基底阻力（D-01 裁定）

$$
\Delta C_{d,\text{base}} = 0.135\,\frac{t_{te}/c}{\big(C_{d,\text{clean}}\big)^{1/3}},
\qquad c_d = c_{d,\text{profile}} + \Delta C_{d,\text{base}}
$$

报告排版写作乘法形式，但 Task 2.1 的验收区间（$t_{te}/c=0.02,\ C_d=0.01 \Rightarrow [0.0125,0.0130]$）
只有**除法**能命中：$0.135\times0.02/0.01^{1/3}=0.01253$。故裁定验收值为权威、排版为笔误。
物理含义是低阻（薄尾流）剖面对同样的尾缘钝度更敏感。$C_{d,\text{clean}}$ 钳到 $\ge10^{-6}$ 防除零。

### 2.4 极线代理的可微结构

- 升力：薄翼线性段 + smooth-softplus 饱和
  $c_l = s\big(|c_{l,\text{lin}}|-\mu\ln(1+e^{(|c_{l,\text{lin}}|-c_{l,\max})/\mu})\big)$，
  大迎角再向平板分离解 $1.98\sin\alpha\cos\alpha$ 以 $\tanh$ 权重过渡。全程 $C^\infty$。
- 阻力：$c_{d0}$ 由「双面湿面积 × ITTC 摩擦系数 × 形状因子 × 低雷诺数分离泡惩罚」给出；
  抛物线斜率 $k$ 由 **Eckhous 相关式**闭合 $k=1/\big(4c_{d0}(L/D)_{\max}^2\big)$，
  乘洁净表面校正 1.6（Eckhous 内含风洞粗糙度假设）。
  这一闭合让 $c_d$ 在 $\alpha\to0$ 处保持有限 $c_{d0}$，同时把阻力上升速率钉在经验相关式上。
- 压缩性：Prandtl–Glauert 修正升力线斜率；波阻 $c_{d,w}=0.35\,\text{softplus}^2(M-M_{cr})$，
  $M_{cr}$ 随厚度/钝度/升力下降。

§2.4 的构型收益表以单调保形插值嵌入 $c_{l,\max}$ 增益；实测
$t_{te}/c:0.001\to0.05$ 使 $c_{l,\max}$ 上升 $+14.8\%$（表列结构型 flatback 为 $+10\%\sim18\%$），
零升阻力单调上升、最大升阻比单调下降。

---

## 3 展向离散与反设计

### 3.1 余弦聚点（D-05 裁定）

$$
\bar r_i=\bar r_{hub}+(1-\bar r_{hub})\cdot\tfrac12\Big[1-\cos\frac{\pi(i-1)}{N-1}\Big]
$$

报告原式缺 $\tfrac12$，为**非对称**半波：间距单调递增、只在叶根聚点，与报告正文
「梯度在两端急剧变化、须在两端集中采样」自相矛盾。取半波（Chebyshev–Lobatto 型）后
$\bar r = 0.150, 0.1756, 0.2494, \dots, 1.0$ 与参考表前三站位逐位吻合且端点严格闭合。

展向弦长/扭转用 **PCHIP**（单调保形分段三次 Hermite）重构：4 个控制节点、无过冲。
测试断言对单调递减的 $c_{ctrl}$ 插值结果处处单调。

### 3.2 优化问题

$$
\min_X\ J(X)=-FM(X)+w_T\Big(\frac{T-T_{\text{target}}}{T_{\text{target}}}\Big)^2+P_{\text{geom}}(X),
\qquad w_T=100
$$

$$
\text{s.t.}\quad M_{\text{tip}}\le0.65,\quad c_{\min}\le c(r)\le c_{\max},\quad
\theta_{\min}\le\theta(r)\le\theta_{\max},\quad t_{te}(r)\equiv t_{te,\text{input}}
$$

$X$ 为 25 维：4 点弦长 + 4 点扭转 + $M_{\text{tip}}$ + 叶根/叶尖两组 CST 权重（各 4+4）。
尾缘厚度**不**进设计空间——它由 $t_{te}/c(r)$ 在几何层派生，恒等于输入值。
$M_{\text{tip}}$ 直接作为变量并以箱式边界封顶，使硬约束不可能被违反。
所有罚项写成二次/softplus 形式，无 if 分段，保证 SLSQP 拿到连续梯度。

两阶段求解：`differential_evolution`（宽边界全局粗搜）→ `SLSQP`（局部精调），
并对 DE 最优与工程初值**分别**精调取优者：25 维上 DE 只跑几十代时常落在次优盆地，
单点精调会把全局阶段的优势抵消掉（实测两种做法的 FM 差 0.05）。

---

## 4 模型保真度实测结论（重要）


### 4.1 Caradonna–Tung 基准（§5 红线第一条）

试验件：NASA-TM-81232，2 片矩形未扭转 NACA0012，$R=1.143$ m、$c=0.1905$ m、
桨根切断 $r=0.1905$ m（$R/c=6$）、1250 rpm、$M_{\text{tip}}=0.439$。
图题原文印刷的实测推力系数：$\theta=5^\circ\to0.0021$、$8^\circ\to0.00459$（另两处印 0.0046/0.00460）、$12^\circ\to0.00796$。定义经 Ferlisi(2018) 明文确认为 $C_T=T/\rho A(\Omega R)^2$（非 $\tfrac{1}{2}\rho$ 基），故不存在约定歧义。

本求解器结果（已含 §1.2 联立 $a'$ 修正）：$C_T=0.003035/0.005815/0.009742$，对图题原印
$0.0021/0.00459/0.00796$ 的误差为 **+44.5% / +26.7% / +22.4%**。

**结论：§5 的 "≤5%" 红线，低阶方法在物理上达不到，本实现未达成。** 依据：

1. 同一试验件、同一方法（2D 极线 + BEMT + Prandtl 损失）的公开复现给出
   +37%/+21.6%/+21.7%（$C_T\approx0.0056$ @ 8°）。本实现与该同方法值的偏差为 +3.8%
   （0.005815 对 0.0056），说明数值链路正确，差距属方法固有偏差而非实现错误。
2. TM-81232 原文自述：改用估计尾迹几何的升力面解 "too high by about 20%"；
   只有把**实测尾迹几何**代入，误差才降到 5% 以内（+4.3% @ 8°、+5.1% @ 12°）。
   即"≤5%"是升力面/CFD 级别的保真度，而报告又同时把求解器指派为
   「全局初搜与反设计主循环」（§1 表格），并要求默认后端是毫秒级解析代理（D-02）。
   这两项要求在同一个验收数值上不可同时满足。
3. 闭式无叶尖损失的均匀下洗解给出 $C_T=0.00588$，与本实现 0.005815 几乎相同——
   进一步说明剩下的差距不在数值格式，而在"二维极线 + 环元动量"这一模型层级。

因此该判据在测试里以 `xfail(strict=True)` 保留：若将来接入高保真极线后端或实测尾迹修正
使误差真的降到 5% 以内，测试会 XPASS 并强制更新本节的结论。

### 4.1.1 为达标尝试并**否决**的两条路（附实测数据）

**(a) 联立切向诱导 a'** —— 已实现并作为默认（见 §1.2）。效果：8° 的 C_T 由 0.00590 降到
0.00582，即 -1.4%。与 a'≈1% 的理论量级一致：必要，但远不充分。

**(b) 非均匀入流全局放大 a -> a * kappa**（`BEMTConfig.nonuniform_inflow_ratio`，默认 1.0 = 关闭）。
用 Caradonna-Tung **三个独立总距点**做最小二乘标定：

| kappa | 5° 误差 | 8° 误差 | 12° 误差 | max 绝对误差 |
| :-- | :-- | :-- | :-- | :-- |
| 1.00 | +45.1% | +27.1% | +22.7% | 45.1% |
| 1.20 | +19.0% | +7.2% | +5.7% | 19.0% |
| 1.33 | +4.7% | -4.0% | -4.1% | **4.7%** |
| 1.45 | -5.7% | -12.4% | -11.6% | 12.4% |

单个常数同时压住跨 4 倍推力范围的三个点，说明它不是过拟合；**但仍被否决作为默认值**：

1. 同一系数下，基准设计工况的两阶段寻优**收敛失败**——推力残差 27.4%，弦长顶到
   `chord_max_over_R = 0.26R`、转速压到 `tip_mach_min` 附近仍打不到 80 N。它在低实度
   试验件上"对"，在高实度设计问题上"错"：全局乘子不可能描述本质上是**叶尖局部**的
   有限桨叶效应。为刷过一条红线而让工具在自身主要适用域上失效，不可接受。
2. 文献侧不存在这个量级的低阶修正。DADM（Aerospace 10(6):494）把 Prandtl F 换成由附着涡
   环量与离散尾涡丝直接算出的诱导速度、且**只进有效迎角**；该文并指出经典 ADM+Prandtl 的
   病症正是"在叶尖高估法向力，总距越小越明显"——与本实现误差随总距增大而收窄
   （5° +45% -> 12° +23%）的分布同形。而经典的 typical-vs-infinity 因子
   1 + 3/(8*pi*lambda) 属前飞/风机文献，在悬停下发散且**符号相反**（叶面诱导低于远尾迹值），
   不能当悬停修正用。

结论：<=5% 需要实测尾迹几何或自由涡尾迹级模型。按 §1 的方法分级表，那一档的定位是
"局部微调与近场涡干扰校核"（单次 1-10 s），与 D-07 给主循环的 3 分钟门禁互斥，
不能同时充当优化内核与验收内核。`nonuniform_inflow_ratio` 作为**显式校核开关**保留、
默认关闭；标定表就在上面，启用者自行判断外推风险。

### 4.2 FM ≥ 0.68 硬门禁（D-06）

基准工况 $T=80$ N、$R=0.75$ m、$t_{te}=2.5$ mm 的两阶段寻优实测 FM 落在
$0.666\sim0.680$ 区间（随随机种子与积分稠密度在 ±1% 内摆动）。
按 D-06，0.68 是硬门禁、$0.72\sim0.79$ 是理想区间；本实现稳定处于**门禁线以下 0.4% 以内**。
主要压制项是 D-01 强制的基底阻力：$t_{te}/c\approx0.023$ 时
$\Delta C_{d,\text{base}}\approx0.014$，约为剖面型阻的一倍，故钝尾缘的物理代价直接吃掉 FM 余量。

### 4.3 优化器可达的构型形态

基准工况最优解的展向工作迎角约 $9^\circ\sim13^\circ$（越过 $c_{l,\max}$ 附近），
叶尖马赫数被压到 $0.16$ 左右（远低于 0.65 上限）。这符合"定推力下 FM 极大化"的数学解：
型阻功率 $\propto\Omega^3$、诱导功率对转速不敏感，故最优偏低速大弦长。
计划的目标函数只惩罚自相交/弦长越界/扭转梯度，**不含失速裕度约束**，
因此该形态是给定目标函数的合法最优，而非实现缺陷；用于实际选型时应另行加入
失速裕度与刚度约束（见 `docs/manufacturing.md` 的强度校核表）。

---

## 5 符号—代码对照

| 论文符号 | 代码 | 位置 |
| :-- | :-- | :-- |
| $\zeta_u,\zeta_l,\psi$ | `generate_flatback_cst` 返回点阵 | `geometry/cst.py` |
| $\Delta\bar z_{te}$ | `dz_te` | `geometry/cst.py`, `aero/polar.py` |
| $t_{te}$（绝对） | `RotorInputs.te_thickness` | `config.py` |
| $F,\,a,\,\phi$ | `prandtl_loss_factor`, `induced_inflow_factor` | `aero/bemt.py` |
| $c_l,c_d,c_m$ | `AeroPolarEvaluator.evaluate -> Polar` | `aero/polar.py` |
| $\Delta C_{d,\text{base}}$ | `compute_base_drag_hoerner` | `aero/base_drag.py` |
| $X,\ J(X)$ | `decode_design`, `HoverObjective` | `optimizer/objective.py` |
| $C_T,C_P,FM$ | `HoverSolution.ct/cp/fm` | `aero/bemt.py` |
| $\bar r_i$ | `cosine_span_positions` | `geometry/discretization.py` |
