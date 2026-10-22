# SunnyNet 架构说明

> 本文档给出 SunnyNet 的全部数学定义与设计依据。
> SunnyNet 为独立原创架构，**不基于** Transformer / RNN(LSTM/GRU) / Mamba(SSM) / CNN
> 等任何既有神经网络范式，也不复用 ReLU/GELU/SiLU、LayerNorm/RMSNorm/InstanceNorm、
> Xavier/Glorot/He/MSRA 初始化等既有方案。

## 0. 设计出发点

在 CPU 上做预训练，瓶颈通常不在浮点算力，而在：

1. **内存带宽**：每个训练步都要把激活、梯度、优化器状态在 DRAM 与缓存之间搬运；
2. **缓存命中率**：矩阵维度一旦超过 L2/L3，命中率断崖下降，有效算力被访存拖住；
3. **序列串行度**：逐 token 递推使序列维度无法并行，CPU 多核吃不满。

因此 SunnyNet 的全部设计围绕三条硬约束展开：

| 约束 | 对应设计 |
|---|---|
| 序列复杂度必须严格 O(N) | 滚动状态 + 一阶线性递推，无任何 token 两两交互 |
| 不保存整条序列的中间激活 | 状态维度固定，跨块只传状态；反向按需重算 |
| 矩阵维度压到中小规模、规避巨型稠密乘法 | 低秩分解 + 深度可分卷积 + 三对角循环滤波 |

同时**不牺牲通用语言建模的长距离能力**，这一点的实现方式见 §1.3。

---

## 1. 基础状态更新单元（`SunnyBlock`）

### 1.1 符号

| 符号 | 含义 | 形状 |
|---|---|---|
| $x_t$ | 第 $t$ 步输入向量 | $\mathbb{R}^{d}$ |
| $s_t$ | 第 $t$ 步全局滚动状态 | $\mathbb{R}^{d}$ |
| $z_t$ | 拼接后的中间量 | $\mathbb{R}^{2d}$ |
| $u_t$ | 候选状态 | $\mathbb{R}^{d}$ |
| $q_t$ | 低秩投影输出 | $\mathbb{R}^{2d}$ |
| $S_t$ | 扫描得到的状态序列 | $\mathbb{R}^{d}$ |
| $d$ | 主干维度；$d_i$ 为非线性支路维度；$r$ 为投影秩 | — |

### 1.2 前向计算

**(a) 状态循环滤波.** 状态先经一个 3 抽头循环滤波做轻量通道混合：

$$
\tilde{s}_t \;=\; \tau_0\,\mathrm{roll}(s_t, +1) \;+\; \tau_1\,s_t \;+\; \tau_2\,\mathrm{roll}(s_t, -1),
\qquad \tau \in \mathbb{R}^3 \ \text{可学习}
$$

它等价于用三对角循环矩阵右乘 $s_t$，但只做位移与逐元素乘加：**没有矩阵乘法，也没有额外内存**。

**(b) 拼接与低秩投影.** 归一化后与滤波状态在特征维拼接（**无加法旁路**），再经低秩分解投影：

$$
z_t = \big[\,\mathcal{N}(x_t) \,;\, \tilde{s}_t\,\big] \in \mathbb{R}^{2d},
\qquad
q_t = U\big(V(z_t)\big),\quad V \in \mathbb{R}^{r \times 2d},\ U \in \mathbb{R}^{2d \times r}
$$

参数量与浮点开销从 $O(d^2)$ 降到 $O(dr)$。当 $r \ll d$ 时，这一步带来的访存压力远低于同容量的稠密矩阵。

**(c) 候选状态（有界）.** 将 $q_t$ 拆成 $(u^{\text{raw}}_t, g_t)$。把 $g_t$ 按每 `group` 个通道
分块，组内做**成对竞争门控**：

$$
w_i \;=\; \sigma\!\big(g_i - g_{\text{next}(i)}\big)\cdot\sigma\!\big(g_i - g_{\text{next2}(i)}\big)
\in (0,1),
\qquad
u_t = u^{\text{raw}}_t \odot w_t
$$

$w_i < 1$ 使 $u_t$ 有界，这是后续"凸组合"保证状态不发散的前提。
组内竞争表达了与 softmax 相同的"多通道争夺同一份资源"语义，但 **Jacobian 量级是 $O(1)$ 而非 $O(1/d)$**。

> **这一处是实测驱动的修正**：最初版本用 $\mathrm{softmax}(g_t)$ 做通道竞争，
> 实测发现梯度范数随深度按约 $0.018$ 倍衰减，16 层后**精确归零**——
> softmax 的 Jacobian 含 $1/d$ 因子，在 128 个通道上每层都打一次折，深堆叠必然梯度死绝。
> 换成组内成对竞争后，$J$ 量级恢复为 $O(1)$。记录于此以说明设计约束的来源。

**(d) 非线性支路.** 只服务于输出，不进入状态递推，因此不破坏扫描的线性性：

$$
y_t \;=\; \mathcal{N}\!\big(\,\mathcal{A}\big(W_z u_t\big)\big)
$$

**(e) 数据驱动保护门.** 逐通道保留率由输入自身决定：

$$
\delta_i \;=\; \frac{1}{H_i},\qquad
p_{t,i} \;=\; 1 - \delta_i\,\big(1 - \sigma(g_{t,i})\big),\qquad
p_{t,i} \in [1-\delta_i,\; 1]
$$

$H_i$ 是通道 $i$ 的**记忆跨度**（单位：token），在对数刻度上均匀取值。
$\theta_i$ 是**可学习**参数（初值使 $H$ 对数均匀），同行（Mamba/RWKV/H3）的衰减
均为可学，冻结成 buffer 会限制模型自选记忆尺度：

$$
H_i = \exp\!\Big(\ln H_{\min} + s(\theta_i)\,\big(\ln H_{\max} - \ln H_{\min}\big)\Big),\qquad s=\sigma(\cdot)
$$

$\theta$ 初值由 `horizon_logit` 生成（通道中点 $s=(i+0.5)/d$ 后取 logit，避开 $\pm\infty$）。

默认 $H_{\min} = 2$、$H_{\max} = 8192$（20M 原型）；放大到 0.5B 时 $H_{\max} = 32768$。

**(f) 状态递推（凸组合）.**

$$
s_{t} \;=\; p_{t} \odot s_{t-1} \;+\; \big(1 - p_{t}\big) \odot u_{t}
$$

由于两个系数非负且和为 1，有**逐元素凸组合**

$$
|s_{t,i}| \;\le\; \max\big(|s_{t-1,i}|,\, |u_{t,i}|\big)
\;\Longrightarrow\;
\|s_t\|_\infty \le \max\big(\|s_0\|_\infty,\ \max_{k\le t}\|u_k\|_\infty\big)
$$

**状态天然有界，不会出现加法累积发散。** 这是本架构与 LSTM 类（$c_t = f \odot c_{t-1} + i \odot \tilde{c}$，
其中新信息是**加法**注入、无上界）在数学结构上的根本差异。

**(g) 输出.**

$$
S_{1:T} = \mathrm{Scan}\big(u_{1:T},\, p_{1:T},\, s_0\big)
\qquad
o_t = W_c\, y_t + \beta\, S_t + W_s\,\mathrm{circ}(S_{\text{prev}}),\qquad \beta \ \text{可学习}
$$

$S_t$ 携带历史信息，$y_t$ 是纯局部非线性变换；块输出再与输入残差相加
（$x^{(l+1)}_t = o^{(l)}_t + x^{(l)}_t$，见 §5）。块末状态接力：$s_{t+1}^{\text{next}} = S_T$。

### 1.3 长距离依赖如何实现

保留率 $p_{t,i}$ 的**下界**是 $1 - \delta_i$，故通道 $i$ 的单步遗忘量最多 $\delta_i = 1/H_i$。
在无门控保护时，通道 $i$ 的记忆时间常数为

$$
\tau_i \;=\; \frac{1}{1-p_{t,i}} \;\le\; H_i
$$

$H_i$ 在对数刻度遍历 $[2,\ 8192]$，意味着**同一层内同时存在**从"只记 2 个 token"
到"记忆跨 8192 个 token"的全部时间尺度。这直接对应通用语言建模的需求：

- 语法/局部搭配 → 由 $H_i$ 小的通道 + 深度可分卷积承担；
- 段落级、跨代码块、古文训诂中的远距离指代 → 由 $H_i$ 大的通道承担。

由于 $p$ 是**逐元素**的（而非逐通道共享一个标量），且 $\sigma(g)$ 由当前输入决定，
模型可以在不同上下文下动态决定"这一刻该记多久"。

### 1.4 分块并行扫描

递推 $s_t = p_t s_{t-1} + (1-p_t)u_t$ 是仿射递推，块内存在**闭式解**。令

$$
P_j = \sum_{k \le j} \ln p_k,\qquad g_k = (1-p_k)\,u_k
$$

则

$$
\boxed{\;s_j \;=\; \exp(P_j)\,\Big(s_{\text{in}} \;+\; \sum_{k \le j} g_k \exp(-P_k)\Big)\;}
\qquad (\star)
$$

两次 inclusive `cumsum` 即可得到整块**全部**状态，块间只需串行 $T/\text{chunk}$ 次。
$(\star)$ 与逐步递推**代数等价**，不是近似。

**块间并行化。** 关键在于 $(\star)$ 对块首状态 $s_{\text{in}}$ 是**严格仿射**的，且

$$
s_j\big|_{j=T} = p_T\, s_{\text{in}} + (1-p_T)\,u_T
\quad\Longrightarrow\quad
s_{\text{in}} \;=\; \frac{S_T - (1-p_T)u_T}{p_T}
$$

即**块首状态可由块末状态精确反解**。因此可以先让所有块并行跑一遍，再逐块回填块首状态，
把序列维度的串行深度从 $O(T)$ 降到 $O(T/\text{chunk})$，与 Flash-Linear-Attention 类
"块内并行、块间串行"的思路一致，但这里的反解是**精确恒等式**，无条件成立。
实现见 `sunnynet/layers.py` 与 `tests/test_architecture.py::test_block_chunk_invariance_*`。

### 1.5 可逆计算

$s_t$ 的递推可逆：由 $(p_t, u_t, s_t)$ 可直接解出 $s_{t-1} = (s_t - (1-p_t)u_t)/p_t$。
因此反向传播时**无需保存**中间状态张量 $S_{1:T}$，可在反向途中用 $(p_t, u_t)$ 重建。
$(p_t, u_t)$ 由 $x_t$ 与块首状态决定，代价是一次局部重算，换来的是 $T \times d$ 量级的激活显存节省。

---

## 2. 原创非线性激活：SoftWedge

$$
\boxed{\;\mathcal{A}(z) \;=\; z \;+\; a \cdot s(z),\qquad s(z) = \tfrac12 + \tfrac12\cdot\frac{z}{1+|z|},\qquad a \in (0,1)\;}
$$

**导数有双侧严格界**（这是设计硬目标）：

$$
\frac{\mathrm{d}s}{\mathrm{d}z} = \frac{1}{2\,(1+|z|)^2} \in \Big(0,\ \tfrac12\Big]
\quad\Longrightarrow\quad
\mathcal{A}'(z) = 1 + a\,s'(z) \;\in\; \Big(1,\ 1 + \tfrac{a}{2}\Big]
$$

- **有正下界 1**：任何尺度下梯度都不会消失，不存在"梯度断崖"；
- **有有限上界 $1+a/2$**：$\mathcal{A}' \le 1.5$，配合 §4 的谱锚定初始化即可保证多层乘积增益有界；
- **渐近线性**：$z \to +\infty$ 时 $\mathcal{A}(z) \to z+a$，$z \to -\infty$ 时 $\mathcal{A}(z) \to z$。
  大值域上是恒等映射，长程信息可无损穿越；
- **无指数/三角运算**：$s(z)$ 只需加法、取绝对值、一次除法，CPU 上比 GELU/SiLU 便宜。

注意这与 GELU（$\Phi(z)$ 高斯 CDF 权重）、SiLU（$z\sigma(z)$）、Mish 形状相近但
**函数形式与导数行为完全不同**：上述三个在 $z \to -\infty$ 时导数趋于 0，
而 $\mathcal{A}'$ 恒 $\ge 1$。

实测（`tests/test_architecture.py::test_softwedge_gradient_bounds_analytic`）：
在 $z \in [-10^6, 10^6]$ 上实测导数值域 $[1.000067,\ 1.249998]$，与解析界 $(1, 1.25]$ 吻合。

---

## 3. 原创归一化：CenteredScaleNorm

$$
\mathcal{N}(x) \;=\; \gamma \odot \frac{x - \mu(x) + c}{\sqrt{\sigma^2(x) + s_{\min}^2 + \epsilon}}
$$

其中 $\mu(x)$、$\sigma^2(x)$ 为最后一维的均值与方差（有偏估计），
$\gamma, c \in \mathbb{R}^{d}$ 可学习，$s_{\min} = 0.02$。

两个自有设计点：

1. **$c$ 是逐维可学习中心，不是统计量。** 对即将被反复读写**状态**而言，
   把中心当作可训练参数更合理——它表达"该通道的零点是哪"。
2. **分母设平滑下界 $s_{\min}$。** 当输入近似常量（$\sigma^2 \to 0$）时，
   既有方案会放大 $\sqrt{\sigma^2+\epsilon}^{-1} \sim 1/\sqrt{\epsilon}$ 倍，把浮点噪声炸成有效信号；
   本设计使最大放大倍数被钳在 $1/s_{\min} = 50$ 倍。

**$s_{\min}$ 的取值是实测驱动的第三次修正（旧值 0.25 是错的）**：$s_{\min}$ 必须
**低于实际激活尺度**，否则归一化退化为常数缩放。实测 `small` 预设各层 norm 输入
std 低至 $0.06$，嵌入 std 仅 $0.02$，而旧默认 $1/s_{\min} = 4$ 的钳制门槛（$0.25$）
高于这些值 —— 于是所有输入事实上都是"近常量分支"，分母恒为 $0.25$，
输出只是输入的 $4\times$ 常数缩放，**归一化从未真正发生**。
后果是深层通道尺度漂移无法被抑制，实测验证 ppl 因此在 600 步内劣化约 26%
（详见实验报告 §12）。定稿取 $s_{\min} = 0.02$——扫描显示 ppl 随 $s_{\min}$
单调下降，平台出现在 $[0.01, 0.025]$，取平台内值（钳制到 $50\times$ 只对
$\mathrm{std} < 0.02$ 的真常量通道生效）。

实测（`test_csn_constant_input_stability`）：常量输入 $\mathbf{3.14159}$ 输出幅值 $< 10^{-3}$；
`test_csn_unit_variance`：$\mathrm{std}=7$ 的输入被归一到 std $\approx 1$。
对照 LayerNorm：近常量输入（$5.0+10^{-7}$ 噪声）下 CSN 输出 std $=1.0\text{e-}6$、
LN 为 $1.6\text{e-}5$，**CSN 抑制噪声放大强 15 倍**（§12.4）。

---

## 4. 原创初始化：谱锚定（Spectral Anchoring）

不使用任何基于 fan-in/fan-out 反推方差分布的方案（Xavier/He/MSRA 皆属此类）。
本方案**直接控制权重矩阵的奇异值谱**：

1. 取高斯矩阵 $G \in \mathbb{R}^{n\times n}$，做 QR 分解 $G = QR$，并用 $\mathrm{diag}(R)$ 的符号修正，得到正交基 $Q$；
2. 取 $Q$ 的一个正交切片 $Q_{[:m,:k]}$，按目标谱半径缩放：$W = \rho \cdot Q_{[:m,:k]}$，则 $\sigma_{\max}(W) = \rho$；
3. 谱半径取**常数** $\rho = 1.0$，不做层深衰减（$\beta = 0$）。

**依据**：正交切片的条件数为 1，前向增益谱平坦（各方向等权）。
$\rho = 1.0$ 的选取由实测标定——见下表。

**为什么不做层深衰减（这是第二个实测驱动的修正）**：

最初版本按 $\rho_l = \rho_0 (l+1)^{-\beta}$（$\beta = 0.15$）让深层权重更小，理由是
"抵消深堆叠的乘积增益"。实测证明符号设反了，它直接制造了梯度消失：

| $\rho_0$ | $\beta$ | $L{=}8$ 嵌入梯度 | $L{=}16$ | $L{=}24$ | 8→16→24 比值 |
|---|---|---|---|---|---|
| 1.15 | 0.00 | 1.35e+0 | 1.73e+0 | 2.50e+0 | 1.28 / 1.44（稳） |
| 1.15 | 0.10 | 3.44e-1 | 3.11e-2 | 5.17e-4 | 0.090 / 0.017（暴死） |

进一步扫描 $\rho_0$（$\beta = 0$）：

| $\rho_0$ | $L{=}4$ | $L{=}8$ | $L{=}16$ | $L{=}32$ | 结论 |
|---|---|---|---|---|---|
| 0.50 | 2.94e-6 | 6.42e-11 | 5.46e-20 | 0 | 强烈消失 |
| 0.85 | 1.42e-2 | 2.16e-3 | 4.22e-5 | 1.76e-7 | 仍消失 |
| **1.00** | **1.86e-1** | **3.38e-1** | **5.66e-1** | **6.58e-1** | **跨深度稳定** |
| 1.15 | 9.81e-1 | 1.35e+0 | 1.73e+0 | 4.17e+0 | 轻微增长 |

因此定稿 $\rho_0 = 1.0$、$\beta = 0$：单块 Jacobian 增益被标定在 1 附近，
层数增加时梯度不膨胀也不消失。实测 $L = 2 \to 32$ 时嵌入梯度
$1.46\text{e-}1 \to 6.58\text{e-}1$，首块与末块权重的梯度范数同量级（5.1e-2 vs 5.1e-2），
不存在"只有浅层在学"的现象。

> **修正说明**：上两表在旧 CSN（$s_{\min}=0.25$，未真正归一化）下测得。
> CSN 修正（§3）后重测，崩塌边界整体下移：$\rho_0=0.5$ 不再崩塌（24 层/4 层
> 梯度比 $1.01$），$\beta=0.15$ 也不再单独触发崩塌（$1.45$）；崩塌只在
> $\rho_0 < 0.5$ 一侧出现。$\rho_0=1$、$\beta=0$ 的定稿结论不变。

实测：初始化后首步 loss $= 6.2383$，$\ln V = \ln 512 = 6.238$ —— 初始分布精确接近均匀，尺度无误。
见 `test_init_loss_near_uniform`、`test_spectral_init_singular_values`。

---

## 5. 跨层信息传递（状态接力 + 残差）

层间传递有两条通路：

1. **状态接力（主通路）**：第 $l$ 块的块末状态 $S^{(l)}_T$ 直接作为第 $l+1$ 块的块首状态 $s^{(l+1)}_0$。
   状态的尺度由 §1.2(f) 的凸组合性质锁定（$\|s\|_\infty$ 不增），因此逐层传递**不会放大**。
2. **特征流（辅通路）**：$o^{(l)}_t$ 作为第 $l+1$ 块的输入 $x_t$，经 $\mathcal{N}$ 与低秩投影再入，
   并在块输出处**与输入残差相加**：$x^{(l+1)}_t = o^{(l)}_t + x^{(l)}_t$。

**残差是第四轮修正加入的（旧设计明确排除残差，实测证明该排除是错的）**。原设计
让 $o_t$ 完全取代 $x_t$，层间只有状态接力一条严格通路。实测（七架构同协议 600 步、
多子种）表明这一"纯粹性"是**最大的单项质量损失**：

| 配置 | ppl（seed 1234 / 2024 / 3407） |
|---|---|
| 无残差（旧默认） | 93.82 |
| **有残差（现默认）** | **67.40 / 70.72** |
| 对照：LayerNorm + 残差 | 54.90 / 54.58 / 56.58 |
| 参照：H3-2 | 53.93 |

残差单项带来 **−28.2% ~ −32.5%** 的相对 ppl 改善，是所有试过改动里最大且可复现的
（其余：可学 decay 无收益、state_norm 叠加无收益、$h_{\max}$ 调整与非线性增强均无效）。
依据：残差提供一条**不经过状态接力**的恒等直连，使深层不必仅靠状态总线承载全部信息，
也让"欠缩放导致的深度梯度崩塌"不再出现（`test_residual_removes_depth_collapse`：
$\rho_0=0.3$ 时 24 层/4 层梯度比，无残差 $1.1\text{e-}4$，有残差 $1.48$）。

原"无残差也能靠谱锚定保持梯度"的论述只对旧 CSN（未真正归一化，见 §3）成立；CSN
修正后该保障消失，残差从"可选"变成"必要"。

| 手段 | 作用 |
|---|---|
| $\mathcal{A}' \in (1, 1.5]$ | 非线性永不给梯度打折 |
| $\mathcal{N}$ 的 $s_{\min}$ 下界 | 消除"方差趋于 0 → 梯度爆炸"，同时保持输出尺度恒定 |
| 谱锚定 $\rho=1$、$\beta=0$ | 单块 Jacobian 增益标定在 1 附近，不做层深衰减（§4） |
| **残差直连** | 为深层提供不经过状态接力的恒等通路，兜住深度梯度 |

实测（无残差、仅状态总线）：深度 $4 \to 24$ 下嵌入梯度比值落在 $[0.1, 10]$
要求 $\rho_0 \ge 1.0$，欠缩放（$\rho_0=0.3$）会崩塌到 $1.1\text{e-}4$ ——
`test_spectral_scale_decides_depth_stability` 守住这条通路。

---

## 6. 输出映射头（LM Head）

$$
\mathrm{logits}_t = W_{\text{head}}\;\mathcal{N}\!\big(o_t\big), \qquad
W_{\text{head}} \in \mathbb{R}^{V \times d}
$$

$W_{\text{head}}$ 与嵌入矩阵**权重绑定**（tie），预训练目标为标准自回归
下一个 token 交叉熵：

$$
\mathcal{L} = -\frac{1}{NT}\sum_{n=1}^{N}\sum_{t=1}^{T} \ln \frac{\exp(\mathrm{logits}_{n,t,y_{n,t}})}{\sum_{v}\exp(\mathrm{logits}_{n,t,v})}
$$

支持续接 SFT 指令微调（同头、同目标，数据换成指令-回复对）。

---

## 7. 输入层

BPE 子词分词器，从零实现（`sunnynet/tokenizer.py`），不依赖 `tokenizers` / `sentencepiece`。

- **预分词**：CJK 连续串 / 拉丁词 / 数字 / 符号 四类分开，
  使中文按短语合并、英文按词合并，同时兼容代码符号；
- **合并**：标准 BPE 最大频次优先，堆优化的增量更新；
- **词表**：4 个特殊符号 + 高频字符 + 合并项。

实测往返一致性：中文古文、英文文学、Python 代码、中英混排 **6/6 全部无损**。

> 输入层不属于原创网络核心单元，其存在不违反架构原创性要求。

---

## 8. 复杂度与 CPU 适配

设序列长度 $T$、主干维度 $d$、秩 $r$、卷积抽头 $k$。

| 路径 | 时间复杂度 | 空间复杂度 |
|---|---|---|
| 状态扫描（$\star$） | $O(T r d)$ | $O(T r + \text{chunk}\cdot d)$ |
| 循环滤波 | $O(T d)$ | $O(1)$ |
| 深度可分卷积 | $O(T d k)$ | $O(1)$ |
| 非线性支路 | $O(T d d_i)$ | $O(T d_i)$ |
| **合计** | $\mathbf{O(T)}$ 严格线性 | $O(T)$（反向可降至 $O(\text{chunk})$） |

**无任何 token 两两交互项**，故不存在 $O(T^2)$ 结构。

CPU 适配要点：

- 主干矩阵维度取中小规模（20M 原型：$d=512$、$r=320$、$d_i=896$、$L=10$，实测 20,955,698 参数），使单次读写的工作集落入 L2/L3；
- 低秩分解使最大矩阵从 $2d \times 2d$ 降为 $r \times 2d$ 与 $2d \times r$；
- 循环滤波与深度可分卷积**零矩阵乘法**；
- 原生 FP32，不依赖 FP16/BF16（CPU 对半精度支持弱，软件模拟反而更慢）。

### 8.1 实测优化尝试（含被否决项）

| 尝试 | 结论 | 依据 |
|---|---|---|
| `logspace_scan` 跳过 `p.amin()`（改由 `p_min_hint` 传入） | **采纳** | 单步 0.596ms→0.043ms，取值不变 |
| 扫描块长调整（64→128/256/512/1024） | **无收益** | T=512…4096 上差异在 ±10% 噪声内 |
| `gated_scan` 单块时免 `torch.cat(outs)` | **否决** | A/B 增益 ±5% 且方向不稳定，无真实收益 |
| 竞争门控 roll→切片改写 | **否决** | 实测更慢（4.91ms vs 1.95ms），保留 `roll` |
| `CenteredScaleNorm` 用 `var_mean` 融合 | **否决** | 前向快 46% 但反向更慢，且数值差 4.8e-7 超出 1e-10 不变量 |
| 状态反馈 `_feedback` 免 `cat` 改写 | **否决** | 数值不等价（偏差 1.8） |
| `torch.compile` 算子融合 | **环境受限** | MSVC 缺 Windows SDK 头文件；Inductor 的 Windows 分支要求 MSVC 风格 CLI，不认 gcc |
| 手写 AVX2 内核（w64devkit GCC 16.2.0） | **采纳** | scan 算子级比 torch 快 4.3–22×；融合内核训练步 1.10–1.17×（报告 §21/§22） |
| 多线程（1→8 线程） | **有效** | T=4096：322ms→147ms（2.2×），并发算子调度可并行 |

**关键诊断**：算子剖析显示瓶颈是**调度开销**而非 FLOP——单次前向
`aten::mm` 31%、`aten::mul` 13%、`aten::add` 11%、`aten::var` 10%，
另有 **128 次 `cumsum`/次**（来自分块扫描）。没有单一支配算子，
故单点改写难以奏效；有效路径只有减少算子总数（融合）。
`torch.compile` 在本机不可用，改由手写 AVX2 内核绕开 Inductor 实现融合。

**测量纪律（重要）**：耗时测量必须**每格独立进程**。同一模型在"已构建过多个模型"
的进程中测量，耗时可达干净进程的 **2 倍以上**（T=4096：325ms vs 776–822ms），
且放大倍数随架构而变。未隔离时的架构间耗时比全部失真，详见实验报告 §2.4。

---

## 9. 与既有范式的差异对照

| 维度 | Transformer | RNN / LSTM / GRU | SSM / Mamba | **SunnyNet** |
|---|---|---|---|---|
| 序列复杂度 | $O(T^2)$ | $O(T)$ | $O(T)$ | $O(T)$ |
| 状态更新 | 无状态 | $c_t = f\odot c_{t-1} + i\odot\tilde c$（加法，无界） | $h_t = \bar A h_{t-1} + \bar B x_t$（输入无关矩阵） | $s_t = p_t\odot s_{t-1} + (1-p_t)\odot u_t$（**凸组合，有界**） |
| 保留率来源 | — | Sigmoid 门（非凸组合） | 输入无关的离散化步长 | 输入自门控，上界钳到 1 |
| 时间尺度 | 无 | 单一尺度 | 混合（HiPPO 等） | 对数均匀 $[2, 8192]$ 逐元素多尺度 |
| 层间连接 | 残差相加 | 残差 / 堆叠 | 残差 | **状态接力 + 特征残差流** |
| 激活 | GELU/SiLU，负端梯度 $\to 0$ | tanh/sigmoid | SiLU | SoftWedge，导数恒 $\in (1, 1.5]$ |
| 归一化 | LayerNorm/RMSNorm | LayerNorm | RMSNorm | CenteredScaleNorm（平滑下界尺度） |
| 初始化 | Xavier/He | Xavier | 特制 | 谱锚定（直接控奇异值谱） |

需要明确的是：**SunnyNet 的状态递推形式与线性注意力类工作有数学亲缘性**
（都是一阶线性递推），但门控构造（凸组合 + 对数均匀多尺度 + 逐元素保护门）、
状态接力与特征残差的双通路层间传递、激活与归一化均为自行设计，
且未依赖任何既有实现的代码或结构。

---

## 10. 状态传递示意

```
        输入 token 流  x_1  x_2  ...  x_T
                        │    │         │
   ┌────────────────────┴────┴─────────┴──────────────────┐
   │  Block 1                                              │
   │   N(x) ─┐                                             │
   │         ├─ concat ─► 低秩投影 ─► u_t ──┐              │
   │   circ(s)┘        │                    ▼              │
   │                   └─► σ(g) ──────► p_t ●── 凸组合递推    │
   │                                          ╲            │
   │  s_0 ─────────────────────────────────────► S_t        │
   │                                   ┌────────┘           │
   │                          W_c·A(W_z u) + β·S_t = o_t    │
   └───────────────────────────┬───────────────────────────┘
                               │
            块末状态 S_T ──────┼──────► 作为下一块的 s_0   (状态接力)
                    o_1..o_T ─┘─► +x ──► 作为下一块的输入  (特征残差流)
                               │
   ┌───────────────────────────┴───────────────────────────┐
   │  Block 2 … Block L   (同上, 谱锚定 ρ = 1.0, β = 0)      │
   └───────────────────────────┬───────────────────────────┘
                               │
                     N(o) ─► W_head (与 Embedding 绑定) ─► logits
```

块内并行扫描：

```
       块 1        块 2        块 3              ← 块内 chunk 全并行
     [并行跑]    [并行跑]    [并行跑]
         │           │           │
         └──► 用 S_T 反解块首状态, 逐块回填 (仅 T/chunk 次串行) ◄──┘
```

---

## 11. 已知局限与风险

1. **长距离依赖**：滚动状态的信息容量受限于固定维度 $d$，理论上界低于全注意力
   （$O(T)$ 可寻址）。$H_{\max} = 8192$ 只保证"不被遗忘"，不保证"被检索"。
   这是本架构最主要的能力风险，评估脚本 `train/evaluate.py` 的
   `long_range_recall` 探针专门量化这一点。
2. **伪收敛**：loss/ppl 下降不等于具备推理与指令跟随能力。
   评估必须包含独立测试集与分域困惑度，不能只看训练 loss。
3. **收益幅度**：CPU 训练提速幅度需实测，论文式宣称不成立。见实验报告实测数据。
---

# SunnyNet Architecture

> This document gives the full mathematical definition and design rationale of SunnyNet.
> SunnyNet is an independent, original architecture: it is **not based on** any existing neural
> network paradigm — Transformer / RNN (LSTM/GRU) / Mamba (SSM) / CNN — and it does not reuse
> existing schemes such as ReLU/GELU/SiLU, LayerNorm/RMSNorm/InstanceNorm, or
> Xavier/Glorot/He/MSRA initialization.

## 0. Design starting point

For CPU pretraining, the bottleneck is usually not floating-point throughput but:

1. **Memory bandwidth**: every training step moves activations, gradients, and optimizer state
   between DRAM and the caches;
2. **Cache hit rate**: once the matrix dimension exceeds L2/L3, the hit rate drops off a
   cliff and effective compute is dragged down by memory access;
3. **Sequence seriality**: the per-token recurrence makes the sequence dimension
   non-parallelizable, so the CPU's multiple cores stay under-utilized.

SunnyNet's entire design therefore revolves around three hard constraints:

| Constraint | Design that answers it |
|---|---|
| Sequence complexity strictly O(N) | rolling state + first-order linear recurrence, no token-pair interactions at all |
| No materialized activations for the whole sequence | state dimension fixed, only the state crosses chunk boundaries; backward recomputes on demand |
| Keep matrix dimensions small-to-mid scale, avoid giant dense multiplies | low-rank decomposition + depthwise-separable convolutions + tridiagonal circulant filter |

At the same time it **does not sacrifice the long-range capability** of general language
modeling; the way this is implemented is in §1.3.

---

## 1. Base state-update unit (`SunnyBlock`)

### 1.1 Notation

| Symbol | Meaning | Shape |
|---|---|---|
| $x_t$ | input vector at step $t$ | $\mathbb{R}^{d}$ |
| $s_t$ | global rolling state at step $t$ | $\mathbb{R}^{d}$ |
| $z_t$ | concatenated intermediate | $\mathbb{R}^{2d}$ |
| $u_t$ | candidate state | $\mathbb{R}^{d}$ |
| $q_t$ | low-rank projection output | $\mathbb{R}^{2d}$ |
| $S_t$ | state sequence obtained by the scan | $\mathbb{R}^{d}$ |
| $d$ | trunk dimension; $d_i$ is the nonlinear-branch dimension; $r$ is the projection rank | — |

### 1.2 Forward computation

**(a) State circulant filter.** The state first passes through a 3-tap circulant filter for
light channel mixing:

$$
\tilde{s}_t \;=\; \tau_0\,\mathrm{roll}(s_t, +1) \;+\; \tau_1\,s_t \;+\; \tau_2\,\mathrm{roll}(s_t, -1),
\qquad \tau \in \mathbb{R}^3 \ \text{learnable}
$$

It is equivalent to right-multiplying $s_t$ by a tridiagonal circulant matrix, but it only does
shifts and elementwise multiply-add: **no matrix multiply, no extra memory**.

**(b) Concatenate and low-rank projection.** After normalization it is concatenated with the
filtered state along the feature dimension (**no additive bypass**), then projected through a
low-rank factorization:

$$
z_t = \big[\,\mathcal{N}(x_t) \,;\, \tilde{s}_t\,\big] \in \mathbb{R}^{2d},
\qquad
q_t = U\big(V(z_t)\big),\quad V \in \mathbb{R}^{r \times 2d},\ U \in \mathbb{R}^{2d \times r}
$$

Parameter count and floating-point cost drop from $O(d^2)$ to $O(dr)$. When $r \ll d$, the
memory-access pressure this step adds is far below that of a dense matrix of the same capacity.

**(c) Candidate state (bounded).** Split $q_t$ into $(u^{\text{raw}}_t, g_t)$. Chunk $g_t$ by
`group` channels and do **pairwise competition gating** within each group:

$$
w_i \;=\; \sigma\!\big(g_i - g_{\text{next}(i)}\big)\cdot\sigma\!\big(g_i - g_{\text{next2}(i)}\big)
\in (0,1),
\qquad
u_t = u^{\text{raw}}_t \odot w_t
$$

$w_i < 1$ keeps $u_t$ bounded — this is the prerequisite for the later "convex combination" to
guarantee the state does not diverge.
The in-group competition expresses the same "many channels competing for one resource" semantics
as softmax, but **the Jacobian is $O(1)$ rather than $O(1/d)$**.

> **This is a measurement-driven correction**: the first version used $\mathrm{softmax}(g_t)$
> for channel competition; measurement found the gradient norm decayed by roughly a factor of
> $0.018$ per layer and **exactly hit zero** after 16 layers — the softmax Jacobian carries a
> $1/d$ factor that discounts every layer over 128 channels, so a deep stack inevitably kills the
> gradient.
> Switching to in-group pairwise competition restored the $J$ to $O(1)$ order. Recorded here to
> document the source of the design constraint.

**(d) Nonlinear branch.** It only serves the output and does not enter the state recurrence, so
it does not break the linearity of the scan:

$$
y_t \;=\; \mathcal{N}\!\big(\,\mathcal{A}\big(W_z u_t\big)\big)
$$

**(e) Data-driven protection gate.** The per-channel retention is decided by the input itself:

$$
\delta_i \;=\; \frac{1}{H_i},\qquad
p_{t,i} \;=\; 1 - \delta_i\,\big(1 - \sigma(g_{t,i})\big),\qquad
p_{t,i} \in [1-\delta_i,\; 1]
$$

$H_i$ is the **memory span** of channel $i$ (in tokens), taken uniformly on a log scale.
$\theta_i$ is a **learnable** parameter (initialized so that $H$ is log-uniform); the decay in
the peers (Mamba/RWKV/H3) is also learnable — freezing it into a buffer would limit the model's
ability to choose its own memory scale:

$$
H_i = \exp\!\Big(\ln H_{\min} + s(\theta_i)\,\big(\ln H_{\max} - \ln H_{\min}\big)\Big),\qquad s=\sigma(\cdot)
$$

The $\theta$ initialization comes from `horizon_logit` (channel midpoint $s=(i+0.5)/d$ then
logit, avoiding $\pm\infty$).

By default $H_{\min} = 2$, $H_{\max} = 8192$ (20M prototype); when scaling to 0.5B,
$H_{\max} = 32768$.

**(f) State recurrence (convex combination).**

$$
s_{t} \;=\; p_{t} \odot s_{t-1} \;+\; \big(1 - p_{t}\big) \odot u_{t}
$$

Because the two coefficients are non-negative and sum to 1, there is an **elementwise convex
combination**

$$
|s_{t,i}| \;\le\; \max\big(|s_{t-1,i}|,\, |u_{t,i}|\big)
\;\Longrightarrow\;
\|s_t\|_\infty \le \max\big(\|s_0\|_\infty,\ \max_{k\le t}\|u_k\|_\infty\big)
$$

**The state is bounded by construction and cannot accumulate additively to divergence.** This
is the fundamental structural difference from LSTM-like systems
($c_t = f \odot c_{t-1} + i \odot \tilde{c}$, where the new information is injected
**additively**, without an upper bound).

**(g) Output.**

$$
S_{1:T} = \mathrm{Scan}\big(u_{1:T},\, p_{1:T},\, s_0\big)
\qquad
o_t = W_c\, y_t + \beta\, S_t + W_s\,\mathrm{circ}(S_{\text{prev}}),\qquad \beta \ \text{learnable}
$$

$S_t$ carries history; $y_t$ is a pure local nonlinear transform; the block output is then
added to the input residual
($x^{(l+1)}_t = o^{(l)}_t + x^{(l)}_t$, see §5). End-of-block state relay:
$s_{t+1}^{\text{next}} = S_T$.
### 1.3 How long-range dependencies work

The **lower bound** of the retention rate $p_{t,i}$ is $1 - \delta_i$, so channel $i$'s
single-step forgetting is at most $\delta_i = 1/H_i$.
Without gate protection, channel $i$'s memory time constant is

$$
\tau_i \;=\; \frac{1}{1-p_{t,i}} \;\le\; H_i
$$

$H_i$ sweeps $[2,\ 8192]$ on a log scale, meaning that **within a single layer** every time
scale from "remember only 2 tokens" to "memory spanning 8192 tokens" is present at once.
This directly matches the needs of general language modeling:

- syntax / local collocation is handled by the small-$H_i$ channels + the depthwise-separable
  convolution;
- paragraph-level, cross-code-block, long-range anaphora in classical text is handled by the
  large-$H_i$ channels.

Since $p$ is **elementwise** (not one scalar shared across channels) and $\sigma(g)$ is decided
by the current input, the model can dynamically decide "how long to remember right now" in
different contexts.

### 1.4 Chunked parallel scan

The recurrence $s_t = p_t s_{t-1} + (1-p_t)u_t$ is an affine recurrence, so a **closed form**
exists within a block. Let

$$
P_j = \sum_{k \le j} \ln p_k,\qquad g_k = (1-p_k)\,u_k
$$

then

$$
\boxed{\;s_j \;=\; \exp(P_j)\,\Big(s_{\text{in}} \;+\; \sum_{k \le j} g_k \exp(-P_k)\Big)\;}
\qquad (\star)
$$

Two inclusive `cumsum` calls give the **entire** state sequence of the block; between blocks
only $T/\text{chunk}$ serial steps are needed.
$(\star)$ is **algebraically equivalent** to stepwise recurrence, not an approximation.

**Inter-block parallelization.** The key is that $(\star)$ is **strictly affine** in the
block-start state $s_{\text{in}}$, and

$$
s_j\big|_{j=T} = p_T\, s_{\text{in}} + (1-p_T)\,u_T
\quad\Longrightarrow\quad
s_{\text{in}} \;=\; \frac{S_T - (1-p_T)u_T}{p_T}
$$

i.e. **the block-start state can be back-solved exactly from the block-end state**.
So all blocks can run in parallel first, then each block's start state is back-filled
sequentially, dropping the serial depth of the sequence dimension from $O(T)$ to
$O(T/\text{chunk})$ — the same "parallel within a block, serial between blocks" idea as
Flash-Linear-Attention and similar work, except here the back-solve is an **exact identity**
that holds unconditionally.
Implementation in `sunnynet/layers.py` and
`tests/test_architecture.py::test_block_chunk_invariance_*`.

### 1.5 Reversible computation

The recurrence for $s_t$ is invertible: from $(p_t, u_t, s_t)$ one solves
$s_{t-1} = (s_t - (1-p_t)u_t)/p_t$ directly.
So at backward time the intermediate state tensor $S_{1:T}$ does **not** need to be saved; it
can be rebuilt on the fly from $(p_t, u_t)$.
$(p_t, u_t)$ are determined by $x_t$ and the block-start state, so the cost is one local
recomputation, buying $O(T \times d)$ activation memory savings.

---

## 2. Original nonlinear activation: SoftWedge

$$
\boxed{\;\mathcal{A}(z) \;=\; z \;+\; a \cdot s(z),\qquad s(z) = \tfrac12 + \tfrac12\cdot\frac{z}{1+|z|},\qquad a \in (0,1)\;}
$$

**The derivative has strict two-sided bounds** (a hard design goal):

$$
\frac{\mathrm{d}s}{\mathrm{d}z} = \frac{1}{2\,(1+|z|)^2} \in \Big(0,\ \tfrac12\Big]
\quad\Longrightarrow\quad
\mathcal{A}'(z) = 1 + a\,s'(z) \;\in\; \Big(1,\ 1 + \tfrac{a}{2}\Big]
$$

- **Positive lower bound of 1**: the gradient never vanishes at any scale; there is no
  "gradient cliff";
- **Finite upper bound $1+a/2$**: $\mathcal{A}' \le 1.5$; combined with the spectral-anchored
  initialization of §4 this keeps the multi-layer product gain bounded;
- **Asymptotically linear**: as $z \to +\infty$, $\mathcal{A}(z) \to z+a$; as
  $z \to -\infty$, $\mathcal{A}(z) \to z$.
  In the large-magnitude regime it is the identity map, so long-range information passes
  through losslessly;
- **No exp/trig operations**: $s(z)$ needs only addition, absolute value, and one division —
  cheaper on CPU than GELU/SiLU.

Note that this is similar in shape to GELU (Gaussian CDF weighting $\Phi(z)$), SiLU
($z\sigma(z)$), and Mish, but the **function form and derivative behavior are completely
different**: all three of those have derivative tending to 0 as $z \to -\infty$, whereas
$\mathcal{A}'$ is always $\ge 1$.

Measured
(`tests/test_architecture.py::test_softwedge_gradient_bounds_analytic`):
over $z \in [-10^6, 10^6]$ the measured derivative range is $[1.000067,\ 1.249998]$, matching
the analytic bound $(1, 1.25]$.

---

## 3. Original normalization: CenteredScaleNorm

$$
\mathcal{N}(x) \;=\; \gamma \odot \frac{x - \mu(x) + c}{\sqrt{\sigma^2(x) + s_{\min}^2 + \epsilon}}
$$

where $\mu(x)$, $\sigma^2(x)$ are the mean and variance over the last dimension
(biased estimates),
$\gamma, c \in \mathbb{R}^{d}$ are learnable, and $s_{\min} = 0.02$.

Two original design points:

1. **$c$ is a per-dimension learnable center, not a statistic.** For a **state** that will be
   read and written repeatedly, treating the center as a trainable parameter is more
   sensible — it encodes "where this channel's zero point is".
2. **A smooth lower bound $s_{\min}$ in the denominator.** When the input is nearly constant
   ($\sigma^2 \to 0$), the existing schemes amplify by
   $\sqrt{\sigma^2+\epsilon}^{-1} \sim 1/\sqrt{\epsilon}$, blowing float noise into a real
   signal; this design clamps the maximum amplification to $1/s_{\min} = 50$.

**The choice of $s_{\min}$ is the third measurement-driven correction (the old value 0.25 was
wrong)**: $s_{\min}$ must be **below the actual activation scale**, otherwise normalization
degenerates into a constant scaling. Measured: in the `small` preset, the norm-input std of
some layers is as low as $0.06$ and the embedding std is only $0.02$, while the old
$1/s_{\min} = 4$ clamp threshold ($0.25$) is above those values — so every input in fact
fell in the "near-constant branch", the denominator was always $0.25$, and the output was just
a $4\times$ constant scaling of the input: **normalization was never actually happening**.
The consequence is that deep-channel scale drift cannot be suppressed; measured, ppl degraded
by about 26% within 600 steps because of it
(see experiment report §12). The final value is $s_{\min} = 0.02$ — a scan shows ppl drops
monotonically with $s_{\min}$, the plateau appears at $[0.01, 0.025]$, and a value inside the
plateau was taken (the 50x clamp only affects genuinely constant channels with
$\mathrm{std} < 0.02$).

Measured (`test_csn_constant_input_stability`): a constant input $\mathbf{3.14159}$ produces
output magnitude $< 10^{-3}$;
`test_csn_unit_variance`: an input with $\mathrm{std}=7$ is normalized to std $\approx 1$.
Against LayerNorm: with a near-constant input ($5.0+10^{-7}$ noise), CSN output std
$=1.0\text{e-}6$, LN's is $1.6\text{e-}5$ — **CSN suppresses noise amplification 15x
stronger** (§12.4).

---
## 4. Original initialization: spectral anchoring

No variance-distribution scheme derived from fan-in/fan-out (Xavier/He/MSRA all belong to
this class). This scheme **directly controls the singular-value spectrum of the weight
matrices**:

1. Take a Gaussian matrix $G \in \mathbb{R}^{n\times n}$, do the QR decomposition $G = QR$,
   and sign-correct with $\mathrm{diag}(R)$ to obtain an orthonormal basis $Q$;
2. Take an orthonormal slice of $Q$, $Q_{[:m,:k]}$, and scale to the target spectral radius:
   $W = \rho \cdot Q_{[:m,:k]}$, giving $\sigma_{\max}(W) = \rho$;
3. The spectral radius is a **constant** $\rho = 1.0$, with no depth decay
   ($\beta = 0$).

**Rationale**: the orthonormal slice has condition number 1 and a flat forward-gain spectrum
(equal weight in every direction).
The choice $\rho = 1.0$ is empirically calibrated — see the table below.

**Why no depth decay (the second measurement-driven correction):**

The first version made deep weights smaller via $\rho_l = \rho_0 (l+1)^{-\beta}$
($\beta = 0.15$), on the reasoning of "offset the product gain of deep stacking".
Measurement proved the sign was set backwards — it directly created gradient vanishing:

| $\rho_0$ | $\beta$ | $L{=}8$ embedding gradient | $L{=}16$ | $L{=}24$ | 8→16→24 ratio |
|---|---|---|---|---|---|
| 1.15 | 0.00 | 1.35e+0 | 1.73e+0 | 2.50e+0 | 1.28 / 1.44 (stable) |
| 1.15 | 0.10 | 3.44e-1 | 3.11e-2 | 5.17e-4 | 0.090 / 0.017 (dies fast) |

Further scan over $\rho_0$ ($\beta = 0$):

| $\rho_0$ | $L{=}4$ | $L{=}8$ | $L{=}16$ | $L{=}32$ | Conclusion |
|---|---|---|---|---|---|
| 0.50 | 2.94e-6 | 6.42e-11 | 5.46e-20 | 0 | strong vanishing |
| 0.85 | 1.42e-2 | 2.16e-3 | 4.22e-5 | 1.76e-7 | still vanishes |
| **1.00** | **1.86e-1** | **3.38e-1** | **5.66e-1** | **6.58e-1** | **stable across depth** |
| 1.15 | 9.81e-1 | 1.35e+0 | 1.73e+0 | 4.17e+0 | slight growth |

So the final choice is $\rho_0 = 1.0$, $\beta = 0$: the single-block Jacobian gain is
calibrated near 1, and increasing the layer count neither inflates nor kills the gradient.
Measured $L = 2 \to 32$ gives embedding gradient
$1.46\text{e-}1 \to 6.58\text{e-}1$; the gradient norms of the first and last block weights are
of the same magnitude (5.1e-2 vs 5.1e-2); there is no "only the shallow layers are learning"
phenomenon.

> **Correction note**: the two tables above were measured under the old CSN
> ($s_{\min}=0.25$, never actually normalizing).
> After the CSN fix (§3) and re-measuring, the collapse boundary shifted down overall:
> $\rho_0=0.5$ no longer collapses (24/4 layer gradient ratio $1.01$), and $\beta=0.15$
> no longer triggers a collapse on its own ($1.45$); collapse only appears on the
> $\rho_0 < 0.5$ side. The final conclusions for $\rho_0=1$, $\beta=0$ are unchanged.

Measured: after initialization the first-step loss $= 6.2383$, and
$\ln V = \ln 512 = 6.238$ — the initial distribution is exactly near-uniform, the scale is
correct.
See `test_init_loss_near_uniform`, `test_spectral_init_singular_values`.

---

## 5. Cross-layer information propagation (state relay + residual)

Layer-to-layer propagation has two paths:

1. **State relay (main path)**: the end-of-block state $S^{(l)}_T$ of layer $l$ directly
   becomes the start-of-block state $s^{(l+1)}_0$ of layer $l+1$.
   The state scale is pinned by the convex-combination property of §1.2(f)
   ($\|s\|_\infty$ does not grow), so per-layer propagation **does not amplify**.
2. **Feature stream (auxiliary path)**: $o^{(l)}_t$ becomes the input $x_t$ of layer $l+1$'s
   block, re-enters through $\mathcal{N}$ and the low-rank projection, and at the block output
   is **added to the input residual**: $x^{(l+1)}_t = o^{(l)}_t + x^{(l)}_t$.

**The residual was added in the fourth correction round (the old design explicitly excluded
residuals; measurement proved that exclusion was wrong)**. The original design had $o_t$
completely replace $x_t$, leaving only the state relay as the single strict inter-layer path.
Measurement (seven-architecture same-protocol, 600 steps, multiple sub-seeds) shows this
"purity" is the **largest single quality loss**:

| Configuration | ppl (seed 1234 / 2024 / 3407) |
|---|---|
| No residual (old default) | 93.82 |
| **With residual (current default)** | **67.40 / 70.72** |
| Comparison: LayerNorm + residual | 54.90 / 54.58 / 56.58 |
| Reference: H3-2 | 53.93 |

The residual alone brings a **−28.2% to −32.5%** relative ppl improvement, the largest and most
reproducible of all changes tried
(others: learnable decay no gain, stacking state_norm no gain, adjusting $h_{\max}$ and
strengthening the nonlinearity all ineffective).
The basis: the residual provides an **identity direct path that does not go through the state
relay**, so the deep layers need not carry all information on the state bus alone, and the
"depth-gradient collapse from under-scaling" no longer happens
(`test_residual_removes_depth_collapse`: at $\rho_0=0.3$, the 24/4 layer gradient ratio is
$1.1\text{e-}4$ without residual, $1.48$ with).

The earlier "even without a residual the spectral anchor keeps the gradient" argument only held
for the old CSN (never actually normalizing, see §3); after the CSN fix that guarantee
disappeared, so the residual went from "optional" to "necessary".

| Means | Role |
|---|---|
| $\mathcal{A}' \in (1, 1.5]$ | the nonlinearity never discounts the gradient |
| $\mathcal{N}$'s $s_{\min}$ lower bound | removes "variance to 0 → gradient explosion" while keeping output scale constant |
| Spectral anchor $\rho=1$, $\beta=0$ | single-block Jacobian gain calibrated near 1, no depth decay (§4) |
| **Residual direct path** | gives the deep layers an identity path that does not go through the state relay, rescuing the depth gradient |

Measured (no residual, state bus only): at depth $4 \to 24$ the embedding gradient ratio
staying inside $[0.1, 10]$ requires $\rho_0 \ge 1.0$; under-scaling ($\rho_0=0.3$) collapses
to $1.1\text{e-}4$ —
`test_spectral_scale_decides_depth_stability` guards this path.

---

## 6. Output mapping head (LM head)

$$
\mathrm{logits}_t = W_{\text{head}}\;\mathcal{N}\!\big(o_t\big), \qquad
W_{\text{head}} \in \mathbb{R}^{V \times d}
$$

$W_{\text{head}}$ is **weight-tied** with the embedding matrix; the pretraining objective is
the standard autoregressive next-token cross-entropy:

$$
\mathcal{L} = -\frac{1}{NT}\sum_{n=1}^{N}\sum_{t=1}^{T} \ln \frac{\exp(\mathrm{logits}_{n,t,y_{n,t}})}{\sum_{v}\exp(\mathrm{logits}_{n,t,v})}
$$

Continuing SFT instruction fine-tuning is supported (same head, same objective; the data is
swapped for instruction-reply pairs).

---

## 7. Input layer

BPE subword tokenizer, implemented from scratch (`sunnynet/tokenizer.py`), with no dependency
on `tokenizers` / `sentencepiece`.

- **Pre-tokenization**: CJK runs / Latin words / numbers / symbols are split into four
  classes, so Chinese merges by phrase and English by word, while code symbols are
  compatible;
- **Merging**: standard BPE highest-frequency-first, with a heap-optimized incremental update;
- **Vocabulary**: 4 special symbols + high-frequency characters + merged items.

Measured round-trip consistency: Chinese classical text, English literature, Python code,
mixed Chinese-English — **6/6 all lossless**.

> The input layer is not part of the original network core; its presence does not violate the
> architecture's originality requirement.

---

## 8. Complexity and CPU fit

Let the sequence length be $T$, trunk dimension $d$, rank $r$, and convolution taps $k$.

| Path | Time complexity | Space complexity |
|---|---|---|
| State scan ($\star$) | $O(T r d)$ | $O(T r + \text{chunk}\cdot d)$ |
| Circulant filter | $O(T d)$ | $O(1)$ |
| Depthwise-separable conv | $O(T d k)$ | $O(1)$ |
| Nonlinear branch | $O(T d d_i)$ | $O(T d_i)$ |
| **Total** | $\mathbf{O(T)}$ strictly linear | $O(T)$ (backward can drop to $O(\text{chunk})$) |

**No token-pair interaction term** exists, so there is no $O(T^2)$ structure.

CPU-fit points:

- The trunk matrix dimensions are kept small-to-mid scale (20M prototype:
  $d=512$, $r=320$, $d_i=896$, $L=10$, measured 20,955,698 parameters) so the working set of
  a single read/write falls in L2/L3;
- Low-rank factorization reduces the largest matrices from $2d \times 2d$ to
  $r \times 2d$ and $2d \times r$;
- The circulant filter and depthwise-separable convolution have **zero matrix multiplies**;
- Native FP32, no FP16/BF16 dependency (CPU half-precision support is weak; software
  emulation is actually slower).

### 8.1 Measured optimization attempts (including rejected ones)

| Attempt | Conclusion | Basis |
|---|---|---|
| `logspace_scan` skips `p.amin()` (pass in via `p_min_hint`) | **Adopted** | single step 0.596ms→0.043ms, value unchanged |
| Scan block-length adjustment (64→128/256/512/1024) | **No gain** | T=512…4096 differences are within ±10% noise |
| `gated_scan` skip `torch.cat(outs)` for single block | **Rejected** | A/B gain ±5% and direction unstable, no real gain |
| Competition-gate roll→slice rewrite | **Rejected** | measured slower (4.91ms vs 1.95ms), keep `roll` |
| `CenteredScaleNorm` with `var_mean` fusion | **Rejected** | forward 46% faster but backward slower, and the numeric diff 4.8e-7 exceeds the 1e-10 invariant |
| State feedback `_feedback` cat-free rewrite | **Rejected** | not numerically equivalent (deviation 1.8) |
| `torch.compile` operator fusion | **Environment-limited** | MSVC missing Windows SDK headers; Inductor's Windows branch demands an MSVC-style CLI, does not accept gcc |
| Hand-written AVX2 kernel (w64devkit GCC 16.2.0) | **Adopted** | scan operator 4.3–22x faster than torch; fused kernel training step 1.10–1.17x (report §21/§22) |
| Multithreading (1→8 threads) | **Effective** | T=4096: 322ms→147ms (2.2x), concurrent operator scheduling is parallelizable |

**Key diagnosis**: operator profiling shows the bottleneck is **scheduling overhead**, not
FLOP — one forward has `aten::mm` 31%, `aten::mul` 13%, `aten::add` 11%, `aten::var` 10%,
plus **128 `cumsum` calls** (from the chunked scan). No single operator dominates, so a
point rewrite rarely pays off; the only effective path is reducing the total operator count
(fusion).
`torch.compile` is unavailable on this machine, so a hand-written AVX2 kernel bypasses
Inductor to achieve fusion.

**Measurement discipline (important)**: timing measurements must use **one process per
cell**. Measuring the same model in a process that has "already built several models" can take
**more than 2x** the time of a clean process (T=4096: 325ms vs 776–822ms), and the inflation
factor varies by architecture. All inter-architecture timing ratios without isolation are
distorted; see experiment report §2.4.

---

## 9. Difference table against existing paradigms

| Dimension | Transformer | RNN / LSTM / GRU | SSM / Mamba | **SunnyNet** |
|---|---|---|---|---|
| Sequence complexity | $O(T^2)$ | $O(T)$ | $O(T)$ | $O(T)$ |
| State update | stateless | $c_t = f\odot c_{t-1} + i\odot\tilde c$ (additive, unbounded) | $h_t = \bar A h_{t-1} + \bar B x_t$ (input-independent matrix) | $s_t = p_t\odot s_{t-1} + (1-p_t)\odot u_t$ (**convex combination, bounded**) |
| Retention source | — | Sigmoid gate (not a convex combination) | input-independent discretization step | input self-gating, upper clamped to 1 |
| Time scale | none | single | mixed (HiPPO, etc.) | log-uniform $[2, 8192]$ elementwise multi-scale |
| Inter-layer connection | residual add | residual / stack | residual | **state relay + feature residual stream** |
| Activation | GELU/SiLU, negative-side gradient $\to 0$ | tanh/sigmoid | SiLU | SoftWedge, derivative always $\in (1, 1.5]$ |
| Normalization | LayerNorm/RMSNorm | LayerNorm | RMSNorm | CenteredScaleNorm (smooth lower-bound scale) |
| Initialization | Xavier/He | Xavier | bespoke | spectral anchoring (directly controls the singular-value spectrum) |

It must be stated clearly: **SunnyNet's state recurrence form is mathematically related to
linear-attention-class work** (both are first-order linear recurrences), but the gating
construction (convex combination + log-uniform multi-scale + elementwise protection gate), the
dual-path inter-layer propagation of state relay and feature residual, the activation and
normalization are all self-designed, and no existing implementation's code or structure was
relied upon.

---

## 10. State-propagation sketch

```
        input token stream  x_1  x_2  ...  x_T
                        │    │         │
   ┌────────────────────┴────┴─────────┴──────────────────┐
   │  Block 1                                              │
   │   N(x) ─┐                                             │
   │         ├─ concat ─► low-rank projection ─► u_t ──┐  │
   │   circ(s)┘        │                               ▼  │
   │                   └─► σ(g) ──────► p_t ●── convex-combination recurrence
   │                                          ╲            │
   │  s_0 ─────────────────────────────────────► S_t        │
   │                                   ┌────────┘           │
   │                          W_c·A(W_z u) + β·S_t = o_t    │
   └───────────────────────────┬───────────────────────────┘
                               │
            end-of-block state S_T ──► becomes next block's s_0   (state relay)
                    o_1..o_T ─┘─► +x ──► becomes next block's input  (feature residual stream)
                               │
   ┌───────────────────────────┴───────────────────────────┐
   │  Block 2 … Block L   (same as above, spectral anchor ρ = 1.0, β = 0)
   └───────────────────────────┬───────────────────────────┘
                               │
                     N(o) ─► W_head (tied with Embedding) ─► logits
```

Chunked parallel scan within a block:

```
       block 1        block 2        block 3              <- all chunks within a block are parallel
     [run in parallel][run in parallel][run in parallel]
         │           │           │
         └──► back-solve block-start state from S_T, fill back block by block (only T/chunk serial steps)
```

---

## 11. Known limits and risks

1. **Long-range dependencies**: the rolling state's information capacity is bounded by the fixed
   dimension $d$, with a theoretical upper bound below full attention
   ($O(T)$ addressable). $H_{\max} = 8192$ only guarantees "not forgotten", not
   "retrievable".
   This is the architecture's primary capability risk; the
   `long_range_recall` probe in the evaluation script `train/evaluate.py` is dedicated to
   quantifying it.
2. **Pseudo-convergence**: a falling loss/ppl does not equal having reasoning and
   instruction-following ability.
   Evaluation must include an independent test set and per-domain perplexity; do not look at
   training loss alone.
3. **Magnitude of gain**: the CPU-training speedup must be measured; paper-style claims do
   not stand. See the experiment report's measured data.
