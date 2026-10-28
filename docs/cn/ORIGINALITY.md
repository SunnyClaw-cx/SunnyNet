# SunnyNet 组件原创性核验

*[English](../en/ORIGINALITY.md)*

对每个组件列出先行工作，标明哪些是既有、哪些是本项目设计。目的是把可声称的范围说准，
既不过度声称，也不把真正自己做的部分让出去。

## 1. 家族归属

状态更新是带门控的凸组合：

$$s_t = p_t \odot s_{t-1} + (1 - p_t) \odot u_t$$

这是一阶线性递推。线性注意力与线性 RNN 的等价性由 Katharopoulos 等 2020 年证明
（arXiv:2006.16236），此后 GLA、DeltaNet、RWKV、Mamba、HGRN 等都在这一范式内。
SunnyNet 属于这一家族，不声称该范式为新。

## 2. 逐组件对照

### 2.1 范式与公式（既有）

| 组件 | 先行工作 |
|---|---|
| 一阶线性递推 / 线性注意力=线性 RNN | Katharopoulos et al., 2020, arXiv:2006.16236 |
| 凸组合更新式 `H_t = Z_t⊙H_{t-1} + (1-Z_t)⊙H̃_t` | Cho et al., 2014（GRU），更新式同形 |
| 矩阵值状态递推 | RWKV-7、Mamba-2、DeltaNet 均为矩阵状态 |
| 逐通道 / 逐维衰减 | RWKV-4（`w`、`u` 是 d 维向量），HGRN2（分层门控） |
| 对数域参数化衰减 `g ≤ 0, exp(g) ∈ (0,1]` | KDA（Kimi Delta Attention），逐 key 维遗忘门 |
| 多时间尺度 / 对数分桶记忆 | Log-Linear Attention，Guo et al., 2025, arXiv:2506.04761（对数增长的隐藏状态集合） |
| 沿序列的 softmax 遗忘门 | Forgetting Transformer, arXiv:2503.02130 |
| sigmoid 门控（头级最优，乘法优于加法） | Qwen Gated Attention, arXiv:2505.06708（NeurIPS 2025 最佳论文） |
| 分组归一化 / 分组结构 | Group Normalization, Wu & He, 2018, arXiv:1803.08494 |
| 短因果卷积做位置混合 | Mamba（S6），`d_conv=4` |
| chunkwise 并行扫描（块内闭式 + 块首反解） | Blelloch 1990 前缀和；FLA 库；Mamba / GLA / DeltaNet / RWKV-7 均用 |
| delta rule 写入（减去旧关联再写新值） | DeltaNet, Yang et al., 2024, arXiv:2406.06484 |
| 分块 delta rule / WY 表示 `W=(I+L)⁻¹diag(β)(V−K S_in)` | DeltaNet 的 chunkwise 形式；FLA 库 `chunk_delta_rule`；Gated DeltaNet |

### 2.2 工程与运行时（既有，属行业标准）

| 组件 | 先行工作 |
|---|---|
| C / SIMD 推理内核（AVX2 / AVX-512 / NEON） | llama.cpp / ggml、rwkv.cpp |
| INT4 分组量化（block-wise） | llama.cpp k-quant 系列、ggml 通用 INT4 GEMV |
| GGUF 权重格式 | llama.cpp / GGML 团队定义，本地推理事实标准 |
| OpenAI 兼容 API（`/v1/chat/completions` + SSE） | llama.cpp server、Ollama、vLLM |
| mmap 只读加载大权重 | llama.cpp、llamafile |
| safetensors / HF 权重加载 | HuggingFace transformers |
| 解码受带宽而非算力约束 | memory wall，Wulf & McKee, 1995 |

### 2.3 本项目设计（可声称）

| 组件 | 形式 | 最近的先行工作与差异 |
|---|---|---|
| SoftWedge 激活 | `f(z)=z+a·s(z)`，底式 `s(z)=½+½·z/(1+\|z\|)` | Mish / SiLU 也是自门控激活；SoftWedge 的底式与导数区间 `f' ∈ (1, 1+a/2]` 未见相同形式 |
| CenteredScaleNorm | 居中方差 + 平滑下界 `var + s_min² + eps` | 思路承接 RMSNorm / LayerNorm；"平滑下界防止近常量向量被放大"的写法为自设 |
| 谱锚定初始化 | QR 正交切片，直接锚定 `σ_max` | 承接正交初始化 Saxe et al., 2013；直接锚定谱半径为自设 |
| 组内成对竞争门控 | `σ(g_i-g_next(i))·σ(g_i-g_next2(i))`，组内两项滚动乘积 | 动机同 Qwen 门控；具体"双 sigmoid 滚动乘积"形式未见相同用法 |
| 跨层状态接力 + 特征残差流 | 状态总线 + 恒等残差直连 | 残差为既有（He et al., 2016）；两者的组合与实测标定为自设 |
| 通用一阶递推 C 内核 | `sn_recur_scan` / `step` / `mat_scan` 一套抽象覆盖 SunnyNet / RWKV-7 / Mamba-2 | ggml 算子层有通用内核概念，但针对"一阶递推"的专门抽象粒度不同 |

## 3. 独立观测

以下为实测数据，谁也无法提前占有：

- 带宽账本与顺序读天花板（`bench_bw.c`）。
- 同权重同精度下 llama.cpp 与 SunnyNet C 内核的对照。
- 同参数跨架构 BPC 对照与同域小模型评测。
- AVX-512 尾路指针绑定 bug 的定位与修复。
- 软写（凸组合）与 delta 写在覆盖召回任务上的对照，以及 delta 分块形式在 fp32 长块下的
  发散定位（`AminBackward0`）与块长上限。
- 写 key 必须取自上一个位置才能写入 (key→value) 关联的结构性定位（k/v/q 同取当前位置只能
  表达 token→token，两规则都停在随机水平）。
- 分块 delta 写比软写更省的实测：软写的 `log`/`cumsum`/`exp` 全跑在 $\text{slots}\cdot m$
  宽上，delta 把状态留在 $(\text{slots},m)$ 窄形状（训练步 0.51–0.89×，随 slots×m 增大）。
- **衰减归约方向的定位**：逐通道衰减压成标量时取 `amin`（最快遗忘通道）会把块内
  $1/\prod d$ 的条件数在 chunk=64 下拉到 9.92e16，`amax` 仅 1.78；`amin` 同时只用掉
  1/16 的通道输出。这一条是标量衰减头（`assoc_decay='scalar'`）与 $p$ 必须沿用 H 参数化
  下界的依据，属独立观测。
- 四处被实测推翻并修正的设计结论（见 ARCHITECTURE.md §3/4/5 与 README）。

## 4. 可声称的边界

**是**本项目的：SoftWedge 激活、CenteredScaleNorm、谱锚定初始化的具体形式；
组内成对竞争门控的具体形式；跨层状态接力的组合；通用一阶递推 C 内核抽象；
以及全部实测数据。

**不是**本项目的：状态更新的凸组合形式、delta rule 写入、分块 delta / WY 表示、
多尺度遗忘、对数域衰减、sigmoid 门控、chunkwise 并行扫描、SIMD/量化/GGUF/OpenAI
接口等工程环节。`ops.chunk_delta_scan` 是 delta rule 分块形式在本代码库里的实现，
算法本身属 DeltaNet / FLA 一系。

一句话：这是线性递推架构的一个变体，在组件选择与组合上有自己的取舍；
最接近的对照是 KDA 之于 Gated DeltaNet——赛道内变体，不是新范式。

## 参考文献

- Katharopoulos et al., *Transformers are RNNs*, 2020. arXiv:2006.16236
- Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder*, 2014. arXiv:1406.1078
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, 2024. arXiv:2406.06484
- Wu & He, *Group Normalization*, 2018. arXiv:1803.08494
- Qiu et al. (Qwen), *Gated Attention for Large Language Models*, 2025. arXiv:2505.06708
- Guo et al., *Log-Linear Attention*, 2025. arXiv:2506.04761
- Lin et al., *Forgetting Transformer*, 2025. arXiv:2503.02130
- Saxe et al., *Exact solutions to the nonlinear dynamics of learning in deep linear networks*, 2013. arXiv:1312.6120
- Wulf & McKee, *Hitting the Memory Wall*, 1995. ACM SIGARCH Comput. Archit. News
- Blelloch, *Prefix Sums and Their Applications*, 1990