# SunnyNet

[![tests](https://github.com/SunnyClaw-cx/SunnyNet/actions/workflows/tests.yml/badge.svg)](https://github.com/SunnyClaw-cx/SunnyNet/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sunnynet-cx)](https://pypi.org/project/sunnynet-cx/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**中文** · [English](README_en.md)

---

## 中文简介

原创通用序列神经网络架构，不基于 Transformer、RNN、Mamba、CNN 等任何现有架构。

仓库地址：<https://github.com/SunnyClaw-cx/SunnyNet>

架构面向 CPU 训练设计，O(N) 线性序列复杂度，降低内存带宽消耗，提升 CPU 缓存利用率。
支持通用文本自回归预训练，可进行指令微调与领域微调。

源代码使用 Apache License 2.0 开源。

---

## 目录

- [快速开始](#快速开始)
- [架构核心思想](#架构核心思想)
- [原创组件](#原创组件)
- [实测结果](#实测结果)
- [与六家主流架构的对照](#与六家主流架构的对照)
- [线性架构的 CPU 推理 Runtime](#线性架构的-cpu-推理-runtime)
- [仓库结构](#仓库结构)
- [原创性说明](#原创性说明)
- [风险与局限](#风险与局限)
- [赞助](#赞助)
- [许可证](#许可证)

英文全文见 [English](README_en.md)。

---

## 快速开始

### 直接跑已发布的小权重

无需自己训练，装完即可生成一句话：

```bash
pip install sunnynet-cx                                          # 会一并装 torch + numpy
hf download SunnyClaw-cx/SunnyNet --local-dir sunnynet-small-zh  # 约 21 MB
sunnynet --ckpt sunnynet-small-zh/sunnynet-small-zh.pt \
    --tokenizer sunnynet-small-zh/tokenizer.json --prompt "子曰：" --max-new 60
```

`hf` 随 `huggingface_hub` 一起装。这个权重是 1.4M 参数（dim 128、4 层、词表 8192），
在本仓库中文语料上训 16000 步（验证 ppl 139.4），输出文言文。加
`--backend c --blob sunnynet-small-zh/sunnynet-small-zh.runtime.bin`
可改走 C 内核解码，不用 torch 前向。

### 环境

- Python ≥ 3.10
- PyTorch ≥ 2.1（CPU 版即可，无 GPU 依赖）
- numpy

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pytest

# 安装本包 (发布名为 sunnynet-cx, 导入名仍是 sunnynet)
pip install sunnynet-cx          # 或 pip install -e . 从源码装
```

从 PyPI 安装的 wheel 已内置预编译原生库（x86-64 且支持 AVX2+FMA 时生效），装完即可
使用 C 内核，无需本机 C 编译器。其他平台或指令集不达标时退回源码编译，此时才需要
下面的编译器。

### 1. 下载语料并构建 token 流

```bash
# 采集真实语料: Project Gutenberg 中文古籍/古典文学 + 英文文学 + TheAlgorithms/Python 源码
python prepare_data.py

# 训练 BPE 分词器 -> 编码为 token 流 -> 划分训练/验证
python build_corpus.py --vocab-size 8192
```

验证集按文件级切分，同一文本的相邻片段不会跨训练与验证。

### 2. 训练

```bash
# 小模型快速验证（4 层）
python train/train.py --preset small --steps 800 --batch-size 8 --seq-len 256

# 20M 原型
python train/train.py --preset proto20m --steps 5000 --batch-size 4 --seq-len 512
```

### 3. 评估

```bash
python train/evaluate.py --ckpt runs/<你的>.pt --gen \
    --out experiments/results/eval_sunnynet.json
```

### 4. 完整实验

```bash
bash experiments/battery.sh          # 基线 + 消融 + 种子 + 缩放
bash experiments/battery2.sh         # 20M 缩放对照 + 长序列训练对照

# O(N) 验证: 前向延迟 vs 序列长度
python experiments/long_sequence.py --only latency

python experiments/make_figures.py
python experiments/collect_results.py --out docs/_tables.md
```

手写 AVX2 内核需要 C 编译器（w64devkit / GCC），但只在源码路径下：从 PyPI 装的
wheel 已带预编译库，或本机 CPU 不支持 AVX2+FMA 时才回退到编译。源码方式使用请将
`SUNNY_W64` 指向 w64devkit 根目录，或把 `gcc` 加入 PATH。

```bash
export SUNNY_W64=<w64devkit 解压目录>
python -X utf8 experiments/_kernel_bench.py        # 算子级加速比
python -X utf8 experiments/_core_e2e.py 256 6 8    # 融合内核训练步 A/B
```

---

## 架构核心思想

CPU 预训练的瓶颈通常不在浮点算力，而在内存带宽与缓存命中率。SunnyNet 的设计围绕三条硬约束：

| 约束 | 对应设计 |
|---|---|
| 序列复杂度严格 O(N) | 滚动状态 + 一阶线性递推，无任何 token 两两交互 |
| 不保存整条序列的中间激活 | 状态维度固定，跨块只传状态；反向按需求重算 |
| 规避巨型稠密矩阵乘法 | 低秩分解 + 深度可分卷积 + 三对角循环滤波 |

核心状态更新是凸组合：

```
s_t = p_t ⊙ s_{t-1} + (1 - p_t) ⊙ u_t ,     p_t ∈ [1-δ_i, 1]
```

两个系数非负且和为 1，因此 `|s_t| ≤ max(|s_{t-1}|, |u_t|)`，状态天然有界，不会发散。

保留率 `p_t` 的逐通道下界由记忆跨度 `H_i` 决定，`H_i` 在 log 刻度上遍历 `[2, 8192]`，
同一层内同时存在从"只记 2 个 token"到"记忆 8192 个 token"的全时间尺度，
兼顾局部语法与跨段落、跨代码块的长距离依赖。

完整数学推导见 **[docs/cn/ARCHITECTURE.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/cn/ARCHITECTURE.md)**。

---

## 原创组件

| 组件 | 方案 | 关键性质 |
|---|---|---|
| 状态更新 | 多时间尺度凸组合滚动 | 状态有界，不发散；多尺度输出门控 |
| 非线性激活 | SoftWedge `f(z)=z+a·s(z)`，底式 `s(z)=½+½·z/(1+\|z\|)` | 导数 `f' ∈ (1, 1+a/2]`，处处非零，无梯度断崖 |
| 归一化 | CenteredScaleNorm | 分母设平滑下界 `s_min`，近似常量向量不被放大（深堆叠前提） |
| 初始化 | 谱锚定（QR 正交切片，σ_max 直接锚定） | 条件数为 1，前向增益谱平坦；ρ=1.0 由实测标定 |
| 跨层传递 | 状态接力 + 特征残差流 | 状态尺度不随深度放大；残差直连兜住深度梯度（实测 −28% ppl） |
| 通道竞争 | 组内成对竞争门控 | 与 softmax 同语义，但 Jacobian 为 O(1) 而非 O(1/d) |
| 并行扫描 | 对数域闭式解 | 与逐步递推代数等价；块首状态可由块末精确反解 |

未复用 ReLU / GELU / SiLU / Mish、LayerNorm / RMSNorm / InstanceNorm / GroupNorm、
Xavier / Glorot / He / MSRA 初始化、LSTM / GRU 门控公式、SSM / HiPPO 离散化。

---

## 实测结果

全部实验共用同一 token 流、同一配置、同一 token 预算与线程数。参数量用 meta device
实例化对齐，层数偏离基准不超过 ±1。完整数据见 [docs/cn/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/cn/EXPERIMENT_REPORT.md)。

### 训练速度

- **同架构提速**：算子融合 19–41%；减层加宽 12–17%；窄残差流 + `p.clamp` 判掉 17.7–20.7%。
- **对同复杂度同行**：T=256 训练步 0.0953 ms/token，名义快 rwkv 3.0%；9 轮逐轮配对检验
  不显著（3–4/9 轮占优）⇒ 写作"与 rwkv 统计持平"，不写"超越"。对 h3/mamba/hyena/linear_attn
  稳定领先 16–41%。
- **持平的边界**：仅 `L=4` 下成立；层数对齐到 `L=6` 时同预算窄配置反慢 30–40%；T≥1024
  同行优势仍在。
- **手写内核**：融合竞争门控 + 门控概率 + 扫描，训练步 A/B 三指标一致同向，
  T=256 6/6 占优、比值 0.8746（1.143×）；ppl 与 torch 逐位一致（不改变质量，只提速度）。
  推理态内核收益不显著（B=1 张量太小，跨语言调用开销与单算子同量级），默认关闭。

### 推理

- 单步解码成本随上下文：SunnyNet 平坦（O(1)，2.87→2.90 ms，漂移 ±5%），
  Transformer 单调上升（O(t)，2.49→6.85 ms）。交叉点 T≈512，T=4096 时 SunnyNet 快 **2.36×**，
  差距随 T 继续拉大。不需内核即可获得。
- 短上下文（T<512）解码不是优势场景，内核融合在此也无收益。
- 边界：单条序列、B=1、随机初始化模型、2 轮中位，未测批处理解码，不含 KV cache 量化/分页。
- 与 O(N) 同行同场对照见下节：同一 torch 实现下 SunnyNet 解码落后 rwkv 1.73–2.16×；
  换整模型 C 后端后单步全长度第 1，比 rwkv 快 1.02–1.15×。两条路径分列。

### 质量

- 架构修正（残差 + CSN 地板）后，T=256 ppl 由 93.82 降至 57.83，七架构第二
  （仅落后 H3 53.93）；其余为 Mamba 63.34 / RWKV 64.10 / Hyena 82.50 / Transformer 90.05 / 线性注意力 91.91。
- 20M 减层加宽后 ppl 再降 10.9%（54.82），在同协议对照中升至五架构第一。
- **scaling 趋势非单调**：small 质量第二 → 20M 质量第一 → 100M 退回第二（109.47，落后
  rwkv 105.66 约 3.6%）。不能说"规模越大优势越大"。

### 长序列与内存

- **前向**：交叉点 T≈384（6 层对照）/ T≈600（4 层对照）；T=4096 时每 token 比
  Transformer 快 22.7×（batch=1、纯前向、8 线程、1.4M 小模型）。
- **训练态 T=1024**：SunnyNet 速度第 1（0.1319 ms/tok），峰值内存 1812 MB 第 2 低档
  （Transformer 3192 MB，高 76%）。
- **对 O(N) 同行的前向**：无稳定领先区间，随 T 从第 6 名追到第 3 名，全程未登顶；
  训练态排名不同（T=1024 时第 1）。

### 跨分布泛化

外部独立数据集（原三域训练，新域测试）：中文科技新闻 ppl 321.08 第 1/7（该域未登录字符率
23%）、英文 arxiv 摘要 35.58 第 2/7；OOD 落差 1.480，七架构最低档（与 hyena 并列）。
绝对 ppl 仍高、600 步未收敛、单种子，只能说"跨分布不比同复杂度同行差"。

### 短序列

交叉点 T≈384 只在 batch=1 纯前向成立。训练态（batch=8）交叉点下移到 T<256：
T=256/384 时比 Transformer 快 1.5–2.4×。T=128 仍慢 10.4%（旧 d128）/ 4.5%（窄配置
smalln80）——窄残差流是已验证的缓解手段。

### GPU

架构无需 CUDA 特化改写即可在 CUDA 上运行。B=8 加速：前向 8.0–15.2×、训练步 9.9–12.5×
（未做 CUDA 侧优化，是下界）。本机 GPU（GTX 1060 6GB）仅用于正确性验证与加速比测量，
无同行 GPU 对照。

### 100M 规模 GPU 训练与权重发布

`p100mw`（d1408/i1920/r1024/L8，栈 93.83M / 总 105.36M）在 GTX 1060 6GB 上跑了
2000 步真实训练（`zh_ancient` 0.4 / `en_literature` 0.35 / `code_python` 0.25，
bs=2 × seq=256 × 2000 = 1,024,000 token，lr 1e-3 + 100 步 warmup）：

| 步 | train loss | val loss | ppl |
|---|---|---|---|
| 200 | 5.8046 | 5.7407 | 311.29 |
| 600 | 5.1452 | 5.3599 | 212.71 |
| 1000 | 4.8508 | 5.1040 | 164.68 |
| 1400 | 5.9248 | 4.8997 | 134.25 |
| 2000 | 2.9600 | 4.7882 | **120.09** |

val 单调下降、无发散，全程 **0.503 ms/token**（1987 tok/s），wall 583.5s，peak_rss 1302MB。

**6GB 卡的可用配置只有 bs=2 × seq=256**（实测边界）：

| 配置 | 结果 |
|---|---|
| bs=1 × 256 | 通过，1.220 ms/tok |
| **bs=2 × 256** | **通过，0.744 ms/tok（最优）** |
| bs=4 × 128 | 通过，0.818 ms/tok |
| bs=4 × 256 / bs=8 × 128 | 训练后期 OOM |

三条测量注意：`expandable_segments` 在 sm_61 上**不可用**，开启反而更早 OOM；
显存被其他进程（如常驻的 Ollama）占用时 free 看似充足但分配失败，
须先 `keep_alive: 0` 卸载；验证批次 `val-batches 8` 也会 OOM，降到 2 才稳定。

权重打包由 `sunnynet-release` 完成（`dist/release/sunnynet-p100mw.*`）：

| 产物 | 大小 | sha256 前 12 位 |
|---|---|---|
| `.pt` checkpoint | 467.7 MB | `399e604e635b` |
| `.safetensors`（139 张量，绑定 head 已去重） | 421.5 MB | `c7287c76ac8b` |
| `.runtime.bin`（C Runtime fp32 布局） | 467.6 MB | `95c195e47637` |

三者均验证可加载并生成：torch 后端 `"“H不得你，也不可不得…"`（143.6 ms/tok）、
C 后端 `"「尚與子之子，雖以一名，於我也。"`（**80.2 ms/tok**）、
OpenAI 服务经官方 SDK 四种调用方式全通过。**1520 步上限、单种子，
不作"已收敛"或跨架构质量声明。**

### 0.5B 在 6GB 卡上的实测边界

| 档位 | 峰值显存 | 结果 |
|---|---|---|
| 构建（498M 参数，fp32） | 1902 MB | 通过 |
| 前向 T=512 | 2010 MB | 通过 |
| 前向 T=1024 | 2060 MB | 通过 |
| **SGD 单步**（T=512） | **5004 MB** | **通过**，但只剩 563 MB（9% 余量） |
| AdamW 单步 | >6144 MB | **OOM** |

约束来自优化器状态的算术：fp32 参数 1.9GB + 梯度 1.9GB + AdamW 一阶/二阶矩 3.8GB
= 7.6GB > 6GB，SGD 只有梯度一项所以勉强塞得下。
优化器状态 CPU offload 在本机失败——主机物理内存只剩 4.5GB，装不下 offload 的
3.7GB 优化器态，触发页文件压力并拖垮 CUDA 上下文。
⇒ **0.5B 在这台机器上可跑单步 SGD，但无多步收敛与质量数据。**

测量注意：同一进程内先跑 T=512/T=1024 前向再跑训练步，会因显存碎片使训练步 OOM
（实测剩余 1.9GB 仍分配不出 16MB）；**每档必须独立子进程**，否则得到错误的"不可行"结论。

---

## 与六家主流架构的对照

对照对象：`h3`、`rwkv`、`mamba`、`hyena`、`linear_attn`、`transformer`。
所有数字来自本机实测、逐格独立子进程。完整数据见 [docs/cn/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/cn/EXPERIMENT_REPORT.md) 第 23 节。

### 记分卡

| 维度 | SunnyNet 位置 | 最优者 | 差距 |
|---|---|---|---|
| 质量 ppl（1.4M / 600 步） | 第 2（57.83） | h3 53.93 | 落后 7.2% |
| 质量 ppl（20M / 200 步） | **第 1**（77.90） | sunnynet | — |
| 质量 ppl（100M / 100 步） | 第 2（109.47） | rwkv 105.66 | 落后 3.6% |
| 训练速度（L=4 等栈参数） | 与 rwkv 统计持平 | rwkv（名义） | 配对检验不显著 |
| 训练速度（L=6 层数对齐） | **落后** | rwkv | 慢 30–40% |
| 训练态内存（T=1024） | 第 2 低 | rwkv 1611MB | 高 12%，比 TF 低 43% |
| 前向 ms/token（batch=1） | 第 6 → 第 3（随 T） | hyena | 全程未登顶 |
| 解码 ms/step（B=1，torch 路径） | 第 7 / 8 | **rwkv** | 慢 1.73–2.16× |
| **解码 ms/step（B=1，整模型 C runtime）** | **第 1 / 8** | sunnynet | 比 rwkv 快 1.02–1.15× |
| 长距离检索（代码域） | 与 TF 持平 | 并列 | — |
| 长距离检索（自然语言） | 接近 0 | — | 无优势 |
| 跨分布泛化 | 中文新域第 1 / 英文第 2 | sunnynet | — |

### 解码态八路径对照

口径：每个 (路径, T) 独立子进程；先逐 token 走满 T 步建立真实解码状态（不计时），
再测该长度下单步解码 ms；1 线程；2 轮取中位。含同一规格的整模型 C 后端（`sunnyC`）。

ms/step（越低越好）：

| 路径 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| **`sunnyC`（整模型 C）** | **1.560** | **1.630** | **1.547** | **1.518** | **1.647** |
| `rwkv` | 1.779 | 1.879 | 1.673 | 1.675 | 1.674 |
| `transformer` | 2.473 | 3.405 | 5.236 | 4.056 | 6.962 |
| `h3` | 2.643 | 2.944 | 2.568 | 2.753 | 2.874 |
| `mamba` | 3.160 | 3.063 | 2.845 | 3.171 | 3.897 |
| `linear_attn` | 3.467 | 4.207 | 4.019 | 3.101 | 3.636 |
| `sunny`（torch） | 3.778 | 3.973 | 3.166 | 3.624 | 2.889 |
| `hyena` | 4.338 | 3.310 | 3.281 | 3.693 | 3.046 |

读数：

- Transformer 的 O(t) 被复现（2.473 → 6.962 ms）；对整模型 C 后端 T=4096 慢 **4.23×**。
- 整模型 C 后端单步平坦（1.56 → 1.65 ms），全 5 个长度第 1，比 rwkv 快 1.02–1.15×。
- **同一 torch 实现下打不平 rwkv**：`sunny`(torch) 全程落后，rwkv 快 1.73–2.16×。
  差异来源是算子数——rwkv 单步只有 `p*s + k*v` 一条逐元素链，SunnyNet 含竞争门控
  （3 次 sigmoid）、门控概率、扫描、`W_c`/`W_s` 反馈。C 后端反超正是因为它把这一串
  算子收进一次 `sn_step` 调用（消除每步约 1100 次 aten 调用），**不是架构的改变**。
- 两条路径必须分开陈述；`sunnyC` 与其余七条不是同一实现层级（C 内核 vs PyTorch）。

### 差距的共同根因

落后的维度（解码速度、前向速度、L=6 训练速度、100M 质量、自然语言长距离检索）
收敛到同一个根因：**逐元素算子数量**。证据：训练步 4524 次 aten 调用；
减层加宽（等栈参数）训练快 12–17% 且 ppl −10.9%；层数对齐到 L=6 后同预算窄配置反慢 30–40%；
解码态 rwkv 靠单条逐元素链取胜。⇒ 有效方向只有减算子数，不是加宽、调 batch 或换矩阵库。

---

## 线性架构的 CPU 推理 Runtime

`sunnynet/runtime/` 是独立于 PyTorch 的 C 推理内核（AVX2/FMA + OpenMP），运行期只依赖
ctypes。原生库随 wheel 预编译分发，源码路径下才需要 C 编译器。它实现通用一阶递推

```
s_t = a_t * s_{t-1} + b_t              (FMA)
s_t = a_t * s_{t-1} + (1 - a_t) * u_t  (门控)
```

RWKV 的 wkv、Mamba/S6 与 H3 的对角 SSM、S4、线性注意力的 KV 累加、SunnyNet 的滚动状态，
全部归约为这两种形式之一。

### 跨架构映射（`experiments/_unified_ir_map.py`）

各架构递推映射到内核参数，与自身 PyTorch 递推逐元素比对，B=2/T=128/N=128/fp32：

| 架构递推形式 | 映射 | max\|diff\| |
|---|---|---|
| RWKV-4 wkv | `a=p`, `b=k*v` | 9.5e-07 |
| Mamba / S6 SSM | `a=decay`, `u=bu` | 2.4e-07 |
| H3 对角 SSM | `a=decay`, `u=v` | 1.2e-07 |
| 线性注意力累积 KV | `a=1`, `b=k v^T` | 0.0 |
| SunnyNet 滚动状态 | `a=p`, `u=u` | 1.9e-06 |
| S4 对角 SSM | `a=A_bar`, `b=B_bar x` | 7.2e-07 |

Hyena 是长卷积（FIR）递推，不属一阶状态递推，需独立卷积后端。

### 速度（`experiments/_runtime_bench.py`）

每格独立子进程、单线程、B=1、10 次中位。基线是 torch 向量化分块扫描
（`ops.gated_scan`，cumsum 闭式解），不是 Python 循环。

| T | N | 模式 | 向量化 ms | 内核 ms | 倍数 |
|---|---|---|---|---|---|
| 256 | 128 | FMA | 0.391 | 0.043 | 9.08× |
| 1024 | 128 | FMA | 1.256 | 0.132 | 9.54× |
| 4096 | 128 | GATED | 8.500 | 0.806 | 10.55× |
| 4096 | 640 | FMA | 26.944 | 7.574 | 3.56× |

中位 **9.90×**，范围 3.56–19.29×。N=640 时降到 3.6–4.0×（单线程内存带宽限制）。
与内核的三分支最大数值差 7.63e-06。

**边界**：这是**批量跨段扫描**口径（B=1、整段 T）。接入 B=1 单步解码后收益消失
（四格比值 1.023 / 1.072 / 0.897 / 1.073，方向不一致，8 线程 T=256 反慢 10.3%），
⇒ 不得声称端到端解码加速。原因：模型侧 T=1、B=1，跨语言调用与状态搬运成本
与单算子同量级。B=1 单序列也不是本 Runtime 的目标场景（同一 torch 实现下解码态最优是 rwkv）。

### 整模型 C 解码（`sunnynet/runtime/sunnynet.c`）

单步算子内核之上，整个前向也搬进了 C：嵌入查表、逐层 norm / 深度卷积历史 / 门控竞争 /
状态递推 / 读出、终态归一、LM head，全在一条无中间张量的路径上完成。权重由
`experiments/_export_runtime.py` 从 ckpt 导出为打包 fp32 blob（矩阵乘权重预转置为
`(in, out)` 以配合 GEMV，`circ ∘ W_s` 折成单矩阵）。

数值一致性（`experiments/_runtime_model_equiv.py`，1.4M ckpt，48 token 逐步）：

| 口径 | 结果 |
|---|---|
| 单步 logits max\|diff\| | 1.24e-05 |
| argmax 一致 | 48 / 48 |
| prefill 末 token 与 torch 整段前向 | 3.81e-06 |

端到端解码延迟（`experiments/_runtime_decode.py`，B=1 单序列、单线程、每格独立子进程、
先走满 prefill 再测单步、3 轮交错）：

| prefill T | torch ms/step | C ms/step | 比值 | 逐轮占优 |
|---|---|---|---|---|
| 256 | 3.796 | 2.487 | 1.53× | 3/3 |
| 1024 | 3.802 | 1.851 | 2.05× | 3/3 |
| 2048 | 4.309 | 2.189 | 1.97× | 3/3 |

中位比值与逐轮符号检验一致（四格全 3/3），可判定为真实提升，非轮间噪声。
提速来源不是算子融合，而是消灭每步约 1100 次 aten 调用 / 21 次 GEMV / 20 次 cat
（torch 单步剖析：`aten::mm` 16.4%、`aten::mul` 10.2%、`aten::select` 8.5%、`aten::cat` 8.0%）。

**口径边界**：上表是**单线程 B=1**。CLI 默认 8 线程时 C 后端 4.40 ms/tok vs
torch 4.56 ms/tok（24 token 单次读数，未做多轮），差距被线程调度摊平——B=1 小张量
并行是负收益（§20.2 已记录 1 线程比 8 线程快 1.42×）。C 后端的优势在单线程与更大批量，
多线程 B=1 下不要按上表线性外推。

### 大模型解码的访存重构（100M 级）

上表是 1.4M 级 ckpt。换到 `p100mw`（105.36M 参数，fp32 blob 467.6 MB）后单步
80 ms，量级明显不对。量化后确认瓶颈在**权重布局**而非算子：单步至少读一遍全部权重，
而原实现的 GEMV 内层按 `out` 步长跨读，有效带宽只有 6.17 GB/s
（本机 AVX2 顺序读上限实测 25.31 GB/s）。

`sn_load` 在内存里把导出的 `(in,out)` 块翻转成 `(out,in)` 行主序（导出格式不变，
已发布的 blob 无需重导），GEMV 改为按输出通道并行、每行连续读：

| 口径 | 结果 |
|---|---|
| 优化前 | 76.60 ms/token，6.10 GB/s |
| 优化后 | **25.72 ms/token，18.18 GB/s** |
| 加速（5 轮交错 head-to-head） | **2.98×**，C 胜 5/5 |
| logits max\|diff\| vs 优化前 | 2.766e-05 |
| argmax 一致 | 32 / 32 |

并行度取**物理核而非逻辑核**：解码访存受限，超线程不增带宽。实测 6 核 26.69 ms
vs 12 逻辑核 36.05 ms（差 26%）。C 侧 `sn_recommended_threads()` 在 Windows 走
`GetLogicalProcessorInformation`，Linux 解析 `/proc/cpuinfo`；CLI 在 `--backend c`
且未指定 `--threads` 时自动采用。

**边界**：2.98× 是本机（i7-8750H 双通道 DDR4-2667）单步解码的倍数，只代表访存效率
从 6.1 提到 18.2 GB/s；18.18 仍低于 25.31 的上限（每步还有归一化等非流式访问）。
不得外推到训练或 prefill。

### fp16 权重：每步读取字节减半

行主序之后单步仍是访存受限，内核已达纯顺序读上限，剩下的杠杆是每步少读字节。
把 5 个 GEMV 矩阵与 LM head 在**载入时**截断成 fp16（小参数仍 fp32），
单步实读从 467.6 MB 降到约 222.3 MB：

| 口径 | fp32 | fp16 |
|---|---|---|
| 中位 ms/token | 24.67 | **18.60** |
| 有效带宽（fp32 等价字节） | 18.95 GB/s | 25.14 GB/s |
| 加速（11 轮交错 head-to-head） | — | **1.33×**，fp16 胜 11/11 |
| logits max\|diff\| / 相对偏差 | — | 4.181e-03 / 7.42e-04 |
| argmax 一致 | — | 64 / 64 |

内核一次读 16 个 half（32 B）后转 fp32 做 FMA，保持与 fp32 相同的加载宽度；
128 位加载会损失宽度（单线程 17.45→15.06 ms），故取 256 位。

fp16 已不再是访存受限（实读 222.3 MB / 18.6 ms = 12.0 GB/s，远低于 fp32 的 19），
瓶颈转移到类型转换。用法：`--backend c --half`，fp32 仍是默认且逐位不变。

**边界**：1.33× 是本机单步小数，不得外推 prefill 或训练；相对偏差 7.4e-04
会让长序列 greedy 解码偶发分歧，本序列 64/64 只是样本，需要逐位复现时用 fp32；
权重磁盘文件仍是 fp32 blob，fp16 只在载入时转换。

### 状态量化（`experiments/_state_quant.py`）

凸组合状态幅度由构造有界。T=2048、真实 ckpt 第 0 层状态：

| 模式 | \|s\| 漂移 (末/1/8 处) | 固定标度相对误差 | 溢出通道比 |
|---|---|---|---|
| 有界递推 | 1.06× | 4.0e-02 | 0.78% |
| 无界累加 | 7.94× | 5.0e-01 | 5.47% |

固定标度下的相对误差相差 12.5×。
边界：只测第 0 层、单 ckpt（1.4M）、单条随机序列，不做端到端 ppl 对照。

### GGUF 只读（`sunnynet/runtime/gguf.py` + `rwkv7.py`）

读入外部 GGUF 权重并逐 token 自回归生成，不经 PyTorch。首个打通的模型是官方
RWKV-7 191M（`BlinkDL/rwkv-7-world` 的 `.pth` 经 llama.cpp 转 GGUF，bf16）。
张量命名与递推形式取自 llama.cpp 的 `src/models/rwkv7*.cpp`；参考数学取自
BlinkDL/RWKV-LM 的 `RWKV-v7/rwkv_v7_numpy.py`。

数值验证（`experiments/_rwkv7_gguf_equiv.py`）：GGUF 反量化结果与官方 `.pth`
**逐位一致**（`max|diff| = 0`）；逐 token 单步 logits 与独立参考实现：

| 口径 | 结果 |
|---|---|
| 单步 logits max\|diff\| | 4.79e-06（bf16 精度内） |
| argmax 一致 | 12 / 12 |

生成样例（`experiments/_rwkv7_gguf_bench.py`，温度 0.7 / top-k 20 / seed 42）：

```
The Eiffel Tower is located in the city of
-> Paris, France. The Eiffel Tower was designed by Gustave Eiffel in 1887 as a monument to
def fibonacci(n):
-> \n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)
```

**口径边界**：numpy 路径证明的是"Runtime 能读外部 GGUF 权重并算对"，
解码约 388 ms/step——**这个数字不作速度成果陈述**。RWKV-7 的递推已搬进 C 内核（见下节）。

一条与既有叙事的差异必须记录：RWKV-7 的状态是**逐 head 的矩阵**
`S ∈ R^{64×64}`，更新为 `S_t = S_{t-1}⊙w_t − (S_{t-1}k_t)k_tᵀ + v_t k_tᵀ`，
含 rank-1 去除项——**不是**元素级 `s = a⊙s + b`。统一 Runtime 若要覆盖它，
IR 需要在元素级递推之外补一个矩阵值递推（外积 / rank-1 校正）层级。

### RWKV-7 的 C 内核（`sunnynet/runtime/rwkv7.c`）

把整模型单步搬进 C：嵌入查表、逐层 norm / time-mix（含 LoRA 与 `tanh` / `sigmoid`）、
逐 head 矩阵状态递推（rank-1 校正）、group-norm、channel-mix、终态归一、LM head
全部在 C 内完成，一步一次调用。权重由 `experiments/_export_rwkv7.py` 从 GGUF 导出为
打包 fp32 blob，矩阵存 `(in, out)` 配合 GEMV。

数值一致性（`experiments/_rwkv7_c_equiv.py`，与 numpy 参考逐步比对）：

| 口径 | 结果 |
|---|---|
| 单步 logits max\|diff\| | 7.25e-05（bf16 精度内） |
| argmax 一致 | 12 / 12 |
| 120 步长程 argmax 一致 | 120 / 120 |

解码速度（`experiments/_rwkv7_c_bench.py`，单序列、已建状态后单步、每格独立子进程、
5 轮交错、60 步取中位）：

| 实现 | ms/step | 相对 numpy |
|---|---|---|
| numpy 参考 | 366.36 | 1.00× |
| C 内核 1 线程 | 67.92 | 5.39×（5/5 逐轮占优） |
| C 内核 4 线程 | 52.26 | 7.01× |
| C 内核 8 线程 | 48.88 | 7.49× |

C 内核 8 线程对 1 线程再快 1.39×（5/5 逐轮占优）。单线程方差偏大（50–84 ms），
绝对值随机器负载浮动，比值与逐轮符号检验一致。

CLI：`--gguf-backend c` 走 C 内核（需先运行 `_export_rwkv7.py` 导出 blob）：

```bash
python -X utf8 experiments/_export_rwkv7.py \
    experiments/gguf/rwkv7-191M-world-bf16.gguf \
    experiments/gguf/rwkv7-191M-world-bf16.runtime.bin
python -m sunnynet.cli --gguf experiments/gguf/rwkv7-191M-world-bf16.gguf \
    --gguf-backend c --prompt "The Eiffel Tower is located in" --max-new 40 --stats
```

### INT4 权重量化（`sunnynet/runtime/quant.c`）

分组对称 INT4：每 64 个连续权重共享一个 fp32 scale，量化值域 [-8, 7]，两值一字节。
量化 GEMV 把激活也按同样分组量化到 INT8，用 `maddubs` 做点积
（`Σ(qᵢ−8)xᵢ = Σqᵢxᵢ − 8Σxᵢ`），权重保持无符号 [0,15] 直接进第一操作数。
整模型 C 内核按矩阵标志位混用 fp32 / INT4（向量与嵌入表保持 fp32）。

导出：`experiments/_export_rwkv7.py`（不带 `--f32` 即 INT4，输出 `.q4.bin`）。

| 口径 | fp32 | INT4 |
|---|---|---|
| blob 体积 | 729.1 MB | 269.0 MB |
| 单步解码（单线程中位） | 49.16 ms | 17.35 ms |

INT4 端到端 **2.83×**（5/5 逐轮占优），体积 2.7× 压缩。整数路径与显式整数公式
**逐位一致**（`tests/test_quant_i4.py`）；对 fp32 GEMV 的相对误差 ~0.11（4-bit 预期内）。
生成质量：英文保持连贯（"Paris, France…"），中文与代码有退化——**4-bit 欲压质量须配
校准/混合精度**，此处只发布"能跑且快"的结论，不声称质量无损。

CLI：`--quant int4`（配合 `--gguf-backend c`）。

### 外部权重：Mamba-2 与 DeltaNet（`sunnynet/runtime/mamba2.py`、`deltanet.py`）

同一套 Runtime 接口加载另外两种线性架构的官方权重，逐 token 单步递推，不经 PyTorch 前向。

**Mamba-2**（`state-spaces/mamba2-130m`，d_model 768 / 24 层 / 24 head × 64）：

```bash
python -m sunnynet.cli --hf-dir <模型目录> --hf-arch mamba2 \
    --tokenizer-dir <GPT-NeoX tokenizer 目录> --prompt "def fibonacci(n):" --stats
```

每个 head 状态是 `(head_dim=64, state_size=128)` 矩阵，但衰减 `dA = exp(dt·A)` 对
`(p, n)` 全同，外积也可分解，实际退化为元素级一阶递推：

```
dA  = exp(dt_h · A_h)                    # 标量
S_t = S_{t-1} ⊙ dA + (dt_h · B_g) ⊗ x_h  # 无需矩阵-向量乘积
```

- 与 `transformers` 的 `Mamba2ForCausalLM` 整段前向 max|diff| **5.9e-04**、argmax **24/24**。
  官方 fp32 自身相对 fp64 偏离 **3.8e-04**，本实现偏离 **2.5e-04**（更接近 fp64 真值）
  ⇒ 差异是单精度累积，非公式错误。
- 分段续算 vs 全程 **逐位 0.0**；贪心续写英文与代码连贯。

**DeltaNet**（`fla-org` 的 DeltaNet 340M，hidden 1024 / 24 层 / 8 head × 128，短卷积 4）：

```bash
python -m sunnynet.cli --hf-dir <模型目录> --hf-arch deltanet \
    --prompt "The capital of France is" --stats
```

真 delta rule，状态 `(k_dim=128, v_dim=128)`，更新含 rank-1 校正：

```
δ_t = (v_t − S_{t-1}ᵀ k_t) · β_t
S_t = S_{t-1} + k_t ⊗ δ_t
o_t = S_tᵀ q_t
```

- 秩 1 形式与显式 `(I − β k kᵀ) S + β k vᵀ` 写法的独立实现 **max|diff| 4.6e-05**
  （fp32 累积量级）。注意 `head_k_dim == head_v_dim` 时把 `S` 乘错一侧不会报维度错，
  只会静默算成另一个式子，故两种写法必须分开验证。
- 分段续算 vs 全程 **逐位 0.0**；贪心输出
  `"<s>The capital of France is the city of Paris. It is the largest city in Europe…"`。

**速度（单线程 numpy 参考路径，每格独立子进程 × 5 轮）**：

| 架构 | median ms/token | min | max | 状态 |
|---|---|---|---|---|
| mamba2 | 45.68 | 38.27 | 55.42 | 元素级递推，**可搬进 C 内核复用 `recur.c`** |
| deltanet | 91.46 | 83.93 | 99.19 | 矩阵值 rank-1 递推，**现有内核无法表达** |

这是参考路径的绝对耗时，**不与任何基线比倍数**。两者的单步安全上界都是
`O(heads × head_dim × state_size)`，与上下文长度无关。

**矩阵值递推是唯一尚未进 C 的层级**：RWKV-7 与 DeltaNet 的状态更新都含
`S ← S + k ⊗ (·)` 这类外积累加，`SN_FMA`/`SN_GATED` 的逐通道形式表达不了，
需要新增一个 rank-1 递推内核；Mamba-2 与 GLA 因衰减外积可分解，不需要。

**GLA**（`fla-hub/gla-340M-15B`，hidden 1024 / 24 层 / 4 head × 128，gate_rank 16）：

```bash
python -m sunnynet.cli --hf-dir <模型目录> --hf-arch gla \
    --prompt "The capital of France is" --max-new 24 --stats
```

GLA 的门控 `gk = logsigmoid(·) / 16` 只作用在 key 维，衰减可分解为按行缩放，
状态递推退化为元素级，与 Mamba-2 同类：

```
S_t = diag(exp(gk_t)) · S_{t-1} + k_t ⊗ v_t
o_t = scale · S_tᵀ q_t,  scale = head_k_dim^{-1/2}
```

读出后接 `RMSNorm(o) · silu(g)`（先归一后乘门，`norm_before_gate=False`）。

- 与独立参考实现 max|diff| **2.05e-05**，argmax **12/12**；分段续算 **逐位 0.0**。
- 贪心续写 `"The capital of France is the city of Paris."`；`--stats` 下格确认。
- 衰减可分解性由测试直接断言（`test_gla_decay_is_elementwise_decomposable`），
  防止后续误当成矩阵值递推。

### OpenAI 兼容 HTTP 服务（`sunnynet/serve.py`）

零依赖实现（标准库 `http.server.ThreadingHTTPServer`，不引入 fastapi/flask），
任何 OpenAI SDK 或客户端可直接指向本服务：

```bash
sunnynet-serve --ckpt <checkpoint.pt> --host 127.0.0.1 --port 8080
sunnynet-serve --gguf <model.gguf> --gguf-backend c --port 8080   # RWKV-7
sunnynet-serve --hf-dir <模型目录> --hf-arch gla --port 8080        # GLA
```

| 端点 | 说明 |
|---|---|
| `GET /v1/models` | 模型列表（`object=model`，`owned_by=sunnynet`） |
| `POST /v1/chat/completions` | 支持 `stream: true`（SSE `data: {...}` + `data: [DONE]`） |
| `POST /v1/completions` | 传统补全 |
| `GET /health` | 存活探针 |

后端选择走 `Engine` 抽象：`HFEngine` / `GGUFEngine` / `GGUFCEngine` / `TorchEngine`，
采样（temperature / top-k / top-p / repetition-penalty）在服务侧统一处理。

### 子命令与 Ollama API 服务

`cli.py` 在主接口之外提供子命令；`-m/--model` 按后缀自动识别 `.gguf` / `.pt` / HF 目录：

```bash
sunnynet chat   -m model.gguf --system "..." --ctx 8      # 交互式，流式
echo "Once upon a time" | sunnynet chat -m model.gguf     # 管道单轮
sunnynet serve  -m model.gguf --port 8080
sunnynet convert <src.gguf> <dst.gguf> --arch rwkv7        # 加 sunnynet.* 标记
sunnynet quantize <src.gguf> <dst.gguf> --type q4_0 --ppl  # 量化 + 困惑度
sunnynet info   -m model.gguf                              # 架构/层数/词表/张量数
```

`chat` / `serve` 默认走 C 内核（`--gguf-backend c`），首次调用自动导出 blob 并缓存。

`sunnynet/ollama.py` 提供 Ollama **原生** API（不是 OpenAI 形状），已有用 Ollama API 的
应用把 `OLLAMA_HOST` 指过来即可：`GET /api/version`、`GET /api/tags`、
`POST /api/generate`、`POST /api/chat`，流式走 NDJSON。

本地实测（官方 `openai` SDK 3.36.0 与裸 HTTP 双路径）：四个 OpenAI 调用与五个 Ollama
端点全部 200；非法 `temperature` 回 400 而非断连。GGUF→C 内核单步
numpy fp32 530.58 / C fp32 86.55 / C INT4 28.21 ms/token（191M 测试件，B=1）。
待真机复现项见 [docs/cn/ROADMAP.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/cn/ROADMAP.md)。

实测（本地启动后）：官方 `openai` SDK 的 `models.list`、chat 非流式、chat 流式、
legacy completions 四种调用全部通过；400（缺字段）/ 404（未知路径）错误处理正常。

Docker 镜像（`Dockerfile`，`python:3.11-slim`）：构建阶段装 gcc 把内核编成
Linux `.so`（PyPI 的 wheel 只带 win-x86_64 预编译库，Linux 侧一律源码编译），
运行层只留 `libgomp1`，不带编译器。权重在运行期挂载：

```bash
docker build -t sunnynet .
docker run --rm -p 8080:8080 -v "$PWD/sunnynet-small-zh:/models" sunnynet \
    --ckpt /models/sunnynet-small-zh.pt --tokenizer /models/tokenizer.json \
    --host 0.0.0.0 --port 8080
```

打 tag 后由 `.github/workflows/docker.yml` 自动构建并推送
`ghcr.io/sunnyclaw-cx/sunnynet`（仅 linux/amd64，内核指令集为 x86-64-v3）。

### Ollama / LM Studio 接入（`sunnynet/integrate.py`）

两者都跑 llama.cpp，接入前提是 GGUF 且架构被内置运行时支持。RWKV-7 GGUF 的
`general.architecture=rwkv7`、`tokenizer.ggml.model=rwkv`，Ollama 直接识别
（`family=rwkv7`、`parameter_size=191.13M`、`runner=llamacpp`）。

```bash
# 生成 Modelfile（默认只打印命令；--run 才真正导入）
sunnynet-integrate ollama <model.gguf> --name sunnynet-rwkv7 --run --smoke
# LM Studio：GGUF 放模型目录 + 生成接入配置
sunnynet-integrate lmstudio --gguf <model.gguf> --models-dir <LM Studio models>
```

实测导入与生成：

```bash
OLLAMA_MODELS=<models> OLLAMA_HOST=127.0.0.1:11511 ollama serve &
sunnynet-integrate ollama <model.gguf> --name sunnynet-rwkv7 --run --smoke
# [smoke] OK: 'The capital of France is Paris.'
```

三个必须记录的坑：

1. **必须显式写 TEMPLATE**。Ollama 内置的 `rwkv-world` 模板会在 prompt 外再包一层
   `User: / Assistant:`，输出退化成复读（`-of-the-world-of-the-world-...`）。
   写 `TEMPLATE """User: {{ .Prompt }}\n\nAssistant:"""` 后输出恢复
   `" The capital of France is Paris."`，与自家 C runtime 逐字一致。
2. **`Error: could not locate ollama app` 的真实含义是 server 未运行**。
   `ollama create` 在 server 离线时会尝试拉起本地 app 并失败，报错文本具有误导性。
   本工具先探活 `GET /api/version`，离线时直接给出 `ollama serve` 提示。
3. **`OLLAMA_MODELS` 必须与 server 侧一致**。CLI 只设 `OLLAMA_HOST` 时 `create`
   会把 blob 写进默认模型目录，与运行中的 server 目录错位。
4. 本地探活须绕开系统 HTTP 代理（企业代理会把 `127.0.0.1` 拦成 502），
   代码内用 `ProxyHandler({})` 构建 opener。

LM Studio 无需 GGUF 转换的另一条路：把 `sunnynet-serve` 配成自定义 OpenAI 兼容
endpoint（`Developer` → 添加 endpoint），配置由 `sunnynet-integrate lmstudio` 生成。

### AVX-512 与 NEON 内核（`sunnynet/kernels/scan_avx512.c`、`core_avx512.c`、`step_avx512.c`、`scan_neon.c`）

AVX2/FMA 之外补两条 SIMD 后端源码，均含前向扫描、反向扫描、单步三个入口。

AVX-512（F/BW/VL）拆成 `scan_avx512.c`（512 位扫描）与 `core_avx512.c` / `step_avx512.c`
（融合内核与推理态单步）：一个 512 位寄存器装**两个 group（2×8 lane）**，组内相邻元素用
16 lane 的 `vpermps` 做 lane 置换，两半区互不干扰；`D` 不是 16 倍数时用掩码处理尾部。
导出 CPUID 探测（leaf 7 EBX bit16）与 ISA 字符串。**本机 CPU 实测 AVX512F=0**
（i7-8750H，Coffee Lake），故只做编译与产物验证：

| 验证项 | 结果 |
|---|---|
| 编译 | `gcc -O3 -mavx512f -mavx512bw -mavx512vl -mfma -funroll-loops -fopenmp -shared` 通过 |
| 反汇编（`core_avx512.c` 单编） | 192 处 `zmm`、10 处 `vfmadd*ps`、12 处 `vpermps` |
| 反汇编（与 `step_avx512.c` 合编） | 315 处 `zmm`、44 处 `vfmadd*ps`、16 处 `vpermps`、33 条掩码访存 |
| 加载保护 | 不支持时 `load_avx512()` 直接抛错，避免非法指令 |

NEON：无 ARM 工具链、无 QEMU（`wsl.exe` 被安全策略拦），改用 `zig cc` 真交叉编译：

```bash
pip install ziglang
PYTHON=<venv python> bash scripts/build_arm.sh aarch64-linux-gnu
PYTHON=<venv python> bash scripts/build_arm.sh arm-linux-gnueabihf
```

| 验证项 | AArch64 | ARMv7 |
|---|---|---|
| 产物 | 22264 B，ELF64，`e_machine=0xb7` | 881104 B，ELF32，`e_machine=0x28` |
| 导出符号 | `scan_neon_f32` / `core_neon_fwd_f32` / `step_core_neon_f32` / `sn_neon_isa` 齐全 | 同左 |
| `__ARM_FEATURE_FMA` | 定义 ⇒ `SN_FMA` 走 `vfmaq_f32` | 未定义 ⇒ 回退 `vaddq_f32+vmluq_f32`（`vmla.f32`） |

**本机既无 AVX-512 CPU 也无 ARM 设备**，改用同名标量 shim 在 x86 上执行**真内核源码**：
`experiments/_avx512_shim/immintrin.h` 把 `__m512` 实现成 16 元数组、
`_neon_shim/arm_neon.h` 把 `float32x4_t` 实现成 4 元数组，组内置换仍用源码里的真索引向量。

| 内核 | 与 torch 参考 max\|diff\| |
|---|---:|
| AVX-512 core fwd（D=32，16 宽主循环 / D=24，8 宽尾部） | 1.19e-07 / 5.96e-08 |
| AVX-512 step core（D=32 / D=24） | 1.19e-07 / 5.96e-08 |
| AVX-512 dw step | 0.0 |
| NEON scan | 0.0（逐位相同） |
| NEON core fwd / step core | 1.19e-07 / 1.19e-07 |

**边界**：只排除「索引写错但数值仍在界内」这类缺陷（例如 VEX 半区索引不分区），
不证明真机浮点行为与吞吐。待真机复现项见 [docs/cn/ROADMAP.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/cn/ROADMAP.md) §3.4。

### 稳定 C ABI（`sunnynet/runtime/sn_abi.h`、`sn_abi.c`）

此前的 C 接口是内部用的，改结构即破坏调用方。新增一层稳定 ABI 隔离：

- 符号统一 `sn_abi_` 前缀，句柄不透明（`SnAbiRwkv7*`），调用方不依赖内部布局
- 版本 `SN_ABI_VERSION = (MAJOR<<16)|(MINOR<<8)|PATCH`，当前 **1.0.0**
- 能力位 `SN_CAP_*`、错误码 `SN_OK=0` / `SN_E_INVAL=-1` / `SN_E_UNSUPPORTED=-2` 等
- `sn_status` 统一为 `int32_t`，函数只返回状态码，不做异常抛出

ctypes 绑定在 `sunnynet/runtime/abi.py`，与内部句柄路径**逐位一致**
（`max|diff| = 0.0`）。契约测试 9 项全绿（版本协商、能力位、错误码、
`check(1)=OK` / `check(9)=E_UNSUPPORTED`、RWKV-7 步进等价）。详见 [docs/cn/ABI.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/cn/ABI.md)。

### 命令行

```bash
# torch 后端
python -m sunnynet.cli --ckpt experiments/ckpts_ood/sunnynet.pt \
    --prompt "子曰：" --max-new 80 --stats

# 纯 C 后端 (先导出权重 blob, 再解码)
python -X utf8 experiments/_export_runtime.py \
    experiments/ckpts_ood/sunnynet.pt experiments/ckpts_ood/sunnynet.runtime.bin
python -m sunnynet.cli --ckpt experiments/ckpts_ood/sunnynet.pt \
    --backend c --prompt "子曰：" --max-new 80 --stats

# 外部 GGUF 权重 (RWKV-7)
python -m sunnynet.cli --gguf experiments/gguf/rwkv7-191M-world-bf16.gguf \
    --prompt "The Eiffel Tower is located in the city of" --max-new 40

# 外部 HF 权重 (Mamba-2 / DeltaNet / GLA)
python -m sunnynet.cli --hf-dir <模型目录> --hf-arch mamba2 \
    --tokenizer-dir <GPT-NeoX tokenizer 目录> --prompt "def fibonacci(n):" --stats
python -m sunnynet.cli --hf-dir <模型目录> --hf-arch deltanet \
    --prompt "The capital of France is" --max-new 20 --stats
python -m sunnynet.cli --hf-dir <模型目录> --hf-arch gla \
    --prompt "The capital of France is" --max-new 24 --stats

# OpenAI 兼容服务
sunnynet-serve --ckpt <checkpoint.pt> --port 8080

# Ollama / LM Studio 接入配置
sunnynet-integrate ollama <model.gguf> --name sunnynet-rwkv7
sunnynet-integrate lmstudio --gguf <model.gguf>

# 权重发布导出 (safetensors + runtime blob + 模型卡 + sha256)
sunnynet-release <checkpoint.pt> --out-dir dist/release
```

加载入口也可直接从 py 调用：

```python
from sunnynet.runtime import load_model
m = load_model("deltanet", "<模型目录>")   # 或 "mamba2" / "rwkv7"
st = m.reset(m.init_state())
logits = m.step(token, st)                 # 每步 O(1)
```

### 路线

现有资产：原创架构、AVX2 融合内核（训练步 1.10–1.17×）、通用递推 C Runtime
（对向量化基线中位 9.9×）、**C 整模型单步解码（B=1 单线程 1.53–2.05×）**、
跨架构映射验证、**GGUF 只读 + RWKV-7 权重跑通（数值与官方 .pth 等价）**、
**RWKV-7 整模型 C 内核（对 numpy 5.39×，8 线程 7.49×）**、
**INT4 权重量化（端到端 2.83×，体积 2.7× 压缩）**、
**Mamba-2 / DeltaNet / GLA 官方权重接入（数值对齐参考实现，贪心生成连贯）**、
**AVX-512 与 NEON 内核源码（编译 + shim 执行等价验证）**、
**GGUF 一行命令走 C 内核（`sunnynet chat -m xxx.gguf`，blob 自动导出缓存）**、
**CLI 子命令（`chat` / `serve` / `convert` / `quantize` / `info`）**、
**GGUF 量化 + 困惑度评测（`sunnynet quantize`）**、
**OpenAI 兼容 HTTP 服务（`sunnynet-serve`）**、
**Ollama 原生 API 服务（`sunnynet/ollama.py`）**、
**Ollama / LM Studio 接入（`sunnynet-integrate`）**、
**C API 稳定 ABI（`sunnynet/runtime/sn_abi.h`，版本化 + 能力位 + 错误码）**。

近期：权重发布（`sunnynet-release`）。

中期：补矩阵值递推（外积 / rank-1 校正）内核，把 RWKV-7 与 DeltaNet 的单步也搬进 C
（Mamba-2 与 GLA 已可复用现有递推）。

不做的：复刻 llama.cpp 的通用张量算子层（GGML 为 Transformer 设计，逐算子调度）；
打平 Transformers / vLLM（分别为模型仓库与 GPU 服务赛道）。

各项验收口径、**待真机复现的性能项**与未做项见 [docs/cn/ROADMAP.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/cn/ROADMAP.md)。

---

## 仓库结构

```
sunnynet/
  ops.py          原创算子: poly_sigmoid / softwedge / CenteredScaleNorm /
                  多尺度衰减 / 对数域并行扫描 / 通用门控扫描
  layers.py       SunnyBlock: 状态更新单元 + 循环滤波 + 局部卷积 + 非线性支路
  kernels/        AVX2 手写内核 (scan / 融合 core / 推理态单步) + 自动编译加载
                  scan_avx512.c  AVX-512F/BW/VL 内核 (掩码尾处理 + CPUID 探测)
                  scan_neon.c    ARM NEON 内核 (zig 交叉编译 / CI aarch64 执行)
                  simd.py        AVX-512 / NEON 加载与交叉编译层
  linear_baselines.py  O(N) 同复杂度对照: 线性注意力 / RWKV / Mamba / Hyena / H3
  model.py        SunnyNet 主体 (状态 (B,L,D), 逐层接力)
  init.py         谱锚定初始化
  cli.py          命令行推理 (加载 ckpt -> 分词器 -> 自回归解码, --backend torch/c;
                  --gguf 走外部 RWKV-7 权重; --hf-dir 走 Mamba-2 / DeltaNet / GLA)
  serve.py        OpenAI 兼容 HTTP 服务 (标准库实现, --host / --port)
  integrate.py    Ollama / LM Studio 接入 (Modelfile 生成 + 探活 + GGUF 放置)
  release.py      权重发布导出 (safetensors + runtime blob + 模型卡 + sha256)
  runtime/        独立 C 推理 Runtime
    recur.c         通用一阶递推内核 (扫描 / 反向 / 单步 / INT8 量化)
    sunnynet.c      整模型单步解码 (B=1, 无中间张量)
    sn_abi.h/.c     稳定 ABI 层 (版本化 + 能力位 + 错误码 + 不透明句柄)
    abi.py          ABI ctypes 绑定
    model.py        ctypes 绑定
    gguf.py         GGUF 只读加载 (f16/bf16/f32 + 量化反量化)
    rwkv7.py        RWKV-7 单步递推 (numpy 参考路径, 读 GGUF 权重)
    rwkv7_c.py      RWKV-7 整模型 C 内核绑定
    mamba2.py       Mamba-2 单步递推 (衰减外积可分解, 元素级)
    deltanet.py     DeltaNet 单步递推 (真 rank-1 校正, 矩阵值)
    gla.py          GLA 单步递推 (门控只在 key 维, 元素级)
  baseline.py     同参数量 Transformer 对照 (仅用于对照, 非架构本体)
  tokenizer.py    从零实现的 BPE 分词器
  profiling.py    真实性能探针 (峰值内存 / 访存 / 吞吐)
train/
  train.py        统一训练器 (两种架构共用同一循环)
  evaluate.py     困惑度 / 长距离探针 / 生成样例
experiments/
  run_experiments.py   实验编排 (baseline / ablation / seeds / scale / probe)
  battery.sh           批量运行基线 + 消融 + 种子 + 缩放
  battery2.sh          20M 缩放对照 + 长序列训练对照
  long_sequence.py     前向延迟 vs 序列长度 (O(N) 验证) + 长序列训练
  make_figures.py      生成全部图表 -> docs/figures/
  collect_results.py   汇总所有结果为 Markdown 表
tests/
  test_architecture.py  架构自检 (分块一致性 / 因果性 / 梯度界 / 深度稳定, 63 项)
  test_runtime_model.py C 整模型解码一致性
  test_runtime_hf.py    Mamba-2 / DeltaNet / GLA 外部权重接入
  test_rwkv7_gguf.py    GGUF 加载 / 张量对齐 / RWKV-7 递推等价 / 分词往返
  test_abi.py           C API 稳定 ABI 契约 (版本 / 能力位 / 错误码 / 步进等价)
  test_simd.py          AVX-512 编译产物 + NEON 交叉编译与反汇编
  test_integrate.py     Ollama / LM Studio 接入 (Modelfile / 探活 / 目录布局)
  test_release.py       权重发布导出 (safetensors 往返 / blob 头 / sha256)
  test_smoke_pipeline.py 端到端冒烟 (合成语料 -> 训练 -> 评估)
prepare_data.py   真实语料采集
build_corpus.py   分词器训练 + token 流构建
docs/
  ARCHITECTURE.md  完整数学推导
  EXPERIMENT_REPORT.md 实测实验报告 (含负面结论)
  ABI.md           C API 稳定 ABI 的版本策略 / 能力位 / 错误码
  ROADMAP.md       路线边界、各项验收记录、待真机复现项
  figures/         实验图表
```

---

## 原创性说明

SunnyNet 为独立原创架构，同时有两点需要说明：

- 其状态递推在数学形式上与线性注意力类工作有亲缘性（都是一阶线性递推），这一点不回避；
- 门控构造（凸组合 + 对数均匀多尺度 + 逐元素保护门）、层间传递（状态接力 + 特征残差流）、
  激活函数、归一化、初始化均为自行设计与实测标定；
- 未参考或移植任何既有项目的代码与结构。

架构设计过程中有四处结论被实测推翻并修正：softmax 通道竞争导致梯度随深度归零；
谱半径层深衰减的符号设反；CenteredScaleNorm 的 `s_min` 高于实际激活尺度导致归一化退化为
常数缩放；原设计排除残差反而造成最大单项质量损失。修正依据与数据记录在
docs/cn/ARCHITECTURE.md 第 3/4/5 节。保留这些记录是为了说明各设计约束的来源。

---

## 风险与局限

1. **短序列劣势**：O(N) 优势只在长序列兑现。batch=1 前向交叉点 T≈384（6 层）/ T≈600（4 层），
   短于此慢于标准 Transformer。该架构不适合短序列场景。
2. **短序列下质量仍不及 H3**：T=256、600 步同协议对比中 ppl 57.83，不及 H3 53.93（差约 7%）。
   该对比未收敛、单种子、学习率未逐架构调优，只能读作早期样本效率排序。
3. **长序列对同行也无绝对优势**：T=4096 时 Hyena 仍快 1.9×、H3 基本追平；不存在"领先全部
   同行"的长度区间。训练态 T=1024 下速度第 1、内存第 2 低档——前向与训练态排名不同，引用须带条件。
4. **早期耗时数据失真**：未做进程隔离时同进程累积构建会使读数偏高达 2 倍以上，且放大倍数随架构而变。
5. **长距离依赖**：滚动状态的信息容量受限于固定维度，理论上界低于全注意力。大跨度通道只保证
   "不被遗忘"，不保证"被检索"。
6. **伪收敛**：loss / ppl 下降不等于具备推理与指令跟随能力。
7. **语料覆盖**：中文维基百科在本机网络不可达，缺少"百科"域，实验只覆盖 古籍 / 英文文学 / 代码 三个真实域。
8. **规模**：已完成 1.4M / 20M / 100M 实测。0.5B 只有前向与单步 SGD 可跑（GPU 6GB 边界态），
   多步 loss 无下降趋势，没有任何 0.5B 质量数据。
9. **消融未成结论**：600 步未收敛，关组件后 ppl 反而更低，不构成组件有效性证据。
10. **同复杂度对比未收敛**：七架构对比仅 600 步，且未逐架构调优学习率，架构间小差距落在种子噪声内。
11. **`torch.compile` 不可用**：本机 MSVC 缺 Windows SDK 头文件，Inductor 的 Windows 分支硬编码
    要求 MSVC 风格命令行；GPU 侧亦无 nvcc / CUDA toolkit / triton。手写内核绕开 Inductor，已独立打通。
12. **GPU 结论的边界**：加速比是未做 CUDA 侧优化的下界；无同行 GPU 对照，不构成"GPU 上性能竞争力"的证据。

详见 **[docs/cn/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/cn/EXPERIMENT_REPORT.md)** 与 **[safety_notice.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/safety_notice.md)**。
模型为实验科研项目，中医相关输出不可作为医疗诊断依据。

---

## 赞助

<a href="https://afdian.com/a/sunny-cx">
  <img src="https://img.shields.io/badge/爱发电-赞助支持-946CE6?style=for-the-badge" alt="爱发电">
</a>

---

## 许可证

[Apache License 2.0](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/LICENSE)

