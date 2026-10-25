# TinyStories 训练质量诊断

现象：在微小故事数据集上训练 SunnyNet，走到 20000 步后生成仍不能正确响应。
怀疑点在架构或配置。

结论：**架构不是根因。** 定位到两处可复现问题，一处是代码缺陷，一处是训练预算。

## 1. 根因一：BPE 训练器的 pair 计数缺陷（代码缺陷，已修）

`sunnynet/tokenizer.py` 的 `BPETokenizer.train` 在贪心合并时维护 pair 计数：

- **累加**按真实出现次数：每出现一次加 `freq`；
- **递减**却先 `set(zip(ids, ids[1:]))` 去重，每个不同 pair 只减一次 `freq`。

高频 pair 的计数因此永远减不到零。堆顶反复弹出同一个 pair，被反复写入 `merges`，
每次生成一个新 id——**但这些 id 解码后是同一串字符**。

### 影响量化

| 分词器 | merges | 互不相同的 pair | 词表槽位浪费 |
|---|---|---|---|
| 项目多域 `data/proc/tokenizer.json` | 4126 | 368 | 45.9% |
| Kaggle 英文 `p100m_en_tokenizer.json` | 8069 | **328** | **94.5%** |
| 修复后 | 8099 | 8099 | 0% |

已发布的 `CodeDoes/SunnyNet-P100M-Kaggle`（102M 参数）用的正是中间那个分词器：
8192 路 softmax 里**只有 447 个不同的可学输出（5.5%）**，字符 `ti` 被 4129 个不同
id 表示、`re` 被 3199 个表示。模型被迫花容量去记住"这一批 id 都输出同一个字符串"。

英文压缩率因此退化：**1.54 chars/token**（正常英文 BPE 约 4）。

### 修复

计数改按出现次数增删（`Counter`），并加"确有可合并处"校验，防止计数残留导致死循环。
回归测试 `tests/test_tokenizer_train.py` 锁定三条不变式：merge 互不相同、
无解码等价的重复 token、压缩率下限。

## 2. 验证：同预算受控对照

TinyStories-valid 按故事级 92/8 切分，1.41M 参数（`small` 预设）、3000 步、
B=8、T=256 = 6.14M token 训练。评测用 **BPC（bits-per-character）**——它按原始
字符数归一化，是跨分词器唯一可比的指标。

| 分词器 | chars/token | val ppl | **BPC（越低越好）** |
|---|---|---|---|
| 坏（旧） | 1.54 | 4.304 | 1.3386 |
| 修复 | 2.31 | 6.111 | **1.1458（改善 14.4%）** |

注意 ppl 一项不可跨分词器直接比较（词表粒度不同，分母不同）；BPC 才是结论依据。

两格的生成文本都已通顺（"Once upon a time, there was a little girl named Lily…"），
说明**分词器缺陷是真实问题，但不是"学不会"的唯一原因**。

## 3. 根因二：训练预算差一个数量级

公开的 TinyStories 小模型都以 **tokens seen** 计量，不是步数：

| 模型 | 参数量 | tokens seen |
|---|---|---|
| karpathy `stories15M` | 15.2M | **39.0B** |
| `child-12m` | 12.3M | ~1.0B |
| `syrinx-10m` | 11.8M | 0.29B |
| `Compactbot/tinystories-24m` | 24.6M | 0.45B |

karpathy 一行为本次实测：从其 checkpoint config 读出 128×4×256×298000 = **39.0B**，
不是早前估值。其余三行来自各自 model card，未逐条核验。

对照报告的场景：20000 步 × B8 × T256 = **41M token**（1.41M 模型上约 29 token/参数）。
即使按 B16/T512 也只有 164M token，与 `stories15M` 的 39.0B 差两个数量级以上，
与 TinyStories 论文系列（单张 V100 ≤30 小时）也差一个数量级。
**这是"两万步仍不会写故事"的另一半原因。**

同预算下其他架构也做不到长距连贯——对照实验（`experiments/_tiny_bench.py`）
见下节。

## 4. 跨架构对照（同数据 / 同分词器 / 同步数）

TinyStories-valid 故事级切分、修复后分词器、3000 步 × B=8 × T=256 = 6.14M token、
栈参数对齐 ~365k（`small` 预设 1.41M 总参数）。指标 BPC。七格同一学习率 3e-3、
同一 warmup 与 cosine 调度，无架构专属超参。

| 架构 | 总参数量 | val ppl | **BPC** | tok/param |
|---|---|---|---|---|
| RWKV | 1,158,144 | 5.868 | **1.1251** | 5.31 |
| Mamba | 1,088,032 | 5.963 | 1.1350 | 5.65 |
| **SunnyNet** | 1,414,420 | 6.111 | 1.1458 | 4.34 |
| H3 | 1,206,608 | 6.239 | 1.1584 | 5.09 |
| Hyena | 1,286,992 | 6.362 | 1.1750 | 4.77 |
| Transformer | 1,270,320 | 7.161 | 1.2478 | 4.84 |
| Linear-Attn | 1,160,544 | 7.992 | 1.3076 | 5.29 |

读数：

- **SunnyNet 位列第三**，落后 RWKV 1.8%、Mamba 0.9%，领先 H3 1.1%、Hyena 2.5%。
  该排序经第二种子复核后稳定（§4.1）。
- **在同预算同栈参数下优于 Transformer**（1.1458 vs 1.2478，低 8.2%）——
  Transformer 在此规模反而排在 SunnyNet 之后，与"线性架构打不过注意力"的直觉相反；
  该差异来自 6.14M token 的极低预算，不代表大规模结论。
  两处不对称都偏向 Transformer：它用 6 层（SunnyNet 4 层），且压到 4 层后更差（1.3106）；
  它与 SunnyNet 的栈参数对齐到 0.06% 以内。
- **七个架构用同一个学习率（3e-3）**，无架构专属超参。Transformer 曾单独用 6e-3，
  实测把它换成与其余六格一致的 3e-3 后其 BPC 由 1.2692 降到 **1.2478**——
  即特殊学习率对 Transformer 不利，统一后 SunnyNet 的相对领先由 9.7% 收窄为 8.2%。
  SunnyNet 侧对学习率不敏感（3e-3 与 6e-3 相差 0.3%）。两格各为独立 3000 步运行。
- **SunnyNet 的 tok/param（4.34）低于 rwkv/mamba（5.31/5.65）**，
  即它在更少的 token 预算下取得了相近的 BPC。

### 4.1 中段排名的种子稳定性

主表每格是单种子读数，而中段相邻架构的 BPC 间距只有 0.9%–1.6%，排序可能落在种子噪声内。
给中段四架构各补跑第二个种子（seed=7，统一 lr=3e-3）：

| 架构 | seed 1234 | seed 7 | 均值 | 极差 |
|---|---|---|---|---|
| RWKV | 1.1251 | 1.1170 | 1.1210 | 0.72% |
| Mamba | 1.1350 | 1.1282 | 1.1316 | 0.60% |
| **SunnyNet** | 1.1458 | 1.1389 | **1.1423** | 0.60% |
| H3 | 1.1584 | 1.1541 | 1.1562 | 0.37% |

- 每种架构两种子的极差 **≤0.72%**，小于相邻架构的均值差（0.93%–1.20%）。
- 相邻架构的两种子取值区间**无重叠**：RWKV 上限 1.1251 < Mamba 下限 1.1282；
  Mamba 上限 1.1350 < SunnyNet 下限 1.1389；SunnyNet 上限 1.1458 < H3 下限 1.1541。
- 结论：中段排序在种子扰动下稳定，**SunnyNet 位列第三**不是单种子的偶然。
- 口径边界：两种子只给出极差，**不构成统计置信区间**；Hyena / Transformer / Linear-Attn
  未补第二种子，其与 H3 的间距（≥1.6%）未做同样的稳定性检验。

**口径边界**：七格训练到 6.14M token 即停止，远未收敛；只作同预算下的
相对定位，不得外推为最终质量。跨架构参数按栈参数对齐（非 embedding），
中段四架构各有两个种子（§4.1），其余三格单种子。

## 5. 与公开 TinyStories 模型的对照口径

公开模型的数字**不能直接与本项目对比**，因为三者不同：分词器不同、
数据切分不同、训练预算差一个数量级。

| 模型 | 参数量 | tokens seen | tokenizer | val ppl |
|---|---|---|---|---|
| karpathy `stories15M` | 15.2M | **39.0B** | Llama-2 32K BPE | 2.92 |
| `child-12m` | 12.3M | ~1.0B | 8K custom BPE | 3.45 |
| `pluto-15M` | 15M | ~1B | — | ~3.64 |
| `syrinx-10m` | 11.8M | 0.29B | 32K byte-BPE | 4.78 |
| `Compactbot/tinystories-24m` | 24.6M | 0.45B | 8K BPE | 8.76 |
| **本项目 SunnyNet（本次对照）** | 1.41M | **0.006B** | 8K BPE | 6.111 |

karpathy 一行为实测（checkpoint config），其余来自各自 model card。上表最后一行与
其余各行**不在同一预算量级**（与 karpathy 差 6000× 以上，与其余差 50–170×），列于此只为说明
差距来源。跨分词器比较 ppl 无效，必须转 BPC 或 bits-per-byte。

## 6. 建议

1. **重训分词器**：升级后重新跑 `build_corpus.py`，旧 token 流与旧 checkpoint 的
   embedding 都不再可用（词表语义已变）。这会同时影响项目内所有模型。
2. **重训公开的 p100m checkpoint**：它受缺陷影响最重（94.5% 浪费）。
3. **把步数换成 token 预算**：目标先定 **≥10 token/参数**，故事级任务建议
   ≥100 token/参数。TinyStories 全集约 470M token，小模型充分训练需要多 epoch。
4. **上下文**：TinyStories 故事平均约 200 词，T=256 够用，但配合修复后的分词器
   实际覆盖约 500 字符，信息量翻倍。

## 7. 复现

```sh
# 缺陷量化
python -X utf8 -c "..."   # 见 tests/test_tokenizer_train.py 的不变式

# 受控对照
python -X utf8 experiments/_tiny_bench.py --tok broken_en --arch sunnynet --steps 3000
python -X utf8 experiments/_tiny_bench.py --tok fixed_en  --arch sunnynet --steps 3000

# 跨架构
python -X utf8 experiments/_tiny_bench.py --tok fixed_en --arch {transformer,rwkv,mamba,linear_attn,hyena,h3}

# 学习率敏感性（--force-lr 覆盖按架构的默认值，七格统一 3e-3）
python -X utf8 experiments/_tiny_bench.py --tok fixed_en --arch transformer --force-lr 3e-3

# 层数敏感性（--base-layers 让 transformer 与 SunnyNet 同为 4 层，自动加宽对齐栈参数）
python -X utf8 experiments/_tiny_bench.py --tok fixed_en --arch transformer --base-layers 4

# 中段排名的种子稳定性（每格独立子进程串行跑，避免线程超订 segfault）
for a in rwkv mamba sunnynet h3; do
  python -X utf8 experiments/_tiny_bench.py --tok fixed_en --arch $a --seed 7 --steps 3000 \
      --out results/tiny/ms_${a}_s7.json
done
python -X utf8 experiments/_tiny_seed_agg.py
```

## 8. 口径边界

- 上述对照训练到 6.14M token，**远未收敛**；只读作同预算下的相对优劣，
  不得外推为最终质量。
- BPC 依赖字符数统计，不同数据切分会有微小差异；本次两格用同一验证集。
- 跨架构对照的参数按"栈参数"对齐（非 embedding），层数未强制一致：
  Transformer 为 6 层，SunnyNet 与其余线性架构为 4 层。该不对称**偏向 Transformer**——
  实测把 Transformer 压到 4 层（栈参数对齐后自动加宽到 d120），其 BPC 由 1.2478 退化到
  **1.3106**，即 6 层对它更有利。它仍在同预算下落后，故 SunnyNet 的领先不依赖层数占优。
- 中段四架构（RWKV/Mamba/SunnyNet/H3）各跑两个种子，极差 ≤0.72%（§4.1）；
  其余三格仍为单种子。两种子只给极差，不构成统计置信区间。
- 七格用同一学习率，但**各架构的最优学习率未逐格搜索**；表中数字是"同一超参下的"
  相对表现，不是各架构调参后的上限。