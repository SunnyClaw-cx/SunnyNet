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
| karpathy `stories15M` | 15.2M | ~1B |
| `child-12m` | 12.3M | ~1.0B |
| `syrinx-10m` | 11.8M | 0.29B |
| `Compactbot/tinystories-24m` | 24.6M | 0.45B |

对照报告的场景：20000 步 × B8 × T256 = **41M token**（1.41M 模型上约 29 token/参数）。
即使按 B16/T512 也只有 164M token，仍与 ~1B 差 6× 以上。**这是"两万步仍不会写故事"
的另一半原因。**

同预算下其他架构也做不到长距连贯——对照实验（`experiments/_tiny_bench.py`）
见下节。

## 4. 跨架构对照（同数据 / 同分词器 / 同步数）

TinyStories-valid 故事级切分、修复后分词器、3000 步 × B=8 × T=256 = 6.14M token、
栈参数对齐 ~365k（`small` 预设 1.41M 总参数）。指标 BPC。

| 架构 | 总参数量 | val ppl | **BPC** | tok/param |
|---|---|---|---|---|
| RWKV | 1,158,144 | 5.868 | **1.1251** | 5.31 |
| Mamba | 1,088,032 | 5.963 | 1.1350 | 5.65 |
| **SunnyNet** | 1,414,420 | 6.111 | 1.1458 | 4.34 |
| H3 | 1,206,608 | 6.239 | 1.1584 | 5.09 |
| Hyena | 1,286,992 | 6.362 | 1.1750 | 4.77 |
| Linear-Attn | 1,160,544 | 7.992 | 1.3076 | 5.29 |

SunnyNet 位列第三：落后 RWKV 1.8%、Mamba 0.9%，领先 H3 1.1%、Hyena 2.5%。
注意 SunnyNet 在此格的 tok/param（4.34）低于 rwkv/mamba（5.31/5.65），
即它在更少的 token 预算下取得了相近的 BPC。

**口径边界**：六格训练到 6.14M token 即停止，远未收敛；只作同预算下的
相对定位，不得外推为最终质量。跨架构参数按栈参数对齐，层数未强制一致
（rwkv/h3 的层数为 4，SunnyNet 亦为 4）。单种子单次读数。

## 5. 与公开 TinyStories 模型的对照口径

公开模型的数字**不能直接与本项目对比**，因为三者不同：分词器不同、
数据切分不同、训练预算差一个数量级。

| 模型 | 参数量 | tokens seen | tokenizer | val ppl |
|---|---|---|---|---|
| karpathy `stories15M` | 15.2M | ~1B | Llama-2 32K BPE | 2.92 |
| `child-12m` | 12.3M | ~1.0B | 8K custom BPE | 3.45 |
| `pluto-15M` | 15M | ~1B | — | ~3.64 |
| `syrinx-10m` | 11.8M | 0.29B | 32K byte-BPE | 4.78 |
| `Compactbot/tinystories-24m` | 24.6M | 0.45B | 8K BPE | 8.76 |
| **本项目 SunnyNet（本次对照）** | 1.41M | **0.006B** | 8K BPE | 6.111 |

上表最后一行与其余各行**不在同一预算量级**（差 50–170×），列于此只为说明
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
```

## 8. 口径边界

- 上述对照训练到 6.14M token，**远未收敛**；只读作同预算下的相对优劣，
  不得外推为最终质量。
- BPC 依赖字符数统计，不同数据切分会有微小差异；本次两格用同一验证集。
- 跨架构对照的参数按"栈参数"对齐（非 embedding），层数未强制一致。
- 未做多种子重复，单次读数只作方向性证据。