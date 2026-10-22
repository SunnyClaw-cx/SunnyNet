# 外部质疑的逐条核查

针对一组外部批评（短序列倒挂 / Scaling Law 失效 / 长距离检索弱 / 随机数据集 /
多种子泛化差 / T=1024 最慢且内存最高），用本仓库既有实测数据与新增实测逐条对照，
分「成立」「不成立」「证据不足」三类。

新增脚本与数据：`experiments/_T1024_audit.py`、`prepare_extra.py`、`experiments/_ood.py`。

---

## 一、逐条判定

| 质疑 | 判定 | 依据 |
|---|---|---|
| T<384 慢于 Transformer | 成立，但仅 batch=1 前向口径 | §2.4 交叉点 T≈384；训练态（batch=8）交叉点下移到 T<256（§19） |
| 因此限制在常规 NLP 任务 | 不成立（前提错） | 短句分类/对话回复属训练/微调场景；训练态 T=256/384 已快 1.5–2.4×（§19） |
| Scaling Law 失效 | 证据不足 | 本仓库只测过 20M / 100M 两点（§14.6 / §14.8），没有 scaling 曲线，判不了"失效" |
| 长距离检索弱 | 对代码域不成立、对自然语言域成立 | §8：代码域 far 桶命中 0.1208，与 Transformer 一致；自然语言域接近 0 |
| 用随机数据集测 | 只对速度基准成立，对质量结论不成立 | 速度脚本用 `torch.randint`（计算图与数值无关）；质量对照走 `DomainMixer`→真实 token 流（§3/§10/§12） |
| 多种子方差大 | 成立，且项目已知 | §6：种子 ppl 极差 6.32 > 架构差 3.53；§10.5 已声明单种子不可判 |
| 跨数据集泛化差 | 不成立 | 中文新域第 1/7、英文新域第 2/7、OOD 落差最低档（§18） |
| T=1024 速度最慢 | 前提相关：batch=1 前向排第 5（不是最慢，TF 垫底）；训练态排第 1 | §10.6 / §17 |
| T=1024 峰值内存最高 | 不成立 | §17：SunnyNet 1812 MB 排第 2 低档，比 Transformer 省 43% |

---

## 二、T=1024 训练态速度与内存（§17）

既有 `on_scaling_t8.json` 只有 batch=1 纯前向，且从未记录内存。补测训练态：

速度（ms/token）：sunnynet **0.13186（第1）** · h3 0.13190 · mamba 0.14721 ·
hyena 0.16754 · linear_attn 0.17646 · rwkv 0.19677 · transformer 0.55693

峰值内存（MB）：transformer 3192 · **sunnynet 1812（第2）** · hyena 1809 ·
mamba 1780 · linear_attn 1777 · h3 1689 · rwkv 1611

⇒ 训练态下两个指控都不成立。前向与训练态排名不同（反向占训练步 56.9%，§13.3）。

---

## 三、短序列倒挂的训练态核查（§19）

按训练态（batch=8，含反向）复测三配置 × 三长度 × 3 轮：

| 配置 | T=128 | T=256 | T=384 |
|---|---|---|---|
| `small`（旧 d128）相对 TF 比值 | 1.104 | 0.672 | 0.581 |
| `smalln80`（窄，§16）相对 TF 比值 | 1.045 | 0.509 | 0.425 |

（比值 <1 = 比 Transformer 快）

- 交叉点从 T≈384（前向）下移到 T<256（训练态）：T=256/384 时已快 1.5–2.4×。
- T=128 仍慢：旧配置 10.4%、窄配置 4.5%（落在内存高压噪声带内）。
- 有效手段是压激活宽度（窄残差流）：T=128 劣势从 10.4% 收窄到 4.5%，
  T=256/384 提速 24–27%。加宽或调 batch 无效（§16.5/§13.3：根因是算子数量）。

---

## 四、跨分布（OOD）泛化

方法：原三域（zh_ancient/en_literature/code_python）训练 600 步，
在训练时未出现过的外部新数据集上测 ppl。

新数据集（`prepare_extra.py`，独立采集，非原语料切分）：
- `arxiv_en`：arxiv API 英文科学摘要 500 篇，token 236,595
- `zh_tech_news`：新华网/光明网/人民网科技科教正文 97 篇，token 164,711

公平性前提：BPE 词表来自繁体古籍，`zh_tech_news` 未登录字符率 23.02%（简体字大量落表外），
`arxiv_en` 仅 0.05%。中文新域的高 ppl 含分词器不适配成分，不全是分布偏移，
因此两者分开陈述，不合并成一个比值。

### 结果（ppl 越低越好；600 步、单种子、T=1024）

| 架构 | arxiv_en（新） | zh_tech_news（新） | ext/orig 落差 |
|---|---|---|---|
| **sunnynet** | **35.58（第2）** | **321.08（第1）** | **1.480** |
| h3 | 31.52（第1） | 446.77（第4） | 1.840 |
| rwkv | 41.03（第4） | 427.22（第3） | 1.691 |
| mamba | 40.87（第3） | 499.74（第6） | 1.964 |
| hyena | 54.22（第5） | 400.39（第2） | 1.481 |
| linear_attn | 61.55（第7） | 489.23（第5） | 1.694 |
| transformer | 61.34（第6） | 531.77（第7） | 1.817 |

### 读数

- "跨数据集泛化差"不成立：中文新域 SunnyNet 第 1/7（该域对它有 23% unk 的不利条件）；
  英文新域第 2/7；OOD 落差 1.480，七架构最低档（与 hyena 并列）。
- 口径陷阱：`zh_ancient` 的"同域" ppl 在所有架构上最高（sunnynet 326、transformer 424），
  因 600 步未收敛 + 该域（繁体古籍）最难。拿它当分母算比值会失真——
  `sunnynet` 的 `news/zh_anc=0.985` 看似"无落差"，实为分母虚高，
  不能解读成"跨域比同域还好"。可靠指标只有 `arxiv_en/en_literature`：
  sunny 1.706，与 h3 1.686 / hyena 1.691 同档。
- 不能反向主张"泛化强"：绝对 ppl 仍高、600 步未收敛、单种子。
  可靠结论只有一条：跨分布不比同复杂度同行差。

（数据见 `experiments/results/_ood.json`。）

---

## 五、批评里"结论正确但归因错误"的两处

1. "短序列倒挂限制常规 NLP 应用"：方向对，但前提是前向延迟。
   训练态 T=256 已追平 rwkv（§16），且 §16.5 已证持平仅在 L=4 成立——
   层数对齐到 L=6 时同预算窄配置反慢 30–40%，根因是算子数量而非 FLOPs。
   缓解的正确方向是减算子数（融合/减层），不是加宽或调 batch。

2. "Scaling Law 失效"：这句话在报告里没有数据支撑——
   本仓库只测了 20M / 100M 两点，TL;DR 明写"scaling 趋势并非单调"，
   本身就说明尚未建立 scaling 曲线，谈不上"失效"。要判定需补 ≥4 个规模点。

---

## 六、结论

- 成立的：T<384 前向倒挂（已实测，结构性）；种子方差大（已知）；
  自然语言域长距离检索接近 0（已实测）。
- 不成立的：跨数据集泛化差（OOD 实测中文新域第 1、英文新域第 2、落差最低档）；
  T=1024 训练态"最慢 + 内存最高"；"用随机数据测质量"（质量用真实语料）；
  代码域"长距离检索弱"（与 Transformer 持平）。
- 证据不足：Scaling Law 失效（本仓库无 scaling 曲线，只有 20M/100M 两点）。
- 新增：T=1024 训练态审计（§17）；OOD 跨域实测（§18，推翻"泛化差"）。

数据不支持的表述：任何"全面更快""规模越大越强""长距离强于 Transformer"
"泛化能力强"（只能说"跨分布不比同行差"）。
---

# Item-by-item check of external criticisms

Against a set of external criticisms (short-sequence inversion / Scaling Law failure / weak
long-range retrieval / random datasets / poor multi-seed generalization / slowest at T=1024 with
the highest memory), each claim is checked against this repository's existing measured data and new
measurements, and classified as "holds", "does not hold", or "insufficient evidence".

New scripts and data: `experiments/_T1024_audit.py`, `prepare_extra.py`, `experiments/_ood.py`.

---

## 1. Item-by-item verdicts

| Criticism | Verdict | Evidence |
|---|---|---|
| Slower than Transformer below T=384 | Holds, but only in the batch=1 forward pass regime | §2.4 crossover T≈384; in the training regime (batch=8) the crossover drops to T<256 (§19) |
| Therefore it is limited to common NLP tasks | Does not hold (wrong premise) | Short-text classification / dialogue response belong to the training/fine-tuning regime; training-regime T=256/384 is already 1.5-2.4x faster (§19) |
| Scaling Law failure | Insufficient evidence | This repository measured only two points, 20M / 100M (§14.6 / §14.8); there is no scaling curve, so "failure" cannot be judged |
| Weak long-range retrieval | Does not hold for the code domain, holds for natural language | §8: code-domain far-bucket hit rate 0.1208, on par with Transformer; natural-language domain is near 0 |
| Measured on random datasets | Holds only for the speed benchmarks, not for the quality claims | Speed scripts use `torch.randint` (computation graph and values are independent); the quality comparison runs on `DomainMixer` -> real token stream (§3/§10/§12) |
| Large multi-seed variance | Holds, and is known to the project | §6: seed ppl range 6.32 > architecture gap 3.53; §10.5 already states that a single seed cannot be the basis of a verdict |
| Poor cross-dataset generalization | Does not hold | 1st/7 on the new Chinese domain, 2nd/7 on the new English domain, lowest OOD drop of the group (§18) |
| Slowest at T=1024 | Depends on the premise: in the batch=1 forward pass it ranks 5th (not the slowest - Transformer is last); in the training regime it ranks 1st | §10.6 / §17 |
| Highest peak memory at T=1024 | Does not hold | §17: SunnyNet 1812 MB ranks 2nd-lowest, 43% less than Transformer |

---

## 2. T=1024 training-regime speed and memory (§17)

The existing `on_scaling_t8.json` only has batch=1 pure forward passes, and memory was never
recorded. Training regime added:

Speed (ms/token): sunnynet **0.13186 (1st)** / h3 0.13190 / mamba 0.14721 /
hyena 0.16754 / linear_attn 0.17646 / rwkv 0.19677 / transformer 0.55693

Peak memory (MB): transformer 3192 / **sunnynet 1812 (2nd)** / hyena 1809 /
mamba 1780 / linear_attn 1777 / h3 1689 / rwkv 1611

=> In the training regime both accusations do not hold. Forward-pass and training-regime rankings
differ (backward is 56.9% of the training step, §13.3).

---

## 3. Training-regime check of the short-sequence inversion (§19)

Re-measured three configurations x three lengths x 3 rounds in the training regime
(batch=8, including backward):

| Configuration | T=128 | T=256 | T=384 |
|---|---|---|---|
| `small` (old d128) ratio to TF | 1.104 | 0.672 | 0.581 |
| `smalln80` (narrow, §16) ratio to TF | 1.045 | 0.509 | 0.425 |

(ratio < 1 = faster than Transformer)

- The crossover drops from T≈384 (forward) to T<256 (training regime): already 1.5-2.4x faster at
  T=256/384.
- T=128 is still slower: old configuration 10.4%, narrow configuration 4.5% (inside the
  high-memory-pressure noise band).
- The effective remedy is reducing activation width (narrow residual stream): the T=128
  disadvantage narrows from 10.4% to 4.5%, T=256/384 speed up 24-27%. Widening or adjusting batch
  does not help (§16.5/§13.3: the root cause is operator count).

---

## 4. Cross-distribution (OOD) generalization

Method: train 600 steps on the original three domains
(zh_ancient/en_literature/code_python), measure ppl on external new datasets that never appeared
during training.

New datasets (`prepare_extra.py`, collected independently, not splits of the original corpus):
- `arxiv_en`: 500 English science abstracts from the arxiv API, 236,595 tokens
- `zh_tech_news`: 97 tech/science articles from Xinhua / Guangming / People's Daily, 164,711
  tokens

Fairness premise: the BPE vocabulary comes from traditional-script ancient texts, so
`zh_tech_news` has a 23.02% out-of-vocabulary character rate (many simplified characters fall
outside the table), while `arxiv_en` has only 0.05%. The high ppl of the new Chinese domain
includes a tokenizer-mismatch component, not purely distribution shift, so the two are reported
separately rather than merged into one ratio.

### Results (lower ppl is better; 600 steps, single seed, T=1024)

| Architecture | arxiv_en (new) | zh_tech_news (new) | ext/orig drop |
|---|---|---|---|
| **sunnynet** | **35.58 (2nd)** | **321.08 (1st)** | **1.480** |
| h3 | 31.52 (1st) | 446.77 (4th) | 1.840 |
| rwkv | 41.03 (4th) | 427.22 (3rd) | 1.691 |
| mamba | 40.87 (3rd) | 499.74 (6th) | 1.964 |
| hyena | 54.22 (5th) | 400.39 (2nd) | 1.481 |
| linear_attn | 61.55 (7th) | 489.23 (5th) | 1.694 |
| transformer | 61.34 (6th) | 531.77 (7th) | 1.817 |

### Reading

- "Poor cross-dataset generalization" does not hold: in the new Chinese domain SunnyNet is 1st/7
  (that domain puts it under the unfavorable condition of 23% unk); 2nd/7 on the new English
  domain; OOD drop 1.480, the lowest of the seven architectures (tied with hyena).
- Regime trap: the "same-domain" ppl of `zh_ancient` is the highest on every architecture
  (sunnynet 326, transformer 424), because 600 steps is unconverged plus that domain
  (traditional-script ancient texts) is the hardest. Using it as a denominator distorts ratios -
  `sunnynet`'s `news/zh_anc=0.985` looks like "no drop" but is an inflated denominator; it cannot
  be read as "cross-domain is better than same-domain". The only reliable metric is
  `arxiv_en/en_literature`: sunny 1.706, in the same band as h3 1.686 / hyena 1.691.
- Nor can one reverse the claim to "strong generalization": absolute ppl is still high, 600 steps
  is unconverged, single seed.
  The only reliable conclusion: cross-distribution is no worse than same-complexity peers.

(Data in `experiments/results/_ood.json`.)

---

## 5. Two places where the critics get the conclusion right but the cause wrong

1. "Short-sequence inversion limits common NLP applications": direction right, but the premise is
   forward-pass latency.
   In the training regime T=256 already ties rwkv (§16), and §16.5 already proved the tie holds
   only at L=4 - when layer counts are aligned to L=6 the same-budget narrow configuration is
   30-40% slower, and the root cause is operator count rather than FLOPs.
   The correct remedy is reducing operator count (fusion / fewer layers), not widening or
   adjusting batch.

2. "Scaling Law failure": the claim has no data support in the report -
   this repository measured only two points, 20M / 100M; the TL;DR explicitly says "the scaling
   trend is not monotonic", which itself shows that no scaling curve has been established, so
   "failure" cannot even be discussed. A verdict requires adding at least 4 more scale points.

---

## 6. Conclusions

- Holds: short-sequence forward inversion below T=384 (measured, structural); large seed variance
  (known); near-zero natural-language long-range retrieval (measured).
- Does not hold: poor cross-dataset generalization (OOD measured 1st on new Chinese, 2nd on new
  English, lowest drop); T=1024 training-regime "slowest + highest memory"; "quality measured on
  random data" (quality uses real corpus); "weak long-range retrieval in the code domain" (ties
  Transformer).
- Insufficient evidence: Scaling Law failure (this repository has no scaling curve, only the
  20M/100M two points).
- Added: T=1024 training-regime audit (§17); OOD cross-domain measurement (§18, refutes
  "poor generalization").

Claims the data does not support: any "faster across the board", "larger is stronger",
"long-range retrieval stronger than Transformer", "strong generalization"
(can only say "cross-distribution no worse than peers").
