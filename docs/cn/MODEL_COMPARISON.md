# TinyStories 跨模型对照

*[English](../en/MODEL_COMPARISON.md)*

同一批验证文本、同一 BPC 口径下，把 SunnyNet 与公开 TinyStories 模型及通用模型并列。  
所有数字来自本机实测，原始记录在 `experiments/results/` 与外部结果目录。  
未测或不可比的项直接标注，不用估算值代替。

## TL;DR

| 结论                                                                                 | 证据            | 可信度 |
| ---------------------------------------------------------------------------------- | ------------- | --- |
| karpathy `stories110M`（134M）BPC **0.4353** 最低，BPC 随参数量单调下降                         | 同协议实测（§2.1）   | 高   |
| `roneneldan` 系列是唯一隔离了参数量的同协议序列：非嵌入 0.53M→26.3M，BPC 0.8231→0.5065 单调平滑              | 同协议实测（§2.1）   | 高   |
| **非嵌入参数与 SunnyNet 同档的 `TinyStories-1M` 已能写出完整连贯故事**，SunnyNet 仍在长程重复 ⇒ 差距归因预算，非架构   | 样本对照（§3）      | 高   |
| **GPT-2（124M）BPC 0.8794 优于多数小模型，但生成跑题且循环**——低 BPC 不等于任务适配                          | 样本对照（§3）      | 高   |
| SunnyNet `small` 非嵌入参数 **0.366M** 是表内最小，BPC 1.0663；分词器粒度（2.35 chars/tok）对它系统性不利    | 同协议实测（§2.1）   | 高   |
| SunnyNet 末位排名由**预算**主导：同配置 3000→9000 步 BPC 1.1449→1.0663，与 karpathy 系列差 3500–6400× | 预算曲线（§2.3）    | 高   |
| 那批 1B–2.7B 指令模型与本次补全任务**不可同协议比较**，只列官方卡片口径（§5）                                     | 官方 model card | 高   |

## 1. 评测协议

统一口径，所有模型看完全相同的文本：

- 数据：`TinyStories-valid.txt` 尾部 1500 个故事（对 SunnyNet 是 held-out；对各外部模型训练集之外的 split）。
- 每故事截到 **900 字符**——覆盖 `stories*` 系列 256-token 上限，同时让不同分词器看到同一段字符。
- 指标：**BPC（bits-per-character）** = `nats_per_token ÷ (chars_per_token × ln2)`。  
  按原始字符归一化，跨分词器唯一可比；ppl 不可直接跨分词器比较。
- 逐故事独立前向，不拼接、不跨故事传上下文；故事边界不产生额外困惑。
- 每模型独立子进程，避免进程内残留影响。
- 参数量一律**实测**：用 `AutoConfig` + meta device 实例化后计数，不手推也不采信模型名。  
  总参数与非嵌入参数分别给出；对齐容量时用非嵌入参数。

外部模型读数脚本 `experiments/_ext_model_bench.py`；SunnyNet 侧 `experiments/_sunny_bpc_bench.py`。

## 2. 同协议实测结果

参数量给出两列：**总参数**与**非嵌入参数**（栈参数）。两者都必要——词表大小直接决定  
embedding 占比，只看总参数会把词表差异误读成模型容量差异。

### 2.1 TinyStories 同域模型（同一训练分布，最可比的一组）

| 模型                               | 总参数    | 非嵌入参数      | 架构            | 分词器        | chars/tok | val ppl | **BPC**    |
| -------------------------------- | ------ | ---------- | ------------- | ---------- | --------- | ------- | ---------- |
| **SunnyNet `small`**（9000 步）     | 1.41M  | **0.366M** | SunnyNet      | 自训 8K      | 2.349     | 5.67    | **1.0663** |
| **SunnyNet `small`**（3000 步）     | 1.41M  | **0.366M** | SunnyNet      | 自训 8K      | 2.349     | 6.45    | **1.1449** |
| `roneneldan/TinyStories-1M`      | 3.75M  | 0.530M     | GPT-Neo 8×64  | GPT-2 50K  | 4.211     | 11.05   | 0.8231     |
| `roneneldan/TinyStories-3M`      | 8.28M  | 1.846M     | GPT-Neo 8×128 | GPT-2 50K  | 4.211     | 7.08    | 0.6707     |
| `roneneldan/TinyStories-8M`      | 19.7M  | 6.84M      | GPT-Neo 8×256 | GPT-2 50K  | 4.211     | 5.25    | 0.5681     |
| **`roneneldan/TinyStories-28M`** | 52.0M  | 26.3M      | GPT-Neo 8×512 | GPT-2 50K  | 4.211     | 4.39    | **0.5065** |
| `roneneldan/TinyStories-33M`     | 68.5M  | 29.9M      | GPT-Neo 4×768 | GPT-2 50K  | 4.211     | 4.42    | 0.5094     |
| karpathy `stories15M`            | 24.4M  | 15.2M      | Llama 6×288   | llama2 32K | 3.883     | 4.32    | 0.5441     |
| karpathy `stories42M`            | 58.1M  | 41.7M      | Llama 8×512   | llama2 32K | 3.883     | 3.64    | 0.4799     |
| karpathy `stories110M`           | 134.1M | 109.5M     | Llama 12×768  | llama2 32K | 3.883     | 3.23    | **0.4353** |

读数：

- **`roneneldan` 系列给出最干净的一条曲线**：同一架构（GPT-Neo）、同一分词器、  
  同一训练分布，非嵌入参数 0.53M → 1.85M → 6.84M → 26.3M 对应 BPC  
  0.8231 → 0.6707 → 0.5681 → 0.5065，**单调且平滑**。这是唯一一组把"参数量"单独隔离出来的  
  同协议序列。
- **`TinyStories-28M` 略优于 `TinyStories-33M`**（0.5065 vs 0.5094），尽管后者非嵌入参数更多。  
  差异来自层数：28M 是 8 层 ×512，33M 是 4 层 ×768。论文对此有同类观察——同量级下深层优于浅层。
- **命名不对应任何统一口径。** `TinyStories-1M` 的实际总参数是 3.75M、非嵌入 0.53M，  
  两个数都不是"1M"。所以对齐参数量必须实测，不能按名字取。
- **SunnyNet 是表内非嵌入参数最小的一格（0.366M）**，比 `TinyStories-1M` 还少 31%。  
  它落后 `TinyStories-1M` 0.243 BPC（1.0663 vs 0.8231）。
- **分词器粒度是 BPC 的一个系统性劣势**：SunnyNet 的 8K 词表每 token 只覆盖 2.35 字符，  
  GPT-2 50K 词表覆盖 4.21 字符。同一段文本，SunnyNet 要多预测 **1.8×** 的 token 步数，  
  每步携带的信息更少。BPC 虽然按字符归一化消除了量纲差，但消除不了"更细粒度分词 =  
  更多次预测"这一难度差。这一项对 SunnyNet 不利，读表时须记入。
- **预算敏感性实测**：同一 `small` 配置训练到 9000 步后 BPC 从 1.1449 降到 1.0663（−6.9%）；  
  `TinyStories-1M` 与它非嵌入参数同档，BPC 已经是 0.8231。

### 2.2 通用模型（分布外，仅作对照）

| 模型                                 | 总参数    | 非嵌入参数 | 架构           | 分词器       | chars/tok | val ppl | **BPC**    |
| ---------------------------------- | ------ | ----- | ------------ | --------- | --------- | ------- | ---------- |
| **GPT-2**（`openai-community/gpt2`） | 124.4M | 85.8M | GPT-2 12×768 | GPT-2 50K | 4.211     | 13.03   | **0.8794** |

**GPT-2 是唯一例外**：它以 124M 总参数、远超小模型的通用预训练量，BPC 0.8794 却明显差于  
非嵌入参数仅 0.53M 的 `TinyStories-1M`（0.8231），更远差于 `stories15M`（0.5441）。  
这与 TinyStories 论文的结论一致——通用模型在儿童故事这一窄分布上效率低。

### 2.3 预算差异（不是同一区间的比较）

| 模型            | batch | accum | seq  | 步数      | **tokens ≈** |
| ------------- | ----- | ----- | ---- | ------- | ------------ |
| `stories15M`  | 128   | 4     | 256  | 298,000 | **39.0B**    |
| `stories42M`  | 32    | 4     | 1024 | 164,000 | **21.5B**    |
| `stories110M` | 16    | 8     | 1024 | 198,000 | **26.0B**    |

本项 SunnyNet 为 **6.14M / 18.4M** token，与 karpathy 系列差 **3500–6400×**（3000 步档）。  
karpathy 系列在 TinyStories 上跑了上千个 epoch；SunnyNet 在 4.3 token/参数下远未收敛。

`roneneldan` 系列的逐项预算未公开。可核验的硬约束来自 TinyStories 论文原文：  
**"所有模型都能在单张 V100 上 30 小时内训完"**，且论文明确给出 1M / 2.5M / 8.3M / 28M / 33M  
这组规格。据此其 token 预算在 **10⁸** 量级，比本项 SunnyNet 高一个数量级，比 karpathy 系列低一个数量级。

## 3. BPC 与生成质量不是同一件事

同一 prompt（`Once upon a time, `）下的贪心续写：

| 模型                              | BPC    | 续写片段                                                                                                                                                                      |
| ------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| karpathy `stories110M`          | 0.4353 | "3 year olds were playing in the park. They were having a lot of fun. Suddenly, one of the children noticed a big, red balloon…"                                          |
| `roneneldan/TinyStories-28M`    | 0.5065 | "The little girl was walking in the park when she saw a big, red balloon. She wanted to play with it, so she ran over to it…"                                             |
| `roneneldan/TinyStories-33M`    | 0.5094 | "there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red apple on the ground…"                                         |
| `roneneldan/TinyStories-8M`     | 0.5681 | "there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and growling…"                     |
| **`roneneldan/TinyStories-1M`** | 0.8231 | "One day, a little girl named Lily went to the park with her mommy. They saw a big slide and wanted to slide down the slide. Lily said, 'Mommy, can I go on the slide?'…" |
| **GPT-2**                       | 0.8794 | "**I was a little bit of a fan of the original series**, but I was also a little bit of a fan of the original series. I was a little bit of a fan…"                       |
| **SunnyNet `small`**            | 1.0663 | "there was a little girl named Lily. She loved to play with her toys and make her favorite toy in the park. One day, she saw a big tree…"                                 |

- **`TinyStories-1M` 是关键的一格。** 它的非嵌入参数 0.53M 与 SunnyNet 的 0.366M 同档，  
  却在 BPC 0.8231 时就写出了**完整、不重复、有对话的多句故事**。这说明在这一规模上，  
  "写出连贯故事"本身不是架构难点——**是预算难点**。SunnyNet 的 0.366M 参数并不缺容量。
- SunnyNet `small` 的 BPC 最差，生成却**正确进入故事句式**（"there was a little girl named Lily"），  
  只在长程出现重复（"she was so happy and she was so happy"）。这个症状与它的预算区间（4.3 token/参数）  
  一致，不是跑题。
- GPT-2 的 BPC 优于 `TinyStories-1M`，生成却**跑题并陷入循环**：它把"Once upon a time"接成  
  对"the original series"的自我重复，完全不进入故事语域。这正是 TinyStories 论文记录的失败模式。
- 结论两条，缺一不可：
  1. **BPC 度量概率建模效率，不度量任务适配。** GPT-2 与 `TinyStories-1M` 是同一 BPC 区间内的  
     正反例，只比 BPC 会把 GPT-2 排在更前。
  2. **BPC 也不能单独用来判定架构优劣。** 它同时受预算、分词器粒度、训练分布支配；  
     本表三者的差异都大于架构本身的差异。

## 4. 公开 TinyStories 小模型参照

这些数字来自各自 model card，**训练预算远高于本项对照**，列此说明差距来源，不作同协议比较。

| 模型                           | 参数量   | 分词器           | 官方 val ppl |
| ---------------------------- | ----- | ------------- | ---------- |
| `Maouz0/child-12m`           | 12.3M | 8K custom BPE | 3.45       |
| `pluto-15M`                  | 15M   | —             | ~3.64      |
| `syrinx-10m`                 | 11.8M | 32K byte-BPE  | 4.78       |
| `Compactbot/tinystories-24m` | 24.6M | 8K BPE        | 8.76       |
| 本项 SunnyNet（§2.1）            | 1.41M | 8K BPE        | 6.45       |

跨分词器比较 ppl 无效；上表只说明"同一数据集、同量级模型，官方预算比本项对照高数千倍"（§2 实测）。

## 5. 指令模型不可同协议比较

| 模型                            | 参数量   | 层×宽     | 词表   | 与本任务的关系          |
| ----------------------------- | ----- | ------- | ---- | ---------------- |
| Gemma-2B                      | 2.51B | 18×2048 | 256K | 通用对话模型，非故事补全     |
| MiniCPM-2B                    | 2.4B  | 40×2304 | 123K | SFT 版，非故事补全      |
| Phi-2                         | 2.7B  | 32×2560 | 51K  | 通用 base，非故事补全    |
| Qwen2.5-1.5B                  | 1.54B | 28×1536 | 152K | 通用 base          |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.78B | 28×1536 | 152K | 推理蒸馏，非故事补全       |
| Llama-3.2-1B                  | 1.24B | 16×2048 | 128K | 通用 base/Instruct |

这些模型与 §2.1 的补全模型有四处不同：参数大 3 个数量级、分词器不同、训练数据不同（非 TinyStories）、  
任务不同（通用对话/推理 vs 儿童故事补全）。把它们与本次 1.4M 补全模型作 BPC 对比会得出误导排序。

若确需数字，正确做法是让同一批模型在 TinyStories 上**同协议复训**，或引用各自基准（MMLU、HumanEval 等）  
并在同族内比较。本报告不做跨族比较。

Jev 与 LaYA 属于**决策模型**，不生成文本：Jev（TypeSafe）是闭源 API，开源版 TinyJev 596M；LaYA 是  
ModernBERT 编码器（421M / 322M）加决策头，输出 typed decision 与校准概率。它们与故事补全不同任务，  
不在本表比较范围。

## 6. 复现

```sh
# 外部模型（权重从 hf-mirror 取, 落到 F:/sunnyclaw-data）
python -X utf8 experiments/_convert_tinyllamas.py --hub HUB --tok TOK \
    --models stories15M,stories42M,stories110M --out-root F:/sunnyclaw-data/ext_models
python -X utf8 experiments/_ext_model_bench.py --model openai-community/gpt2 \
    --stories 1500 --char-budget 900 --out F:/sunnyclaw-data/ext_results/_ext_gpt2.json

# TinyStories 同域系列 (1M/3M/8M 的 tokenizer.json 常取不到, slow 路径只需 vocab.json + merges.txt)
for m in 1M 3M 8M 28M 33M; do
  python -X utf8 experiments/_ext_model_bench.py --model roneneldan/TinyStories-$m \
      --stories 1500 --char-budget 900 --out F:/sunnyclaw-data/ext_results/_ext_ts$m.json
done
# 28M 权重 241MB, hub 并发下载易被代理中断, 用 curl -C - 直取后放本地目录, 加 --slow-tok

# SunnyNet 侧: 训练 small 并存 ckpt, 再用同一协议评测
python -X utf8 experiments/_tiny_bench.py --tok fixed_en --arch sunnynet --preset small \
    --steps 3000 --ckpt OUT/_ck_sunnynet_small.pt --out OUT/_tiny_sunnynet_small.json
python -X utf8 experiments/_sunny_bpc_bench.py --arch sunnynet --preset small \
    --ckpt OUT/_ck_sunnynet_small.pt --stories 1500 --char-budget 900 --out OUT/_ext_sunnynet_small.json
```

`_convert_tinyllamas.py` 对 q/k 投影做交错 RoPE 置换：llama2.c 按相邻元素对旋转，HF Llama 按前后两半旋转，  
两种约定下同一权重不等价。漏掉置换会使 logits 全错（实测手写参考与 HF 差 12.1 降至 2.4e-05）。

## 7. 边界

- 外部模型只在通用语料或 TinyStories 上训练，本报告不做任何复训；其读数反映已发布权重在该验证集上的表现。
- SunnyNet 侧训练到 6.14M / 18.4M token（约 4.3 / 12.9 token/参数），**仍远未收敛**；BPC 会随预算继续下降。  
  karpathy 系列的实测预算高出 3500–6400×，两边不在同一预算区间。
- **`roneneldan` 系列的逐项 token 预算未公开**，只有论文的"单张 V100 ≤30 小时"硬约束与  
  1M/2.5M/8.3M/28M/33M 这组规格。§2.3 的 10⁸ 量级是估计，标为估计；对它的一切比较都不是同预算比较。
- **同域模型之间也不是同分词器。** `roneneldan` 用 GPT-2 50K 词表（4.21 chars/tok），  
  SunnyNet 用自训 8K（2.35 chars/tok）。同一段文本 SunnyNet 要多预测 1.8× 的 token 步，  
  每步携带信息更少——BPC 按字符归一化消除了量纲差，但消除不了这一难度差。此项对 SunnyNet 系统性不利。
- 单种子单次读数，未做多种子重复。
- 字符截断 900 对不同分词器略有信息量差异（`chars_per_token` 2.35 至 4.21）。
- BPC 与 ppl 都不是任务质量指标，二者须与生成样本并列看（§3）。
- 未同协议实测、故未列入本表的对照项：`rwkv-nano`（`RWKV/rwkv-4-nano-pile` 为 gated 仓库，  
  镜像站不可取）；可达的 `BlinkDL/rwkv-4-pile-169m` 训练在 Pile 上，属分布外，纳入会与 GPT-2 同性质地失真，  
  故不列。jamba / granite / olmo 等未在清单内，不属本报告范围。

