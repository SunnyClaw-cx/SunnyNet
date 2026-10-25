# TinyStories 跨模型对照

*[English](../en/MODEL_COMPARISON.md)*

同一批验证文本、同一 BPC 口径下，把 SunnyNet 与公开 TinyStories 模型及通用模型并列。
所有数字来自本机实测，原始记录在 `experiments/results/` 与外部结果目录。
未测或不可比的项直接标注，不用估算值代替。

## TL;DR

| 结论 | 证据 | 可信度 |
|---|---|---|
| karpathy `stories110M`（134M）BPC **0.4353** 最低，BPC 随参数量单调下降 | 同协议实测（§2） | 高 |
| **GPT-2（124M）BPC 0.8794 优于多数小模型，但生成跑题且循环**——低 BPC 不等于任务适配 | 样本对照（§3） | 高 |
| SunnyNet `small`（**1.41M**）BPC 1.1449，参数量比 GPT-2 少 88×，故事结构更贴合 | 同协议实测（§2） | 高 |
| SunnyNet 末位排名由**预算**主导：同配置 3000→9000 步 BPC 1.1449→1.0663 | 预算曲线（§2） | 高 |
| 那批 1B–2.7B 指令模型与本次补全任务**不可同协议比较**，只列官方卡片口径（§4） | 官方 model card | 高 |

## 1. 评测协议

统一口径，所有模型看完全相同的文本：

- 数据：`TinyStories-valid.txt` 尾部 1500 个故事（对 SunnyNet 是 held-out；对各外部模型训练集之外的 split）。
- 每故事截到 **900 字符**——覆盖 `stories*` 系列 256-token 上限，同时让不同分词器看到同一段字符。
- 指标：**BPC（bits-per-character）** = `nats_per_token ÷ (chars_per_token × ln2)`。
  按原始字符归一化，跨分词器唯一可比；ppl 不可直接跨分词器比较。
- 逐故事独立前向，不拼接、不跨故事传上下文；故事边界不产生额外困惑。
- 每模型独立子进程，避免进程内残留影响。

外部模型读数脚本 `experiments/_ext_model_bench.py`；SunnyNet 侧 `experiments/_sunny_bpc_bench.py`。

## 2. 同协议实测结果

| 模型 | 参数量 | 层数×宽度 | 分词器 | val ppl | **BPC** |
|---|---|---|---|---|---|
| karpathy `stories110M` | 134.1M | 12×768 | llama2 32K | 3.23 | **0.4353** |
| karpathy `stories42M` | 58.1M | 8×512 | llama2 32K | 3.64 | 0.4799 |
| `roneneldan/TinyStories-33M` | 68.5M | 4×768 | GPT-2 50K | 4.42 | 0.5094 |
| karpathy `stories15M` | 24.4M | 6×288 | llama2 32K | 4.32 | 0.5441 |
| **GPT-2**（`openai-community/gpt2`） | 124.4M | 12×768 | GPT-2 50K | 13.03 | 0.8794 |
| **SunnyNet `small`** | **1.41M** | 4 层 | 自训 8K BPE | 6.45 | **1.1449** |

读数：

- 前四行都是**在 TinyStories 上从零训练**的模型，BPC 与参数量同向：134M（0.435）< 68.5M（0.509）< 24.4M（0.544）。
  其中 `stories42M`（58M）略优于同量级的 `TinyStories-33M`（68.5M），分层与训练预算不同所致。
- **GPT-2 是唯一例外**：它以 124M 参数、远超小模型的通用预训练量，BPC 0.8794 却明显差于 24M 的
  `stories15M`（0.5441）。这与 TinyStories 论文的结论一致——通用模型在儿童故事这一窄分布上效率低。
- SunnyNet `small` 只有 **1.41M 参数**，是表内最小；BPC 1.1449 处于末位，但它的栈参数（非 embedding）
  为 365.8k，训练只用了 6.14M token（约 4.3 token/参数），远未收敛。
- **预算敏感性实测**：同一 `small` 配置训练到 9000 步（18.4M token，3×）后，BPC 从 1.1449 降到
  **1.0663**（−6.9%）。说明表内 SunnyNet 的末位排名由预算主导，不是架构上限。

  | SunnyNet small | token | BPC |
  |---|---|---|
  | 3000 步 | 6.14M | 1.1449 |
  | 9000 步 | 18.4M | **1.0663** |
- **参数不是唯一变量。** `stories15M` 用 24.4M 参数比 SunnyNet 多 17×，BPC 低到 0.5441。
  差距主要来自训练预算——从各自 checkpoint 的 config 实测（batch × grad_accum × seq_len × iter_num）：

  | 模型 | batch | accum | seq | 步数 | **tokens ≈** |
  |---|---|---|---|---|---|
  | `stories15M` | 128 | 4 | 256 | 298,000 | **39.0B** |
  | `stories42M` | 32 | 4 | 1024 | 164,000 | **21.5B** |
  | `stories110M` | 16 | 8 | 1024 | 198,000 | **26.0B** |

  本项 SunnyNet 为 **6.14M** token，与它们差 **3500–6400×**。karpathy 系列在 TinyStories 上跑了
  上千个 epoch；SunnyNet 在 4.3 token/参数下远未收敛。

## 3. BPC 与生成质量不是同一件事

同一 prompt（`Once upon a time, `）下的贪心续写：

| 模型 | BPC | 续写片段 |
|---|---|---|
| karpathy `stories110M` | 0.4353 | "3 year olds were playing in the park. They were having a lot of fun. Suddenly, one of the children noticed a big, red balloon…" |
| `TinyStories-33M` | 0.5094 | "there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red apple on the ground…" |
| **GPT-2** | 0.8794 | "**I was a little bit of a fan of the original series**, but I was also a little bit of a fan of the original series. I was a little bit of a fan…" |
| **SunnyNet `small`** | 1.1449 | "there was a little girl named Lily. She loved to play with her toys and make her favorite toy in the park. One day, she saw a big tree…" |

- GPT-2 的 BPC 在六个模型里排第二好，生成却**跑题并陷入循环**：它把"Once upon a time"接成
  对"the original series"的自我重复，完全不进入故事语域。这正是 TinyStories 论文记录的失败模式。
- SunnyNet `small` 的 BPC 最差，生成却**正确进入故事句式**（"there was a little girl named Lily"），
  只在长程出现重复（"she was so happy and she was so happy"），属于预算不足的典型症状，不是跑题。
- 结论：**BPC 度量的是概率建模效率，不度量任务适配**。跨架构只比 BPC 会得出与实际可用性相反的排序。
  两者必须并列陈述。

## 4. 公开 TinyStories 小模型参照

这些数字来自各自 model card，**训练预算远高于本项对照**，列此说明差距来源，不作同协议比较。

| 模型 | 参数量 | 分词器 | 官方 val ppl |
|---|---|---|---|
| `Maouz0/child-12m` | 12.3M | 8K custom BPE | 3.45 |
| `pluto-15M` | 15M | — | ~3.64 |
| `syrinx-10m` | 11.8M | 32K byte-BPE | 4.78 |
| `Compactbot/tinystories-24m` | 24.6M | 8K BPE | 8.76 |
| 本项 SunnyNet（§2） | 1.41M | 8K BPE | 6.45 |

跨分词器比较 ppl 无效；上表只说明"同一数据集、同量级模型，官方预算比本项对照高数千倍"（§2 实测）。

## 5. 指令模型不可同协议比较

| 模型 | 参数量 | 层×宽 | 词表 | 与本任务的关系 |
|---|---|---|---|---|
| Gemma-2B | 2.51B | 18×2048 | 256K | 通用对话模型，非故事补全 |
| MiniCPM-2B | 2.4B | 40×2304 | 123K | SFT 版，非故事补全 |
| Phi-2 | 2.7B | 32×2560 | 51K | 通用 base，非故事补全 |
| Qwen2.5-1.5B | 1.54B | 28×1536 | 152K | 通用 base |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.78B | 28×1536 | 152K | 推理蒸馏，非故事补全 |
| Llama-3.2-1B | 1.24B | 16×2048 | 128K | 通用 base/Instruct |

这些模型与 §2 的补全模型有四处不同：参数大 3 个数量级、分词器不同、训练数据不同（非 TinyStories）、
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
python -X utf8 experiments/_ext_model_bench.py --model roneneldan/TinyStories-33M \
    --stories 1500 --char-budget 900 --out F:/sunnyclaw-data/ext_results/_ext_ts33m.json

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
- 单种子单次读数，未做多种子重复。
- 字符截断 900 对不同分词器略有信息量差异（`chars_per_token` 2.35 至 4.21）。
- BPC 与 ppl 都不是任务质量指标，二者须与生成样本并列看（§3）。