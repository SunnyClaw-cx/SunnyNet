*[中文版](../cn/MODEL_COMPARISON.md)*

# Cross-model comparison on TinyStories

The same validation text and the same BPC protocol are applied to SunnyNet, to public TinyStories
models, and to a general-purpose model. Every number is measured locally; raw records are under
`experiments/results/` and the external result directory. Items that were not measured, or that
cannot be compared, are marked as such rather than replaced with estimates.

## TL;DR

| Finding | Evidence | Confidence |
|---|---|---|
| karpathy `stories110M` (134M) has the lowest BPC, **0.4353**; BPC falls monotonically with parameters among TinyStories-trained models | Same-protocol measurement (§2.1) | High |
| The `roneneldan` series is the only same-protocol sequence that isolates parameter count: non-embedding 0.53M→26.3M gives BPC 0.8231→0.5065, monotone and smooth | Same-protocol measurement (§2.1) | High |
| **`TinyStories-1M`, whose non-embedding size matches SunnyNet's, already writes complete coherent stories** while SunnyNet still repeats over long range — so the gap is budget, not architecture | Sample comparison (§3) | High |
| **GPT-2 (124M) reaches BPC 0.8794, better than several smaller models, yet drifts off-genre and loops** — low BPC is not task fitness | Sample comparison (§3) | High |
| SunnyNet `small` has the smallest non-embedding size in the table, **0.366M**, at BPC 1.0663; its tokenizer granularity (2.35 chars/tok) is a systematic handicap | Same-protocol measurement (§2.1) | High |
| SunnyNet's last-place rank is driven by **budget**: 3000→9000 steps moves BPC 1.1449→1.0663, and the karpathy models sit 3500–6400× higher | Budget curve (§2.3) | High |
| The 1B–2.7B instruct models **cannot be compared under this protocol**; only their published figures are listed (§5) | Official model cards | High |

## 1. Protocol

One protocol, one identical text for every model:

- Data: the last 1500 stories of `TinyStories-valid.txt` (held out for SunnyNet; outside the training
  split of every external model).
- Each story truncated to **900 characters** — inside the 256-token limit of the `stories*` models,
  and the same character span is fed to every tokenizer.
- Metric: **BPC (bits-per-character)** = `nats_per_token ÷ (chars_per_token × ln2)`.
  Normalising by raw character count is the only cross-tokenizer comparison that holds; perplexity is
  not comparable across tokenizers.
- Each story is a separate forward pass. Nothing is concatenated, no context crosses a story boundary,
  so the boundary adds no extra loss.
- One subprocess per model, so no state carries over inside a process.
- Parameter counts are **measured**: instantiate from `AutoConfig` on the meta device and count. No
  hand derivation, and no trusting the model name. Total and non-embedding sizes are both reported;
  capacity alignment uses the non-embedding size.

External models: `experiments/_ext_model_bench.py`. SunnyNet side: `experiments/_sunny_bpc_bench.py`.

## 2. Same-protocol result

Both columns are reported: **total** and **non-embedding** (stack) parameters. Both are needed —
vocabulary size dictates the embedding share, so reading total parameters alone mistakes a vocabulary
difference for a capacity difference.

### 2.1 TinyStories-domain models (same training distribution, the most comparable group)

| Model | Total | Non-embed | Architecture | Tokenizer | chars/tok | val ppl | **BPC** |
|---|---|---|---|---|---|---|---|
| **SunnyNet `small`** (9000 steps) | 1.41M | **0.366M** | SunnyNet | self-trained 8K | 2.349 | 5.67 | **1.0663** |
| **SunnyNet `small`** (3000 steps) | 1.41M | **0.366M** | SunnyNet | self-trained 8K | 2.349 | 6.45 | **1.1449** |
| `roneneldan/TinyStories-1M` | 3.75M | 0.530M | GPT-Neo 8×64 | GPT-2 50K | 4.211 | 11.05 | 0.8231 |
| `roneneldan/TinyStories-3M` | 8.28M | 1.846M | GPT-Neo 8×128 | GPT-2 50K | 4.211 | 7.08 | 0.6707 |
| `roneneldan/TinyStories-8M` | 19.7M | 6.84M | GPT-Neo 8×256 | GPT-2 50K | 4.211 | 5.25 | 0.5681 |
| **`roneneldan/TinyStories-28M`** | 52.0M | 26.3M | GPT-Neo 8×512 | GPT-2 50K | 4.211 | 4.39 | **0.5065** |
| `roneneldan/TinyStories-33M` | 68.5M | 29.9M | GPT-Neo 4×768 | GPT-2 50K | 4.211 | 4.42 | 0.5094 |
| karpathy `stories15M` | 24.4M | 15.2M | Llama 6×288 | llama2 32K | 3.883 | 4.32 | 0.5441 |
| karpathy `stories42M` | 58.1M | 41.7M | Llama 8×512 | llama2 32K | 3.883 | 3.64 | 0.4799 |
| karpathy `stories110M` | 134.1M | 109.5M | Llama 12×768 | llama2 32K | 3.883 | 3.23 | **0.4353** |

Reading:

- **The `roneneldan` series gives the cleanest curve here.** One architecture (GPT-Neo), one tokenizer,
  one training distribution: non-embedding 0.53M → 1.85M → 6.84M → 26.3M gives BPC
  0.8231 → 0.6707 → 0.5681 → 0.5065, **monotone and smooth**. It is the only same-protocol sequence
  that isolates parameter count as a single variable.
- **`TinyStories-28M` edges out `TinyStories-33M`** (0.5065 vs 0.5094) despite fewer non-embedding
  parameters. Depth explains it: 28M is 8 layers ×512, 33M is 4 layers ×768. The paper observes the
  same effect — at equal scale, deeper beats wider.
- **The names match no consistent convention.** `TinyStories-1M` is really 3.75M total and 0.53M
  non-embedding; neither is "1M". Capacity alignment therefore has to be measured, not inferred from names.
- **SunnyNet is the smallest non-embedding entry (0.366M)**, 31% below `TinyStories-1M`. It trails
  `TinyStories-1M` by 0.243 BPC (1.0663 vs 0.8231).
- **Tokenizer granularity is a systematic handicap for BPC.** SunnyNet's 8K vocabulary covers 2.35
  characters per token; GPT-2's 50K covers 4.21. On the same text SunnyNet must make **1.8×** more
  prediction steps, each carrying less information. Normalising by character removes the unit difference
  but not this difficulty difference. This works against SunnyNet and must be read into the table.
- **Budget sensitivity, measured**: the same `small` config at 9000 steps drops BPC from 1.1449 to
  1.0663 (−6.9%); `TinyStories-1M`, at the same non-embedding scale, is already at 0.8231.

### 2.2 General-purpose model (out of distribution, reference only)

| Model | Total | Non-embed | Architecture | Tokenizer | chars/tok | val ppl | **BPC** |
|---|---|---|---|---|---|---|---|
| **GPT-2** (`openai-community/gpt2`) | 124.4M | 85.8M | GPT-2 12×768 | GPT-2 50K | 4.211 | 13.03 | **0.8794** |

**GPT-2 is the outlier.** With 124M total parameters and far more general pretraining than the small
models, its BPC of 0.8794 is clearly worse than `TinyStories-1M` (0.8231) at 0.53M non-embedding, and
far worse than `stories15M` (0.5441). This matches the TinyStories paper: a general model is inefficient
on this narrow distribution.

### 2.3 Budget (not a same-regime comparison)

| Model | batch | accum | seq | steps | **tokens ≈** |
|---|---|---|---|---|---|
| `stories15M` | 128 | 4 | 256 | 298,000 | **39.0B** |
| `stories42M` | 32 | 4 | 1024 | 164,000 | **21.5B** |
| `stories110M` | 16 | 8 | 1024 | 198,000 | **26.0B** |

measured from each checkpoint's own config (batch × grad_accum × seq_len × iter_num). This SunnyNet run
saw **6.14M / 18.4M** tokens — a factor of **3500–6400×** below those at the 3000-step setting. The
karpathy models ran for over a thousand epochs on TinyStories; SunnyNet is far from converged at
4.3 tokens/parameter.

Per-model budgets for the `roneneldan` series are not published. The verifiable hard constraint comes
from the TinyStories paper itself: **"all of the models can be trained on a single V100 GPU within at
most 30 hours"**, with the paper naming the 1M / 2.5M / 8.3M / 28M / 33M set. That places their budget
at the **10⁸** scale — one order of magnitude above this SunnyNet run and one below the karpathy series.

## 3. BPC and generation quality are different things

Greedy continuation of the same prompt (`Once upon a time, `):

| Model | BPC | Continuation |
|---|---|---|
| karpathy `stories110M` | 0.4353 | "3 year olds were playing in the park. They were having a lot of fun. Suddenly, one of the children noticed a big, red balloon…" |
| `roneneldan/TinyStories-28M` | 0.5065 | "The little girl was walking in the park when she saw a big, red balloon. She wanted to play with it, so she ran over to it…" |
| `roneneldan/TinyStories-33M` | 0.5094 | "there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red apple on the ground…" |
| `roneneldan/TinyStories-8M` | 0.5681 | "there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and growling…" |
| **`roneneldan/TinyStories-1M`** | 0.8231 | "One day, a little girl named Lily went to the park with her mommy. They saw a big slide and wanted to slide down the slide. Lily said, 'Mommy, can I go on the slide?'…" |
| **GPT-2** | 0.8794 | "**I was a little bit of a fan of the original series**, but I was also a little bit of a fan of the original series. I was a little bit of a fan…" |
| **SunnyNet `small`** | 1.0663 | "there was a little girl named Lily. She loved to play with her toys and make her favorite toy in the park. One day, she saw a big tree…" |

- **`TinyStories-1M` is the key row here.** At 0.53M non-embedding — the same scale as SunnyNet's
  0.366M — it already produces a **complete, non-repeating, multi-sentence story with dialogue** at BPC
  0.8231. At this scale, writing coherent stories is not an architectural difficulty; it is a **budget**
  difficulty. SunnyNet's 0.366M parameters do not lack capacity.
- SunnyNet `small` has the worst BPC, yet it **enters the correct story sentence pattern**
  ("there was a little girl named Lily"), repeating only over long range ("she was so happy and she was
  so happy"). That symptom matches its budget regime (4.3 tokens/parameter); it is not drift.
- GPT-2's BPC beats `TinyStories-1M`'s, yet its output **drifts off-genre and loops**: it turns
  "Once upon a time" into self-repetition about "the original series" and never enters the story
  register. This is the failure mode the TinyStories paper documents.
- Two conclusions, both required:
  1. **BPC measures how efficiently a model models the distribution; it does not measure task fitness.**
     GPT-2 and `TinyStories-1M` are the positive and negative example inside one BPC band; ranking by
     BPC alone would put GPT-2 ahead.
  2. **BPC alone cannot rank architectures either.** It is simultaneously governed by budget, tokenizer
     granularity, and training distribution — and in this table all three differ by more than the
     architecture does.

## 4. Published TinyStories models for reference

These figures come from the models' own cards and were trained with a budget far above this
comparison's. They show where the gap comes from, not a same-protocol comparison.

| Model | Params | Tokenizer | Reported val ppl |
|---|---|---|---|
| `Maouz0/child-12m` | 12.3M | 8K custom BPE | 3.45 |
| `pluto-15M` | 15M | — | ~3.64 |
| `syrinx-10m` | 11.8M | 32K byte-BPE | 4.78 |
| `Compactbot/tinystories-24m` | 24.6M | 8K BPE | 8.76 |
| This SunnyNet (§2.1) | 1.41M | 8K BPE | 6.45 |

Perplexity cannot be compared across tokenizers; the table makes only one point — on the same dataset
and at the same scale, the published budget exceeds this comparison's by thousands of times (§2).

## 5. Instruct models are not comparable under this protocol

| Model | Params | Layers×width | Vocab | Relation to this task |
|---|---|---|---|---|
| Gemma-2B | 2.51B | 18×2048 | 256K | general chat model, not story completion |
| MiniCPM-2B | 2.4B | 40×2304 | 123K | SFT variant, not story completion |
| Phi-2 | 2.7B | 32×2560 | 51K | general base, not story completion |
| Qwen2.5-1.5B | 1.54B | 28×1536 | 152K | general base |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.78B | 28×1536 | 152K | reasoning distillation, not story completion |
| Llama-3.2-1B | 1.24B | 16×2048 | 128K | general base/Instruct |

These differ from the completion models in §2.1 in four ways: three orders of magnitude more parameters,
a different tokenizer, different training data (not TinyStories), and a different task (general
chat/reasoning vs children's story completion). Comparing their BPC against a 1.4M completion model
would produce a misleading ordering.

For actual numbers the correct approach is to retrain the same models on TinyStories under one
protocol, or to cite each model's own benchmarks (MMLU, HumanEval, …) and compare within a family.
This report does no cross-family comparison.

Jev and LaYA are **decision models**; they do not generate text. Jev (TypeSafe) is a closed API, with
the open TinyJev at 596M; LaYA is a ModernBERT encoder (421M / 322M) with a decision head that emits
typed decisions and calibrated probabilities. They are a different task from story completion and are
outside this table.

## 6. Reproduce

```sh
# External models (weights from hf-mirror, stored on F:/sunnyclaw-data)
python -X utf8 experiments/_convert_tinyllamas.py --hub HUB --tok TOK \
    --models stories15M,stories42M,stories110M --out-root F:/sunnyclaw-data/ext_models
python -X utf8 experiments/_ext_model_bench.py --model openai-community/gpt2 \
    --stories 1500 --char-budget 900 --out F:/sunnyclaw-data/ext_results/_ext_gpt2.json

# TinyStories-domain series (tokenizer.json for 1M/3M/8M often will not download;
# the slow path only needs vocab.json + merges.txt, so pass --slow-tok)
for m in 1M 3M 8M 28M 33M; do
  python -X utf8 experiments/_ext_model_bench.py --model roneneldan/TinyStories-$m \
      --stories 1500 --char-budget 900 --out F:/sunnyclaw-data/ext_results/_ext_ts$m.json
done
# 28M weights are 241MB; hub's concurrent download gets cut by the proxy, so fetch with
# curl -C - into a local directory and run with --slow-tok

# SunnyNet side: train small and keep a checkpoint, then evaluate under the same protocol
python -X utf8 experiments/_tiny_bench.py --tok fixed_en --arch sunnynet --preset small \
    --steps 3000 --ckpt OUT/_ck_sunnynet_small.pt --out OUT/_tiny_sunnynet_small.json
python -X utf8 experiments/_sunny_bpc_bench.py --arch sunnynet --preset small \
    --ckpt OUT/_ck_sunnynet_small.pt --stories 1500 --char-budget 900 --out OUT/_ext_sunnynet_small.json
```

`_convert_tinyllamas.py` permutes the q/k projections for interleaved RoPE: llama2.c rotates adjacent
element pairs, HF Llama rotates the two halves. The same weights are not equivalent under the two
conventions; omitting the permutation makes every logit wrong (the hand-written reference against HF
fell from 12.1 to 2.4e-05 once permuted).

## 7. Boundaries

- External models are used as published; no retraining was done. Their numbers reflect released weights
  on this validation set.
- The SunnyNet side trained to 6.14M / 18.4M tokens (about 4.3 / 12.9 tokens/parameter), **still far from
  converged**; BPC will keep falling with budget. The karpathy models' measured budget is 3500–6400×
  higher — the two are not in the same budget regime.
- **Per-model token budgets for the `roneneldan` series are not published.** Only the paper's
  "single V100, at most 30 hours" constraint and the 1M/2.5M/8.3M/28M/33M set are verifiable. The 10⁸
  scale in §2.3 is an estimate and is marked as one; no comparison against it here is a
  same-budget comparison.
- **The domain models do not share a tokenizer either.** `roneneldan` uses GPT-2's 50K vocabulary
  (4.21 chars/tok); SunnyNet uses a self-trained 8K one (2.35 chars/tok). On the same text SunnyNet
  makes 1.8× more prediction steps with less information per step. Character normalisation removes the
  unit difference but not this difficulty difference. It works systematically against SunnyNet.
- Single seed, single reading; no multi-seed repetition.
- The 900-character truncation carries slightly different information across tokenizers
  (`chars_per_token` from 2.35 to 4.21).
- Neither BPC nor perplexity is a task-quality metric; both must be read alongside generated samples (§3).
- Not measured under this protocol, therefore not in this table: `rwkv-nano`
  (`RWKV/rwkv-4-nano-pile` is a gated repo, not fetchable through the mirror). The reachable
  `BlinkDL/rwkv-4-pile-169m` was trained on the Pile, so it is out of distribution and would distort
  in the same way GPT-2 does; it is left out.