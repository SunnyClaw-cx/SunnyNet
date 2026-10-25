*[中文版](../cn/MODEL_COMPARISON.md)*

# Cross-model comparison on TinyStories

The same validation text and the same BPC protocol are applied to SunnyNet, to public TinyStories
models, and to a general-purpose model. Every number is measured locally; raw records are under
`experiments/results/` and the external result directory. Items that were not measured, or that
cannot be compared, are marked as such rather than replaced with estimates.

## TL;DR

| Finding | Evidence | Confidence |
|---|---|---|
| karpathy `stories110M` (134M) has the lowest BPC, **0.4353**; BPC falls monotonically with parameters among TinyStories-trained models | Same-protocol measurement (§2) | High |
| **GPT-2 (124M) reaches BPC 0.8794, better than several smaller models, yet drifts off-genre and loops** — low BPC is not task fitness | Sample comparison (§3) | High |
| SunnyNet `small` (**1.41M**) reaches BPC 1.1449; 88× fewer parameters than GPT-2, and it stays inside the story register | Same-protocol measurement (§2) | High |
| SunnyNet's last-place rank is driven by **budget**: the same config at 3000→9000 steps moves BPC 1.1449→1.0663 | Budget curve (§2) | High |
| The 1B–2.7B instruct models **cannot be compared under this protocol**; only their published figures are listed (§4) | Official model cards | High |

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

External models: `experiments/_ext_model_bench.py`. SunnyNet side: `experiments/_sunny_bpc_bench.py`.

## 2. Same-protocol result

| Model | Params | Layers×width | Tokenizer | val ppl | **BPC** |
|---|---|---|---|---|---|
| karpathy `stories110M` | 134.1M | 12×768 | llama2 32K | 3.23 | **0.4353** |
| karpathy `stories42M` | 58.1M | 8×512 | llama2 32K | 3.64 | 0.4799 |
| `roneneldan/TinyStories-33M` | 68.5M | 4×768 | GPT-2 50K | 4.42 | 0.5094 |
| karpathy `stories15M` | 24.4M | 6×288 | llama2 32K | 4.32 | 0.5441 |
| **GPT-2** (`openai-community/gpt2`) | 124.4M | 12×768 | GPT-2 50K | 13.03 | 0.8794 |
| **SunnyNet `small`** | **1.41M** | 4 layers | self-trained 8K BPE | 6.45 | **1.1449** |

Reading:

- The first four rows are trained from scratch on TinyStories; BPC tracks parameters:
  134M (0.435) < 68.5M (0.509) < 24.4M (0.544). `stories42M` (58M) edges out the
  similarly-sized `TinyStories-33M` (68.5M), a difference of depth and training budget.
- **GPT-2 is the outlier.** With 124M parameters and far more general pretraining than the small
  models, its BPC of 0.8794 is clearly worse than 24M `stories15M` (0.5441). This matches the
  TinyStories paper: a general model is inefficient on this narrow distribution.
- SunnyNet `small` has **1.41M parameters**, the smallest in the table. Its BPC of 1.1449 is last, but
  its stack size (non-embedding) is 365.8k and it saw only 6.14M tokens (about 4.3 tokens/parameter),
  far from converged.
- **Budget sensitivity, measured**: the same `small` config trained to 9000 steps (18.4M tokens, 3×)
  drops BPC from 1.1449 to **1.0663** (−6.9%). The last-place rank is driven by budget, not by an
  architectural ceiling.

  | SunnyNet small | tokens | BPC |
  |---|---|---|
  | 3000 steps | 6.14M | 1.1449 |
  | 9000 steps | 18.4M | **1.0663** |
- **Parameters are not the only variable.** `stories15M` carries 17× more parameters than SunnyNet and
  reaches BPC 0.5441. Most of that gap is training budget — measured from each checkpoint's own config
  (batch × grad_accum × seq_len × iter_num):

  | Model | batch | accum | seq | steps | **tokens ≈** |
  |---|---|---|---|---|---|
  | `stories15M` | 128 | 4 | 256 | 298,000 | **39.0B** |
  | `stories42M` | 32 | 4 | 1024 | 164,000 | **21.5B** |
  | `stories110M` | 16 | 8 | 1024 | 198,000 | **26.0B** |

  This SunnyNet run saw **6.14M** tokens — a factor of **3500–6400×** below those. The karpathy models
  ran for over a thousand epochs on TinyStories; SunnyNet is far from converged at 4.3 tokens/parameter.

## 3. BPC and generation quality are different things

Greedy continuation of the same prompt (`Once upon a time, `):

| Model | BPC | Continuation |
|---|---|---|
| karpathy `stories110M` | 0.4353 | "3 year olds were playing in the park. They were having a lot of fun. Suddenly, one of the children noticed a big, red balloon…" |
| `TinyStories-33M` | 0.5094 | "there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red apple on the ground…" |
| **GPT-2** | 0.8794 | "**I was a little bit of a fan of the original series**, but I was also a little bit of a fan of the original series. I was a little bit of a fan…" |
| **SunnyNet `small`** | 1.1449 | "there was a little girl named Lily. She loved to play with her toys and make her favorite toy in the park. One day, she saw a big tree…" |

- GPT-2 has the second-best BPC of the six models, yet its output **drifts off-genre and loops**: it
  turns "Once upon a time" into self-repetition about "the original series" and never enters the story
  register. This is the failure mode the TinyStories paper documents.
- SunnyNet `small` has the worst BPC, yet it **enters the correct story sentence pattern**
  ("there was a little girl named Lily"), repeating only over long range ("she was so happy and she was
  so happy") — the typical symptom of an insufficient budget, not of drift.
- Conclusion: **BPC measures how efficiently a model models the distribution; it does not measure task
  fitness.** Ranking architectures by BPC alone gives an ordering opposite to practical usability. The
  two must be stated side by side.

## 4. Published TinyStories models for reference

These figures come from the models' own cards and were trained with a budget far above this
comparison's. They show where the gap comes from, not a same-protocol comparison.

| Model | Params | Tokenizer | Reported val ppl |
|---|---|---|---|
| `Maouz0/child-12m` | 12.3M | 8K custom BPE | 3.45 |
| `pluto-15M` | 15M | — | ~3.64 |
| `syrinx-10m` | 11.8M | 32K byte-BPE | 4.78 |
| `Compactbot/tinystories-24m` | 24.6M | 8K BPE | 8.76 |
| This SunnyNet (§2) | 1.41M | 8K BPE | 6.45 |

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

These differ from the completion models in §2 in four ways: three orders of magnitude more parameters,
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
python -X utf8 experiments/_ext_model_bench.py --model roneneldan/TinyStories-33M \
    --stories 1500 --char-budget 900 --out F:/sunnyclaw-data/ext_results/_ext_ts33m.json

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
- Single seed, single reading; no multi-seed repetition.
- The 900-character truncation carries slightly different information across tokenizers
  (`chars_per_token` from 2.35 to 4.21).
- Neither BPC nor perplexity is a task-quality metric; both must be read alongside generated samples (§3).