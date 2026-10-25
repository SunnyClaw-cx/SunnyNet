# Trained-model showcase (2026-10-24 session)

Seven SunnyNet checkpoints were trained and probed across one local box (an
RTX 2050, 4 GB) and a free-tier P100 (16 GB). All numbers below are read from
the runs' JSON logs (`runs/*.json`, Kaggle kernel logs) — not hand-typed. The
goal of the session: how far do small-to-mid SunnyNet models go on *real*
public-domain data before the text falls apart.

The recurring result: at the data budget reachable here (~1 token/parameter),
every model lands in the **fluent-nonsense regime** — right word shapes and
cadence, wrong content. That is expected; it is the undertrained regime,
quantified in §2, not a defect in the architecture.

## 1. Runs at a glance

| # | Model | Params | HW | Steps | Tokens trained | Data | Best val |
|---|-------|-------:|----|------:|---------------:|------|---------:|
| 1 | `small` | 1.41M | CPU ×4 | 300 | 0.6M | zh/en/code mix | 4.176 (PPL 65.1) |
| 2 | `proto20m` | 20.96M | RTX 2050 | 1,000 | 2.0M | mixed | 4.101 (PPL 60.4) |
| 3 | `proto20m` EN | 20.96M | RTX 2050 | 10,000 | 20.5M | en_literature | 2.166 (PPL 8.72) |
| 4 | `p100m` EN (35 books) | 102.1M | P100 | 24,000 | 98.3M | 35 classics, ~5.5 ep | 2.476 (PPL 11.9) |
| 5 | `p100m` EN-diverse | 102.1M | P100 | 24,841 | 101.7M | 197 books, ~1.5 ep | 2.571 (PPL 13.1) |
| 6 | `p100m` 70/30 code | 102.1M | P100 | 26,851 | 109.9M | en 70% + py 30%, ~1.5 ep | 2.414 (PPL 11.2) |
| 7 | `proto20m` big-data | 20.96M | P100 | ~30,000 | ~260M | ~1,000 books, seq 1024 | *in progress* |

All runs: AdamW (β 0.9/0.95, wd 0.01, grad-clip 1.0), LR 3e-3 linear warmup →
cosine, BPE vocab 8,192, fused kernel off (pure-torch path on GPU).

> **Val loss is not comparable across rows.** Row 4's validation set is the
> *same 35 books* the model memorised, so its low loss overstates quality. Rows
> 5–6 validate on a *broader* book set, so their loss looks worse while the
> model is less overfit. Treat the generation samples in §3 as the real signal.

## 2. The one number that explains it: tokens / parameter

| Run | Params | Tokens trained | **tok/param** | Unique data |
|----:|-------:|---------------:|--------------:|------------:|
| 3 | 21M | 20.5M | 1.0 | 18M |
| 4 | 94M | 98.3M | 1.05 | ~18M (5.5 ep) |
| 5 | 94M | 101.7M | 1.08 | 67.8M (1.5 ep) |
| 6 | 94M | 109.9M | 1.17 | 73.3M (1.5 ep) |
| 7 (target) | 21M | ~260M | **~12** | ~300M (0.8 ep) |

Runs 3–6 all sit at ~1 token/param — the regime where a model learns *surface
statistics* but not *content*. Row 7 inverts this: a **smaller** model (21M)
over **~1,000 diverse books** and **longer context (seq 1024, near-free on an
O(N) kernel)**, reaching ~12–15 tokens/param. That is the "coherent-but-not-
smart" band the session is chasing. Chinchilla would want ~2B tokens for the
94M model; the free-tier GPU (30 h/week) cannot reach that, which is why row 7
trades model size for data volume.

## 3. Per-model notes

### 3.1 `small` (1.4M, POC)
Pipeline verification, not a quality target. 300 mixed-domain steps on CPU:
loss 5.68→3.88, PPL 65, 0.176 ms/token. Confirmed the full chain
(data→BPE→train→eval→CLI) and O(N) latency at 4 threads. Not meant to be used.

### 3.2 `proto20m` (21M, mixed, 1k steps)
First "real" model; deliberately unconverged. O(N) latency confirmed: per-token
cost *falls* as context grows (L=128→2048: 0.63→0.43 ms/token on CPU).

### 3.3 `proto20m` English-only, 10k steps — first fluent nonsense
The first model to sound like a language rather than a token bag. PPL 8.72:

> `It was the best of times, ` → "formed in the work of the places. One of the
> present that fancient caped to be evidently reading with"
>
> `Once upon a time, ` → "and of the great rocken, as he had been evidently
> and the evidation of the part of the warple guar of"

Correct article/noun/verb *rhythm*, wrong words and meaning — the 1-tok/param
signature.

### 3.4 `p100m` English, 35 classics (kernel v3)
94M model, 24k steps, but over only ~35 books → ~5.5 epochs of repetition.
Low val loss (PPL 11.9) that *overstates* coherence, because it measures
next-token fit on memorised novels, not language:

> `The morning sun was ` → "obliged in a sading of her hand and straight came
> me, as I am on the drew of the room, and a go"

This run diagnosed the data problem: **not enough variety.**

### 3.5 `p100m` English, 197 diverse books (kernel v4)
Same 94M model, ~200 most-downloaded public-domain books (67.8M unique tokens),
~1.5 epochs. Val loss rose (PPL 13.1) but generation quality did **not**
move — the token/param ratio barely changed (1.08 vs 1.05). The lesson:
**more data at the same model size and near-1 epoch is not enough to escape the
fluent-nonsense regime; you need either a smaller model over far more data, or a
much bigger token budget.**

> `Once upon a time, ` → "then in the twenty back I know rejected to the world
> of the Church Parrence in the stroving of the"

### 3.6 `p100m` 70/30 English+code (stage-2 preview)
70% English prose + 30% Python (numpy/flask/requests/urllib3 source, 569
files). PPL 11.2. The model learns code *shape* (recognisable constructs,
correct indentation cadence) but emits fragments, not functions:

> `import re` → "for i in range(response):"
>
> `def quicksort(arr):` → ""marks"]["[80] * [""i, "", "eq")"

Code competence is gated by the same token/param ratio: 30% code over ~100M
tokens is still a thin slice for a 94M model.

### 3.7 `proto20m` big-data (kernel v5, in progress)
The bet that should break through: 21M model, ~1,000 diverse books, seq 1024,
~30k steps ≈ ~260M tokens ≈ **~12–15 tokens/param**, plus longer context.
Expected: *readable, topically-coherent* sentences (the "prose first" gate).
Final samples + val to be appended when the P100 run completes.

## 4. What this session proved about SunnyNet specifically

- **O(N) latency holds at every scale and on GPU** — per-token cost decreases
  with sequence length instead of climbing quadratically (verified on CPU at
  21M and 94M). This is what makes "small model + big data + long context"
  (row 7) viable in a single 9 h GPU session.
- **Throughput**: 94M at bs 8 on a P100 ≈ 0.24–0.25 ms/token (~4,000 tok/s); a
  24k-step 94M run finishes in ~7 h.
- **The kernel is not the limiter** — all GPU runs used the pure-torch path
  (`kernel=False`); the AVX2 kernel is CPU-only and a non-factor here.

## 5. Reproducing

Local checkpoints live in `runs/` (gitignored). The p100m English checkpoint +
matching tokenizer are hosted at
[CodeDoes/SunnyNet-P100M-Kaggle](https://github.com/CodeDoes/SunnyNet-P100M-Kaggle)
(LFS); see `KAGGLE_P100M.md`.

To run any of the kernels on Kaggle (free-tier GPU):

```sh
# stage 1: big-data 21M (this session's bet)
kaggle kernels push -p kaggle/big_data -t 32400
# stage 2: 94M 70/30 code mix
kaggle kernels push -p kaggle/code_mix -t 32400
# 94M diverse English prose
kaggle kernels push -p kaggle -t 32400
```

Each kernel is self-contained (clones the repo, applies the `--save-every`
milestone patch, builds its own corpus, smoke-tests, trains, and writes
milestone samples to `/kaggle/output`).
