# SunnyNet

**English** · [中文](#中文简介)

An original general-purpose sequence architecture, not derived from Transformer, RNN, Mamba, CNN, or any existing design.

Repository: <https://github.com/SunnyClaw-cx/SunnyNet>

Built for CPU training: O(N) sequence complexity with a fixed-size rolling state, no token-to-token interaction, and no materialized activations across the sequence.

Licensed under Apache License 2.0.

---

## English Overview

### What it is

SunnyNet is built around a single recurrent primitive: a convex combination.

```
s_t = p_t ⊙ s_{t-1} + (1 - p_t) ⊙ u_t ,   p_t ∈ [1-δ_i, 1]
```

Both coefficients are non-negative and sum to one, so `|s_t| ≤ max(|s_{t-1}|, |u_t|)`. The state is bounded by construction and cannot diverge. Each channel's retention floor is tied to a memory span `H_i` laid out on a log scale over `[2, 8192]`, so one layer carries time scales from two tokens to eight thousand.

The design targets the real CPU bottleneck — memory bandwidth and cache behavior — not FLOPs. It avoids dense matmul, keeps the state dimension fixed, and passes only the state across chunks.

### Measured results

All numbers are measured on one machine (i7-8750H, AVX2, PyTorch 2.14.1 CPU) with per-cell process isolation. Full data, including negative results, is in [docs/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/EXPERIMENT_REPORT.md).

| Metric | SunnyNet | Best peer | Setting |
|---|---|---|---|
| Training step, ms/token | 0.0953 | RWKV 0.0982 | T=256, batch 8, tied over 9 paired rounds |
| Quality, ppl | 57.83 (2nd of 7) | H3 53.93 | 600 steps, single seed, not converged |
| Decode, ms/step at T=4096 | 2.85 | RWKV 1.56 | B=1, single sequence |
| Forward, ms/token at T=4096 | 0.0352 | Hyena 0.0184 | batch=1, pure forward |

- Training step is statistically tied with RWKV (3–4 of 9 rounds faster, median paired ratio 1.042) and 16–41% faster than H3, Mamba, Hyena, and linear attention. The tie holds only at L=4; at L=6 the same parameter budget is 30–40% slower.
- Long-context decoding: single-step cost stays flat from T=256 to T=4096 (3.28 → 2.85 ms), while a parameter-matched Transformer climbs 2.55 → 6.22 ms. SunnyNet is 2.18× faster at T=4096. Against O(N) peers it is not the fastest — RWKV decodes 1.9–2.6× faster at every length.
- Forward pass: crossover with a matched Transformer at T≈384 (6-layer) or T≈600 (4-layer). At T=4096 it is 22.7× faster per token. Against O(N) peers there is no length where it leads all of them.
- Cross-distribution: on independently collected external data, 1st of 7 on Chinese tech news, 2nd of 7 on English arXiv abstracts.
- Hand-written AVX2 kernel (competitive gating, gate probability, and scan fused into one pass): 1.10–1.17× on the training step, bit-identical loss. Needs a C compiler (w64devkit / GCC).

### Limitations

- Short sequences are a structural weakness: below T≈384 (batch=1 forward) a standard Transformer is faster. The cause is operator count, not FLOPs.
- No 0.5B quality number. On the 6 GB GPU available here, forward and a single SGD step run (5.0 GB peak); AdamW needs 7.6 GB and does not fit.
- The scaling trend is not monotonic: 2nd at 1.4M, 1st at 20M, 2nd at 100M.
- Long-range retrieval is verified only on structured text (code, far-bucket hit 0.13); on natural language it is near zero.
- No GPU kernel. The architecture runs on CUDA unmodified (8.0–15.2× forward on a GTX 1060), but there is no Triton or CUDA implementation.
- Decode speed is not competitive against O(N) peers: RWKV decodes 1.9–2.6× faster at every length tested.

### CPU inference runtime

`sunnynet/runtime/` is a standalone C kernel (AVX2/FMA + OpenMP) that has no Python or PyTorch dependency. It implements the general first-order recurrence

```
s_t = a_t * s_{t-1} + b_t              (FMA)
s_t = a_t * s_{t-1} + (1 - a_t) * u_t  (gated)
```

RWKV's wkv, Mamba/S6 and H3 diagonal SSM, S4, linear-attention KV accumulation, and SunnyNet's rolling state all reduce to one of these two forms. Each is mapped to the kernel and checked elementwise against its own PyTorch recurrence (max diff 1.2e-07 to 1.9e-06). Against torch's vectorized block scan (`gated_scan`, cumsum closed form), the kernel is 9.90× faster at the median (3.56–19.29×, single thread, B=1, per-cell process isolation).

The convex-combination state has a bounded magnitude by construction. Over T=2048 the state magnitude drifts 1.06× versus 7.94× for unbounded accumulation, and under a fixed INT8 scale the relative error is 4.0e-02 versus 5.0e-01.

The whole decode step — embedding lookup, per-layer norm, depthwise conv history, gated competition, state recurrence, readout, final norm, LM head — also runs in C with no intermediate tensors. Weights are exported from a checkpoint into a packed fp32 blob (matmul weights pre-transposed for GEMV, `circ ∘ W_s` folded into one matrix). Against `SunnyNet.forward_step`, per-token logits agree to 1.24e-05 and argmax matches 48/48. Whole-model decode is 1.53× / 2.05× / 1.97× faster than torch at prefill T=256/1024/2048 (B=1, single thread, 3 rounds, C wins 3/3 every cell).

CLI: `python -m sunnynet.cli --ckpt runs/small.pt --prompt "..."` (`--backend c` for the pure-C decode path).

Foreign weights: `sunnynet/runtime/gguf.py` reads GGUF and `rwkv7.py` runs RWKV-7 per token without PyTorch. Dequantized tensors are bit-identical to the official `.pth` (max diff 0), and per-token logits match an independent reference implementation to 4.79e-06 (argmax 12/12). The official RWKV-7 191M model generates coherent text (English, Chinese, and Python all verified). This path is numpy, not C — it demonstrates that the runtime consumes external weights correctly, and its ~388 ms/step is not a speed result. RWKV-7's state is a per-head matrix with a rank-1 removal term, so a matrix-valued recurrence layer is needed in the IR beyond the elementwise forms above.

---

## Complete English Documentation

### Table of Contents

- [Quick Start](#quick-start)
- [Core Ideas of the Architecture](#core-ideas-of-the-architecture)
- [Original Components](#original-components)
- [Measured Results](#measured-results)
- [Comparison with Six Major Architectures](#comparison-with-six-major-architectures)
- [CPU Inference Runtime for Linear Architectures](#cpu-inference-runtime-for-linear-architectures)
- [Repository Structure](#repository-structure)
- [Originality Statement](#originality-statement)
- [Risks and Limitations](#risks-and-limitations)
- [Sponsors](#sponsors)
- [License](#license)

---

## Quick Start

### Environment

- Python >= 3.10
- PyTorch >= 2.1 (the CPU build is enough, no GPU dependency)
- numpy

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pytest

# Install this package (release name sunnynet-cx, import name stays sunnynet)
pip install sunnynet-cx          # or pip install -e . from source
```

### 1. Download the corpus and build the token stream

```bash
# Collect real corpora: Project Gutenberg Chinese classical texts + English literature + TheAlgorithms/Python source code
python prepare_data.py

# Train the BPE tokenizer -> encode into a token stream -> split into train/validation
python build_corpus.py --vocab-size 8192
```

The validation set is split at file level: adjacent fragments of the same text never cross
between training and validation.

### 2. Train

```bash
# Small-model quick validation (4 layers)
python train/train.py --preset small --steps 800 --batch-size 8 --seq-len 256

# 20M prototype
python train/train.py --preset proto20m --steps 5000 --batch-size 4 --seq-len 512
```

### 3. Evaluate

```bash
python train/evaluate.py --ckpt runs/<yours>.pt --gen \
    --out experiments/results/eval_sunnynet.json
```

### 4. Full experiments

```bash
bash experiments/battery.sh          # baselines + ablations + seeds + scaling
bash experiments/battery2.sh         # 20M scaling comparison + long-sequence training comparison

# O(N) verification: forward latency vs sequence length
python experiments/long_sequence.py --only latency

python experiments/make_figures.py
python experiments/collect_results.py --out docs/_tables.md
```

The hand-written AVX2 kernels need a C compiler (w64devkit / GCC):
point `SUNNY_W64` at the w64devkit root, or put `gcc` on PATH.

```bash
export SUNNY_W64=<w64devkit unpack dir>
python -X utf8 experiments/_kernel_bench.py        # operator-level speedup
python -X utf8 experiments/_core_e2e.py 256 6 8    # fused-kernel training step A/B
```

---

## Core Ideas of the Architecture

The bottleneck of CPU pretraining is usually not floating-point throughput but memory bandwidth
and cache-hit rate. SunnyNet's design revolves around three hard constraints:

| Constraint | Design that answers it |
|---|---|
| Strictly O(N) sequence complexity | rolling state + first-order linear recurrence, no token-pair interactions at all |
| No materialized activations for the whole sequence | state dimension fixed, only the state crosses chunk boundaries; backward recomputes on demand |
| Avoid huge dense matrix multiplies | low-rank decomposition + depthwise-separable convolutions + tridiagonal circulant filter |

The core state update is a convex combination:

```
s_t = p_t ⊙ s_{t-1} + (1 - p_t) ⊙ u_t ,     p_t ∈ [1-δ_i, 1]
```

Both coefficients are non-negative and sum to 1, so `|s_t| ≤ max(|s_{t-1}|, |u_t|)`: the state
is bounded by construction and cannot diverge.

The per-channel retention floor `p_t` is set by the memory span `H_i`; `H_i` sweeps `[2, 8192]`
on a log scale, so a single layer carries every time scale from "remember 2 tokens" to "remember
8192 tokens", covering both local syntax and long-range dependencies across paragraphs or code
blocks.

The full math is in **[docs/ARCHITECTURE.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/ARCHITECTURE.md)**.

---

## Original Components

| Component | Design | Key property |
|---|---|---|
| State update | multi-timescale convex-combination rolling | state bounded, cannot diverge; multi-scale output gating |
| Nonlinear activation | SoftWedge `f(z)=z+a*s(z)`, base `s(z)=1/2+1/2*z/(1+\|z\|)` | derivative `f' ∈ (1, 1+a/2]`, nowhere zero, no gradient cliff |
| Normalization | CenteredScaleNorm | smooth lower bound `s_min` in the denominator; near-constant vectors are not amplified (prerequisite for deep stacking) |
| Initialization | spectral anchoring (QR-orthogonal slice, σ_max anchored directly) | condition number 1, flat forward-gain spectrum; ρ=1.0 empirically calibrated |
| Cross-layer propagation | state relay + feature residual stream | state scale does not grow with depth; the residual shortcut rescues the deep gradient (measured −28% ppl) |
| Channel competition | intra-group pairwise competition gating | same semantics as softmax, but the Jacobian is O(1) instead of O(1/d) |
| Parallel scan | log-domain closed form | algebraically equivalent to stepwise recurrence; the block-start state can be back-solved exactly from the block-end state |

No reuse of ReLU / GELU / SiLU / Mish, LayerNorm / RMSNorm / InstanceNorm / GroupNorm,
Xavier / Glorot / He / MSRA initialization, LSTM / GRU gating formulas, SSM / HiPPO
discretization.

---

## Measured Results

All experiments share the same token stream, the same configuration, the same token budget and
thread count. Parameter counts are aligned by meta-device instantiation, with layer-count
deviation from the reference within ±1. Full data:
[docs/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/EXPERIMENT_REPORT.md).

### Training speed

- **Same-architecture speedups**: operator fusion 19-41%; layer reduction + widening 12-17%;
  narrow residual stream + dropping the `p.clamp` 17.7-20.7%.
- **Vs same-complexity peers**: T=256 training step 0.0953 ms/token, nominally 3.0% faster than
  rwkv; 9 paired rounds are not statistically significant (wins in 3-4/9 rounds) => written as
  "statistically tied with rwkv", not "beats". A stable 16-41% lead over
  h3/mamba/hyena/linear_attn.
- **Boundary of the tie**: holds only at `L=4`; with layers aligned to `L=6` the same-budget
  narrow configuration is 30-40% slower; the T>=1024 peer advantage persists.
- **Hand-written kernel**: fuses competition gating + gate probability + scan; the three
  training-step A/B metrics agree in direction, T=256 wins 6/6, ratio 0.8746 (1.143x); ppl is
  bit-identical to torch (changes speed only, not quality).
  The inference-regime kernel benefit is not significant (B=1 tensors are too small, the
  cross-language call overhead is on the same scale as a single operator), off by default.

### Inference

- Single-step decode cost vs context: SunnyNet is flat (O(1), 2.87 -> 2.90 ms, drift ±5%);
  Transformer climbs monotonically (O(t), 2.49 -> 6.85 ms). Crossover at T≈512; at T=4096
  SunnyNet is **2.36x** faster, and the gap keeps growing with T. No kernel required.
- Short-context (T<512) decoding is not an advantage; kernel fusion brings no gain there
  either.
- Boundaries: single sequence, B=1, randomly initialized model, median of 2 rounds, batched
  decoding not measured, no KV-cache quantization/paging.
- See the O(N) peer comparison below: SunnyNet decoding is **not the fastest**; rwkv is 1st
  throughout, 1.9-2.6x faster.

### Quality

- After the architecture fixes (residual + CSN floor), T=256 ppl dropped from 93.82 to 57.83,
  2nd of seven (behind only H3 at 53.93); the rest: Mamba 63.34 / RWKV 64.10 / Hyena 82.50 /
  Transformer 90.05 / linear attention 91.91.
- After 20M layer reduction + widening, ppl dropped a further 10.9% (54.82), rising to 1st of
  five in the same-protocol comparison.
- **The scaling trend is not monotonic**: 2nd at small -> 1st at 20M -> back to 2nd at 100M
  (109.47, ~3.6% behind rwkv 105.66). One cannot say "bigger scale, bigger advantage".

### Long sequences and memory

- **Forward**: crossover at T≈384 (6-layer comparison) / T≈600 (4-layer comparison); at T=4096
  it is 22.7x faster per token than Transformer (batch=1, pure forward, 8 threads, 1.4M small
  model).
- **Training regime T=1024**: SunnyNet is 1st in speed (0.1319 ms/tok), 2nd-lowest peak memory
  at 1812 MB (Transformer 3192 MB, 76% higher).
- **Forward vs O(N) peers**: no stable lead interval; it climbs from 6th to 3rd as T grows,
  never 1st throughout; the training-regime ranking differs (1st at T=1024).

### Cross-distribution generalization

External, independently collected datasets (trained on the original three domains, tested on the
new domain): 1st/7 on Chinese tech news ppl 321.08 (that domain's out-of-vocabulary character
rate is 23%), 2nd/7 on English arxiv abstracts at 35.58; the OOD drop is 1.480, the lowest
band of the seven architectures (tied with hyena).
Absolute ppl is still high, 600 steps are unconverged, single seed - one can only say
"cross-distribution is no worse than same-complexity peers".

### Short sequences

The T≈384 crossover holds only in batch=1 pure forward. In the training regime (batch=8) the
crossover drops to T<256: at T=256/384 it is 1.5-2.4x faster than Transformer. T=128 is still
10.4% slower (old d128) / 4.5% slower (narrow configuration smalln80) - the narrow residual
stream is a verified mitigation.

### GPU

The architecture runs on CUDA without any CUDA-specific rewrite. B=8 speedups: forward
8.0-15.2x, training step 9.9-12.5x (no CUDA-side optimization done, so this is a lower
bound). The local GPU (GTX 1060 6GB) was used only for correctness verification and
speedup measurement; there is no peer GPU comparison.

### 100M-scale GPU training and weight release

`p100mw` (d1408/i1920/r1024/L8, stack 93.83M / total 105.36M) ran 2000 real training steps on
a GTX 1060 6GB (`zh_ancient` 0.4 / `en_literature` 0.35 / `code_python` 0.25,
bs=2 x seq=256 x 2000 = 1,024,000 tokens, lr 1e-3 + 100-step warmup):

| Step | train loss | val loss | ppl |
|---|---|---|---|
| 200 | 5.8046 | 5.7407 | 311.29 |
| 600 | 5.1452 | 5.3599 | 212.71 |
| 1000 | 4.8508 | 5.1040 | 164.68 |
| 1400 | 5.9248 | 4.8997 | 134.25 |
| 2000 | 2.9600 | 4.7882 | **120.09** |

val decreased monotonically, no divergence, **0.503 ms/token** throughout (1987 tok/s), wall
583.5s, peak_rss 1302MB.

**On the 6GB card only bs=2 x seq=256 is usable** (measured boundary):

| Configuration | Result |
|---|---|
| bs=1 x 256 | passes, 1.220 ms/tok |
| **bs=2 x 256** | **passes, 0.744 ms/tok (best)** |
| bs=4 x 128 | passes, 0.818 ms/tok |
| bs=4 x 256 / bs=8 x 128 | OOM in late training |

Three measurement notes: `expandable_segments` is **unavailable** on sm_61 - enabling it OOMs
even earlier; when other processes (e.g. a resident Ollama) occupy VRAM, `free` looks
sufficient but allocation fails - unload with `keep_alive: 0` first; a validation batch of
`val-batches 8` also OOMs, dropping to 2 is stable.

Weight packaging is done by `sunnynet-release` (`dist/release/sunnynet-p100mw.*`):

| Artifact | Size | First 12 sha256 chars |
|---|---|---|
| `.pt` checkpoint | 467.7 MB | `399e604e635b` |
| `.safetensors` (139 tensors, bound head deduplicated) | 421.5 MB | `12784fbfadf7` |
| `.runtime.bin` (C runtime fp32 layout) | 467.6 MB | `95c195e47637` |

All three are verified to load and generate: torch backend "H不得你，也不可不得…" (143.6
ms/tok), C backend "尚與子之子，雖以一名，於我也。" (**80.2 ms/tok**), the OpenAI service
passes all four official-SDK call patterns. **1520-step ceiling, single seed - no
"converged" or cross-architecture quality claims are made.**

### 0.5B measured boundary on the 6GB card

| Tier | Peak VRAM | Result |
|---|---|---|
| Build (498M params, fp32) | 1902 MB | passes |
| Forward T=512 | 2010 MB | passes |
| Forward T=1024 | 2060 MB | passes |
| **SGD single step** (T=512) | **5004 MB** | **passes**, but only 563 MB (9% headroom) left |
| AdamW single step | >6144 MB | **OOM** |

The constraint comes from the optimizer-state arithmetic: fp32 params 1.9GB + gradients 1.9GB
+ AdamW first/second moments 3.8GB = 7.6GB > 6GB; SGD has only the gradient term, so it
barely fits.
CPU offload of optimizer state failed on this machine - the host has only 4.5GB physical RAM
left, cannot hold the 3.7GB offloaded optimizer state, triggering page-file pressure that
crashed the CUDA context.
=> **0.5B can run a single SGD step on this machine, but there is no multi-step convergence or
quality data.**

Measurement note: running the T=512/T=1024 forwards and then a training step inside the same
process makes the training step OOM because of VRAM fragmentation (measured: 1.9GB remaining
still cannot allocate 16MB); **each tier must run in an independent subprocess**, otherwise a
wrong "infeasible" conclusion results.

---

## Comparison with Six Major Architectures

Compared against: `h3`, `rwkv`, `mamba`, `hyena`, `linear_attn`, `transformer`.
All numbers are local measurements with per-cell subprocess isolation. Full data: section 23 of
[docs/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/EXPERIMENT_REPORT.md).

### Scorecard

| Dimension | SunnyNet position | Best | Gap |
|---|---|---|---|
| Quality ppl (1.4M / 600 steps) | 2nd (57.83) | h3 53.93 | 7.2% behind |
| Quality ppl (20M / 200 steps) | **1st** (77.90) | sunnynet | — |
| Quality ppl (100M / 100 steps) | 2nd (109.47) | rwkv 105.66 | 3.6% behind |
| Training speed (L=4, matched stack params) | statistically tied with rwkv | rwkv (nominal) | paired test not significant |
| Training speed (L=6 layer-aligned) | **behind** | rwkv | 30-40% slower |
| Training-regime memory (T=1024) | 2nd lowest | rwkv 1611MB | 12% higher, 43% lower than TF |
| Forward ms/token (batch=1) | 6th -> 3rd (with T) | hyena | never 1st throughout |
| **Decode ms/step (B=1)** | 5th -> 2nd | **rwkv** | 1.9-2.6x slower |
| Long-range retrieval (code domain) | tied with TF | tied | — |
| Long-range retrieval (natural language) | near 0 | — | no advantage |
| Cross-distribution generalization | 1st on new Chinese / 2nd on new English | sunnynet | — |

### Decode-regime seven-architecture comparison

Protocol: each (architecture, T) in an independent subprocess; first run T single-token steps to
build a real decode state (untimed), then measure single-step decode ms at that length; 1
thread; median of 2 rounds.

ms/step (lower is better):

| Architecture | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| `rwkv` | **1.768** | **1.574** | **1.598** | **2.113** | **1.564** |
| `h3` | 2.517 | 2.524 | 2.487 | 2.969 | 3.227 |
| `transformer` | 2.550 | 2.667 | 3.426 | 5.049 | 6.219 |
| `mamba` | 2.686 | 2.742 | 2.573 | 2.773 | 3.205 |
| **`sunnynet`** | **3.279** | **4.067** | **3.086** | **3.331** | **2.854** |
| `hyena` | 3.346 | 3.583 | 3.662 | 5.101 | 4.530 |
| `linear_attn` | 4.061 | 3.452 | 4.599 | 3.676 | 3.624 |

Reading:

- The Transformer O(t) behavior is reproduced (2.550 -> 6.219 ms); at T=4096 SunnyNet is 2.18x
  faster.
- SunnyNet's per-step cost is roughly flat and its ranking improves with T (5->7->4->4->2), but
  it **never catches rwkv**.
- **The decode-regime best is rwkv** (1.57-2.11 ms): its single step is one elementwise
  `p*s + k*v` chain; SunnyNet's single step includes competition gating (3 sigmoids), gate
  probability, scan, and the feedback branch - several times the operator count.
- The five O(N) peers had no single-step decode interface before; this repository implements
  `init_state`/`step` for them, with maxdiff <= 4e-05 against the full-pass forward.

### The common root cause of the gaps

The dimensions where SunnyNet is behind (decode speed, forward speed, L=6 training speed, 100M
quality, natural-language long-range retrieval) converge on one root cause: **the number of
elementwise operators**. Evidence: 4524 aten calls per training step; layer reduction +
widening (matched stack params) trains 12-17% faster and ppl −10.9%; layer alignment to L=6
makes the same-budget narrow configuration 30-40% slower; in the decode regime rwkv wins with a
single elementwise chain. => The only effective direction is reducing operator count - not
widening, batch tuning, or switching matrix libraries.

---

## CPU Inference Runtime for Linear Architectures

`sunnynet/runtime/` is a C inference kernel independent of PyTorch (AVX2/FMA + OpenMP),
depending only on ctypes and a C compiler. It implements the general first-order recurrence

```
s_t = a_t * s_{t-1} + b_t              (FMA)
s_t = a_t * s_{t-1} + (1 - a_t) * u_t  (gated)
```

RWKV's wkv, the Mamba/S6 and H3 diagonal SSMs, S4, the linear-attention KV accumulation, and
SunnyNet's rolling state all reduce to one of these two forms.

### Cross-architecture mapping (`experiments/_unified_ir_map.py`)

Each architecture's recurrence is mapped to kernel parameters and checked elementwise against
its own PyTorch recurrence, B=2/T=128/N=128/fp32:

| Architecture recurrence form | Mapping | max\|diff\| |
|---|---|---|
| RWKV-4 wkv | `a=p`, `b=k*v` | 9.5e-07 |
| Mamba / S6 SSM | `a=decay`, `u=bu` | 2.4e-07 |
| H3 diagonal SSM | `a=decay`, `u=v` | 1.2e-07 |
| Linear attention KV accumulation | `a=1`, `b=k v^T` | 0.0 |
| SunnyNet rolling state | `a=p`, `u=u` | 1.9e-06 |
| S4 diagonal SSM | `a=A_bar`, `b=B_bar x` | 7.2e-07 |

Hyena is a long-convolution (FIR) recurrence, not a first-order state recurrence; it needs a
separate convolution backend.

### Speed (`experiments/_runtime_bench.py`)

Per-cell subprocess, single thread, B=1, median of 10 runs. The baseline is torch's vectorized
block scan (`ops.gated_scan`, cumsum closed form), not a Python loop.

| T | N | Mode | Vectorized ms | Kernel ms | Multiple |
|---|---|---|---|---|---|
| 256 | 128 | FMA | 0.391 | 0.043 | 9.08x |
| 1024 | 128 | FMA | 1.256 | 0.132 | 9.54x |
| 4096 | 128 | GATED | 8.500 | 0.806 | 10.55x |
| 4096 | 640 | FMA | 26.944 | 7.574 | 3.56x |

Median **9.90x**, range 3.56-19.29x. At N=640 it drops to 3.6-4.0x (single-thread
memory-bandwidth limit). Max numeric difference against the kernel's three branches: 7.63e-06.

**Boundary**: this is a **batched cross-segment scan** regime (B=1, whole T). Once wired into
B=1 single-step decoding the gain disappears (four-cell ratios 1.023 / 1.072 / 0.897 / 1.073,
mixed direction, 8 threads T=256 actually 10.3% slower) => no end-to-end decode speedup may be
claimed. Reason: on the model side T=1, B=1, so the cross-language call and state-shuffle cost
is on the same scale as a single operator. B=1 single sequence is also not this runtime's target
regime (the decode-regime best is rwkv).

### Whole-model C decode (`sunnynet/runtime/sunnynet.c`)

On top of the single-step operator kernel, the entire forward was moved into C: embedding
lookup, per-layer norm / depthwise-conv history / gated competition / state recurrence /
readout, final norm, LM head - all in one pass with no intermediate tensors. Weights are
exported from the ckpt by `experiments/_export_runtime.py` into a packed fp32 blob (matmul
weights pre-transposed to `(in, out)` for GEMV, `circ W_s` folded into one matrix).

Numeric equivalence (`experiments/_runtime_model_equiv.py`, 1.4M ckpt, 48 tokens stepwise):

| Regime | Result |
|---|---|
| Single-step logits max\|diff\| | 1.24e-05 |
| argmax agreement | 48 / 48 |
| Prefill last token vs torch full forward | 3.81e-06 |

End-to-end decode latency (`experiments/_runtime_decode.py`, B=1 single sequence, single
thread, per-cell subprocess, full prefill run first then single steps measured, 3 interleaved
rounds):

| Prefill T | torch ms/step | C ms/step | Ratio | Per-round wins |
|---|---|---|---|---|
| 256 | 3.796 | 2.487 | 1.53x | 3/3 |
| 1024 | 3.802 | 1.851 | 2.05x | 3/3 |
| 2048 | 4.309 | 2.189 | 1.97x | 3/3 |

The median ratio and the per-round sign test agree (all four cells 3/3) - a real improvement,
not inter-round noise.
The speedup does not come from operator fusion but from eliminating ~1100 aten calls / 21 GEMVs
/ 20 cats per step (torch single-step profile: `aten::mm` 16.4%, `aten::mul` 10.2%,
`aten::select` 8.5%, `aten::cat` 8.0%).

**Regime boundary**: the table above is **single-thread B=1**. The CLI default of 8 threads
gives C backend 4.40 ms/tok vs torch 4.56 ms/tok (24 tokens, single reading, no multi-round) -
the gap is smoothed out by thread scheduling; B=1 small-tensor parallelism is a negative
result (§20.2 already recorded 1 thread 1.42x faster than 8 threads). The C backend's
advantage is at single thread and larger batches; do not linearly extrapolate the table above
to multithreaded B=1.

### Memory-access restructuring for large-model decode (100M scale)

The table above is a 1.4M-scale ckpt. Switching to `p100mw` (105.36M params, fp32 blob
467.6 MB), a single step is 80 ms - clearly the wrong order of magnitude. Quantification
confirmed the bottleneck is the **weight layout**, not the operators: a step reads all weights
at least once, and the original implementation's GEMV inner loop strides reads across `out`,
giving an effective bandwidth of only 6.17 GB/s (measured AVX2 sequential-read ceiling on this
machine: 25.31 GB/s).

`sn_load` flips the exported `(in,out)` blocks in memory to `(out,in)` row-major (the export
format is unchanged, published blobs need no re-export); GEMV is parallelized per output
channel, reading each row contiguously:

| Regime | Result |
|---|---|
| Before | 76.60 ms/token, 6.10 GB/s |
| After | **25.72 ms/token, 18.18 GB/s** |
| Speedup (5 interleaved head-to-head rounds) | **2.98x**, C wins 5/5 |
| Logits max\|diff\| vs pre-optimization | 2.766e-05 |
| argmax agreement | 32 / 32 |

Parallelism takes **physical cores, not logical cores**: decode is memory-bound, and
hyperthreading adds no bandwidth. Measured: 6 cores 26.69 ms vs 12 logical cores 36.05 ms
(26% gap). The C-side `sn_recommended_threads()` uses `GetLogicalProcessorInformation` on
Windows and parses `/proc/cpuinfo` on Linux; the CLI adopts it automatically when
`--backend c` is set without `--threads`.

**Boundary**: the 2.98x is the single-step decode multiple on this machine
(i7-8750H dual-channel DDR4-2667); it only means memory efficiency moved from 6.1 to
18.2 GB/s. 18.18 is still below the 25.31 ceiling (per-step non-streaming access like
normalization remains). Do not extrapolate to training or prefill.

### fp16 weights: per-step read bytes halved

After row-major, a single step is still memory-bound and the kernel has reached the pure
sequential-read ceiling; the remaining lever is reading fewer bytes per step.
The 5 GEMV matrices and the LM head are truncated to fp16 **at load time** (smaller parameters
stay fp32), and the per-step actual read drops from 467.6 MB to about 222.3 MB:

| Regime | fp32 | fp16 |
|---|---|---|
| Median ms/token | 24.67 | **18.60** |
| Effective bandwidth (fp32-equivalent bytes) | 18.95 GB/s | 25.14 GB/s |
| Speedup (11 interleaved head-to-head rounds) | — | **1.33x**, fp16 wins 11/11 |
| Logits max\|diff\| / relative deviation | — | 4.181e-03 / 7.42e-04 |
| argmax agreement | — | 64 / 64 |

The kernel reads 16 halves at a time (32 B), converts to fp32, and does the FMA, keeping the
same load width as fp32; a 128-bit load loses width (single-thread 17.45 -> 15.06 ms), so the
256-bit form is used.

fp16 is no longer memory-bound (actual read 222.3 MB / 18.6 ms = 12.0 GB/s, far below fp32's
19); the bottleneck moves to type conversion. Usage: `--backend c --half`; fp32 remains the
default and is bit-identical.

**Boundary**: the 1.33x is this machine's single-step number - do not extrapolate to prefill or
training; the 7.4e-04 relative deviation makes long-sequence greedy decoding diverge
occasionally; the 64/64 on this sequence is a sample - use fp32 for bit-exact reproduction;
the weight files on disk are still fp32 blobs, fp16 is a load-time conversion only.

### State quantization (`experiments/_state_quant.py`)

The convex-combination state magnitude is bounded by construction. T=2048, real ckpt layer-0
state:

| Mode | \|s\| drift (end / at 1/8) | Fixed-scale relative error | Overflow channel ratio |
|---|---|---|---|
| Bounded recurrence | 1.06x | 4.0e-02 | 0.78% |
| Unbounded accumulation | 7.94x | 5.0e-01 | 5.47% |

A 12.5x gap in relative error at a fixed scale.
Boundary: layer 0 only, single ckpt (1.4M), single random sequence, no end-to-end ppl
comparison.

### GGUF read-only (`sunnynet/runtime/gguf.py` + `rwkv7.py`)

Loads external GGUF weights and generates autoregressively per token without PyTorch. The
first model wired up is the official RWKV-7 191M (`BlinkDL/rwkv-7-world` `.pth` converted to
GGUF via llama.cpp, bf16).
Tensor naming and the recurrence form come from llama.cpp's `src/models/rwkv7*.cpp`; the
reference math from BlinkDL/RWKV-LM's `RWKV-v7/rwkv_v7_numpy.py`.

Numeric verification (`experiments/_rwkv7_gguf_equiv.py`): the GGUF dequantization result is
**bit-identical** to the official `.pth` (`max|diff| = 0`); per-token single-step logits
against the independent reference implementation:

| Regime | Result |
|---|---|
| Single-step logits max\|diff\| | 4.79e-06 (within bf16 precision) |
| argmax agreement | 12 / 12 |

Generation samples (`experiments/_rwkv7_gguf_bench.py`, temperature 0.7 / top-k 20 / seed 42):

```
The Eiffel Tower is located in the city of
-> Paris, France. The Eiffel Tower was designed by Gustave Eiffel in 1887 as a monument to
def fibonacci(n):
-> \n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)
```

**Regime boundary**: the numpy path proves "the runtime can read external GGUF weights and get
the math right"; decoding is ~388 ms/step - **this number is not claimed as a speed result**.
RWKV-7's recurrence has been moved into the C kernel (see below).

One divergence from the existing narrative must be recorded: RWKV-7's state is a **per-head
matrix** `S ∈ R^{64x64}`, updated as
`S_t = S_{t-1}⊙w_t − (S_{t-1}k_t)k_tᵀ + v_t k_tᵀ`, including a rank-1 removal term -
**not** the elementwise `s = a⊙s + b`. To cover it, the unified runtime's IR needs a
matrix-valued recurrence layer (outer product / rank-1 correction) on top of the elementwise
forms.

### RWKV-7 C kernel (`sunnynet/runtime/rwkv7.c`)

The whole-model single step moved into C: embedding lookup, per-layer norm / time-mix (with
LoRA and `tanh` / `sigmoid`), per-head matrix state recurrence (rank-1 correction),
group-norm, channel-mix, final norm, LM head - all in C, one call per step. Weights are
exported from GGUF by `experiments/_export_rwkv7.py` into a packed fp32 blob, matrices stored
as `(in, out)` for GEMV.

Numeric equivalence (`experiments/_rwkv7_c_equiv.py`, stepwise against the numpy reference):

| Regime | Result |
|---|---|
| Single-step logits max\|diff\| | 7.25e-05 (within bf16 precision) |
| argmax agreement | 12 / 12 |
| 120-step long-range argmax agreement | 120 / 120 |

Decode speed (`experiments/_rwkv7_c_bench.py`, single sequence, state established then single
steps, per-cell subprocess, 5 interleaved rounds, median of 60 steps):

| Implementation | ms/step | Relative to numpy |
|---|---|---|
| numpy reference | 366.36 | 1.00x |
| C kernel, 1 thread | 67.92 | 5.39x (wins 5/5 per round) |
| C kernel, 4 threads | 52.26 | 7.01x |
| C kernel, 8 threads | 48.88 | 7.49x |

The C kernel at 8 threads is another 1.39x faster than 1 thread (wins 5/5 per round).
Single-thread variance is large (50-84 ms); absolute values float with machine load, and the
ratio and per-round sign test agree.

CLI: `--gguf-backend c` uses the C kernel (run `_export_rwkv7.py` first to export the blob):

```bash
python -X utf8 experiments/_export_rwkv7.py \
    experiments/gguf/rwkv7-191M-world-bf16.gguf \
    experiments/gguf/rwkv7-191M-world-bf16.runtime.bin
python -m sunnynet.cli --gguf experiments/gguf/rwkv7-191M-world-bf16.gguf \
    --gguf-backend c --prompt "The Eiffel Tower is located in" --max-new 40 --stats
```

### INT4 weight quantization (`sunnynet/runtime/quant.c`)

Grouped symmetric INT4: every 64 consecutive weights share one fp32 scale, the quantized value
range is [-8, 7], two values per byte.
The quantized GEMV quantizes activations to INT8 with the same grouping and does the dot
product with `maddubs` (`Σ(qᵢ−8)xᵢ = Σqᵢxᵢ − 8Σxᵢ`), keeping the weights unsigned [0,15]
directly in the first operand.
The whole-model C kernel mixes fp32 / INT4 per matrix flag (vectors and embedding tables stay
fp32).

Export: `experiments/_export_rwkv7.py` (without `--f32` gives INT4, output `.q4.bin`).

| Regime | fp32 | INT4 |
|---|---|---|
| Blob size | 729.1 MB | 269.0 MB |
| Single-step decode (single-thread median) | 49.16 ms | 17.35 ms |

INT4 end-to-end **2.83x** (wins 5/5 per round), size compressed 2.7x.
The integer path is **bit-identical** to the explicit integer formula
(`tests/test_quant_i4.py`); the relative error against fp32 GEMV is ~0.11 (expected for
4-bit).
Generation quality: English stays coherent ("Paris, France..."); Chinese and code degrade -
**4-bit compression that wants quality must add calibration / mixed precision**; here only the
"runs and is fast" result is published, no quality-lossless claim.

CLI: `--quant int4` (combined with `--gguf-backend c`).

### External weights: Mamba-2 and DeltaNet (`sunnynet/runtime/mamba2.py`, `deltanet.py`)

The same runtime interface loads the official weights of two more linear architectures, doing
per-token single-step recurrence without a PyTorch forward.

**Mamba-2** (`state-spaces/mamba2-130m`, d_model 768 / 24 layers / 24 heads x 64):

```bash
python -m sunnynet.cli --hf-dir <model dir> --hf-arch mamba2 \
    --tokenizer-dir <GPT-NeoX tokenizer dir> --prompt "def fibonacci(n):" --stats
```

Each head state is a `(head_dim=64, state_size=128)` matrix, but the decay
`dA = exp(dt·A)` is uniform over `(p, n)` and the outer product factors, so it actually
degrades to an elementwise first-order recurrence:

```
dA  = exp(dt_h · A_h)                    # scalar
S_t = S_{t-1} ⊙ dA + (dt_h · B_g) ⊗ x_h  # no matrix-vector product needed
```

- Against `transformers`' `Mamba2ForCausalLM` full forward: max|diff| **5.9e-04**, argmax
  **24/24**.
  The official fp32 itself deviates **3.8e-04** from fp64; this implementation deviates
  **2.5e-04** (closer to the fp64 ground truth)
  => the difference is single-precision accumulation, not a formula error.
- Segmented continuation vs full run: **bit-identical 0.0**; greedy continuation of English
  and code is coherent.

**DeltaNet** (`fla-org` DeltaNet 340M, hidden 1024 / 24 layers / 8 heads x 128, short conv
4):

```bash
python -m sunnynet.cli --hf-dir <model dir> --hf-arch deltanet \
    --prompt "The capital of France is" --stats
```

A true delta rule, state `(k_dim=128, v_dim=128)`, update with rank-1 correction:

```
δ_t = (v_t − S_{t-1}ᵀ k_t) · β_t
S_t = S_{t-1} + k_t ⊗ δ_t
o_t = S_tᵀ q_t
```

- The rank-1 form against the independent implementation of the explicit
  `(I − β k kᵀ) S + β k vᵀ` writing: **max|diff| 4.6e-05**
  (fp32 accumulation scale). Note: when `head_k_dim == head_v_dim`, multiplying `S` on the
  wrong side does not raise a dimension error - it silently computes a different formula, so
  the two writings must be verified separately.
- Segmented continuation vs full run: **bit-identical 0.0**; greedy output
  `"<s>The capital of France is the city of Paris. It is the largest city in Europe..."`.

**Speed (single-thread numpy reference path, per-cell subprocess x 5 rounds)**:

| Architecture | median ms/token | min | max | State |
|---|---|---|---|---|
| mamba2 | 45.68 | 38.27 | 55.42 | elementwise recurrence, **can move into the C kernel reusing `recur.c`** |
| deltanet | 91.46 | 83.93 | 99.19 | matrix-valued rank-1 recurrence, **cannot be expressed by the existing kernels** |

These are the reference path's absolute times; **no multiple is compared against any
baseline**.
Both have a single-step safety upper bound of
`O(heads x head_dim x state_size)`, independent of context length.

**The matrix-valued recurrence is the only layer not yet in C**: RWKV-7's and DeltaNet's
state updates both include an outer-product accumulation like `S ← S + k ⊗ (·)`, which the
`SN_FMA`/`SN_GATED` per-channel forms cannot express; a new rank-1 recurrence kernel is
needed; Mamba-2 and GLA factor their decay outer products, so they do not need one.

**GLA** (`fla-hub/gla-340M-15B`, hidden 1024 / 24 layers / 4 heads x 128, gate_rank 16):

```bash
python -m sunnynet.cli --hf-dir <model dir> --hf-arch gla \
    --prompt "The capital of France is" --max-new 24 --stats
```

GLA's gate `gk = logsigmoid(·) / 16` acts only on the key dimension; the decay factors into
per-row scaling, so the state recurrence degrades to elementwise, the same class as Mamba-2:

```
S_t = diag(exp(gk_t)) · S_{t-1} + k_t ⊗ v_t
o_t = scale · S_tᵀ q_t,  scale = head_k_dim^{-1/2}
```

After the readout comes `RMSNorm(o) · silu(g)` (normalize before gate,
`norm_before_gate=False`).

- Against the independent reference implementation: max|diff| **2.05e-05**, argmax **12/12**;
  segmented continuation **bit-identical 0.0**.
- Greedy continuation: `"The capital of France is the city of Paris."`; confirmed under
  `--stats`.
- The decay factorization is asserted directly by a test
  (`test_gla_decay_is_elementwise_decomposable`), preventing later mistaken treatment as a
  matrix-valued recurrence.

### OpenAI-compatible HTTP service (`sunnynet/serve.py`)

Zero-dependency implementation (stdlib `ThreadingHTTPServer`, no fastapi/flask);
any OpenAI SDK or client can point at this service:

```bash
sunnynet-serve --ckpt <checkpoint.pt> --host 127.0.0.1 --port 8080
sunnynet-serve --gguf <model.gguf> --gguf-backend c --port 8080   # RWKV-7
sunnynet-serve --hf-dir <model dir> --hf-arch gla --port 8080     # GLA
```

| Endpoint | Notes |
|---|---|
| `GET /v1/models` | model list (`owned_by=sunnynet`) |
| `POST /v1/chat/completions` | supports `stream: true` (SSE) |
| `POST /v1/completions` | legacy completions |
| `GET /health` | liveness probe |

Backend selection goes through the `Engine` abstraction (`HFEngine` / `GGUFEngine` / `GGUFCEngine` / `TorchEngine`); sampling (temperature / top-k / top-p) is handled uniformly on the service side.

Measured (after local startup): the official `openai` SDK passes all four call patterns — `models.list`, non-streaming chat, streaming chat, legacy completions; 400 (missing field) / 404 (unknown path) errors handled correctly.

### Ollama / LM Studio integration (`sunnynet/integrate.py`)

Both run llama.cpp, so integration requires GGUF and a runtime-supported architecture. RWKV-7 GGUF has `general.architecture=rwkv7`, `tokenizer.ggml.model=rwkv`; Ollama recognizes it directly (`family=rwkv7`, `parameter_size=191.13M`, `runner=llamacpp`).

```bash
# Generate the Modelfile (defaults to printing commands only; --run actually imports)
sunnynet-integrate ollama <model.gguf> --name sunnynet-rwkv7 --run --smoke
# LM Studio: put the GGUF in the models dir + generate integration config
sunnynet-integrate lmstudio --gguf <model.gguf> --models-dir <LM Studio models>
```

Import and generation verified:

```bash
OLLAMA_MODELS=<models> OLLAMA_HOST=127.0.0.1:11511 ollama serve &
sunnynet-integrate ollama <model.gguf> --name sunnynet-rwkv7 --run --smoke
# [smoke] OK: 'The capital of France is Paris.'
```
Three pitfalls that must be recorded:
1. **TEMPLATE must be written explicitly**. The Ollama built-in `rwkv-world` template wraps the prompt in an extra `User: / Assistant:` layer, degenerating output into repetition (`-of-the-world-...`). Writing `TEMPLATE """User: {{ .Prompt }}\n\nAssistant:"""` restores output to " The capital of France is Paris.", byte-identical to the in-house C runtime.
2. **`Error: could not locate ollama app` actually means the server is not running**. `ollama create` tries to start the local app when the server is offline and fails, and the error text is misleading. This tool probes `GET /api/version` first and, when offline, prints an `ollama serve` hint directly.
3. **`OLLAMA_MODELS` must match the server side**. If the CLI only sets `OLLAMA_HOST`, `create` writes blobs into the default models directory, mismatching the running server directory.
4. The local probe must bypass the system HTTP proxy (enterprise proxies turn `127.0.0.1` into a 502); the code builds the opener with `ProxyHandler({})`.

Another route where LM Studio needs no GGUF conversion: point `sunnynet-serve` at a custom OpenAI-compatible endpoint (`Developer` -> add endpoint); the config is generated by `sunnynet-integrate lmstudio`.
### AVX-512 and NEON kernels (`sunnynet/kernels/scan_avx512.c`, `scan_neon.c`)

Two more SIMD backend sources besides AVX2/FMA, each with forward scan, backward scan, and
single-step entry points.

AVX-512 (F/BW/VL): uses `__mmask16` for masked tail handling, exports a CPUID probe
(leaf 7 EBX bit16) and the ISA string. **The host CPU measured AVX512F=0**
(i7-8750H, Coffee Lake), so only compile and artifact verification are done — no execution
verification:

| Check | Result |
|---|---|
| Compile | `gcc -O3 -mavx512f -mavx512bw -mavx512vl -shared` passes |
| Disassembly | 198 `zmm` uses, 16 `vfmadd132ps` |
| Load guard | when unsupported, `load_avx512()` raises immediately, avoiding an illegal instruction |

NEON: no ARM toolchain, no QEMU (`wsl.exe` is blocked by security policy); cross-compile with
`zig cc` instead:

```bash
pip install ziglang
python -c "from sunnynet.kernels.simd import cross_compile_neon; cross_compile_neon('<out.so>')"
```

| Check | Result |
|---|---|
| Artifact | AArch64 ELF64 (`e_machine=183`, `zig cc -target aarch64-linux-gnu`) |
| Disassembly | capstone confirms 4-lane `fmla v3.4s, v2.4s, v4.4s` plus `fmul`/`fsub`/`fadd` |

**NEON cannot be executed on this machine**; real execution verification is on the CI
`ubuntu-24.04-arm` runner (native aarch64).
### Stable C ABI (`sunnynet/runtime/sn_abi.h`, `sn_abi.c`)

The previous C interface was for internal use — changing a struct broke callers. A new stable
ABI layer isolates:

- Symbols unified under the `sn_abi_` prefix, handles are opaque (`SnAbiRwkv7*`); callers do not
  depend on internal layouts
- Version `SN_ABI_VERSION = (MAJOR<<16)|(MINOR<<8)|PATCH`, currently **1.0.0**
- Capability bits `SN_CAP_*`, error codes `SN_OK=0` / `SN_E_INVAL=-1` /
  `SN_E_UNSUPPORTED=-2`, etc.
- `sn_status` is unified to `int32_t`; functions return status codes only, no exceptions

The ctypes bindings live in `sunnynet/runtime/abi.py` and are **bit-identical** to the internal
handle path (`max|diff| = 0.0`). All 9 contract tests pass (version negotiation, capability
bits, error codes, `check(1)=OK` / `check(9)=E_UNSUPPORTED`, RWKV-7 step equivalence). See
[docs/ABI.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/ABI.md).
### Command line

```bash
# torch backend
python -m sunnynet.cli --ckpt experiments/ckpts_ood/sunnynet.pt \
    --prompt "子曰：" --max-new 80 --stats

# pure C backend (export the weight blob first, then decode)
python -X utf8 experiments/_export_runtime.py \
    experiments/ckpts_ood/sunnynet.pt experiments/ckpts_ood/sunnynet.runtime.bin
python -m sunnynet.cli --ckpt experiments/ckpts_ood/sunnynet.pt \
    --backend c --prompt "子曰：" --max-new 80 --stats

# external GGUF weights (RWKV-7)
python -m sunnynet.cli --gguf experiments/gguf/rwkv7-191M-world-bf16.gguf \
    --prompt "The Eiffel Tower is located in the city of" --max-new 40

# external HF weights (Mamba-2 / DeltaNet / GLA)
python -m sunnynet.cli --hf-dir <model dir> --hf-arch mamba2 \
    --tokenizer-dir <GPT-NeoX tokenizer dir> --prompt "def fibonacci(n):" --stats
python -m sunnynet.cli --hf-dir <model dir> --hf-arch deltanet \
    --prompt "The capital of France is" --max-new 20 --stats
python -m sunnynet.cli --hf-dir <model dir> --hf-arch gla \
    --prompt "The capital of France is" --max-new 24 --stats

# OpenAI-compatible service
sunnynet-serve --ckpt <checkpoint.pt> --port 8080

# Ollama / LM Studio integration config
sunnynet-integrate ollama <model.gguf> --name sunnynet-rwkv7
sunnynet-integrate lmstudio --gguf <model.gguf>

# weight release export (safetensors + runtime blob + model card + sha256)
sunnynet-release <checkpoint.pt> --out-dir dist/release
```

The loading entry point can also be called directly from Python:

```python
from sunnynet.runtime import load_model
m = load_model("deltanet", "<model dir>")   # or "mamba2" / "rwkv7"
st = m.reset(m.init_state())
logits = m.step(token, st)                 # O(1) per step
```
### Roadmap

Existing assets: original architecture, AVX2 fused kernels (training step 1.10–1.17×), general
recurrence C runtime (9.9× median vs the vectorized baseline), **whole-model C single-step decode
(B=1 single-thread 1.53–2.05×)**, cross-architecture mapping verification, **GGUF read-only +
RWKV-7 weights wired up (numerics equivalent to the official .pth)**,
**RWKV-7 whole-model C kernel (5.39× vs numpy, 7.49× at 8 threads)**,
**INT4 weight quantization (2.83× end-to-end, 2.7× size compression)**,
**Mamba-2 / DeltaNet / GLA official-weight integration (numerics aligned with the reference,
greedy generation coherent)**, **AVX-512 and NEON kernel sources (cross-compiled +
disassembly-verified)**, **OpenAI-compatible HTTP service (`sunnynet-serve`)**,
**Ollama / LM Studio integration (`sunnynet-integrate`)**,
**stable C API ABI (`sunnynet/runtime/sn_abi.h`, versioned + capability bits + error codes)**.

Near term: weight release (`sunnynet-release`).

Mid term: add the matrix-valued recurrence (outer product / rank-1 correction) kernel, moving
RWKV-7 and DeltaNet single-steps into C as well (Mamba-2 and GLA can reuse the existing
recurrence).

Not doing: replicate llama.cpp's general tensor operator layer (GGML is designed for
Transformers, per-operator dispatch); match Transformers / vLLM (a model repo and a GPU serving
track, respectively).
---

## Repository Structure

```
sunnynet/
  ops.py          original operators: poly_sigmoid / softwedge / CenteredScaleNorm /
                  multi-scale decay / log-domain parallel scan / general gated scan
  layers.py       SunnyBlock: state-update unit + circulant filter + local conv + nonlinear branch
  kernels/        AVX2 hand-written kernels (scan / fused core / inference-step) + auto-compile load
                  scan_avx512.c  AVX-512F/BW/VL kernel (masked tail + CPUID probe)
                  scan_neon.c    ARM NEON kernel (zig cross-compile / CI aarch64 execution)
                  simd.py        AVX-512 / NEON load and cross-compile layer
  linear_baselines.py  O(N) same-complexity comparisons: linear attention / RWKV / Mamba / Hyena / H3
  model.py        SunnyNet body (state (B,L,D), per-layer relay)
  init.py         spectral-anchored initialization
  cli.py          command-line inference (load ckpt -> tokenizer -> autoregressive decode,
                  --backend torch/c; --gguf for external RWKV-7 weights;
                  --hf-dir for Mamba-2 / DeltaNet / GLA)
  serve.py        OpenAI-compatible HTTP service (stdlib implementation, --host / --port)
  integrate.py    Ollama / LM Studio integration (Modelfile generation + probing + GGUF placement)
  release.py      weight-release export (safetensors + runtime blob + model card + sha256)
  runtime/        standalone C inference runtime
    recur.c         general first-order recurrence kernel (scan / backward / step / INT8 quant)
    sunnynet.c      whole-model single-step decode (B=1, no intermediate tensors)
    sn_abi.h/.c     stable ABI layer (versioned + capability bits + error codes + opaque handles)
    abi.py          ABI ctypes bindings
    model.py        ctypes bindings
    gguf.py         GGUF read-only loading (f16/bf16/f32 + quantized dequant)
    rwkv7.py        RWKV-7 single-step recurrence (numpy reference path, reads GGUF weights)
    rwkv7_c.py      RWKV-7 whole-model C kernel bindings
    mamba2.py       Mamba-2 single-step recurrence (decay outer product factorable, elementwise)
    deltanet.py     DeltaNet single-step recurrence (true rank-1 correction, matrix-valued)
    gla.py          GLA single-step recurrence (gate on key dim only, elementwise)
  baseline.py     parameter-matched Transformer comparison (comparison only, not the architecture)
  tokenizer.py    BPE tokenizer implemented from scratch
  profiling.py    real performance probes (peak memory / memory access / throughput)
train/
  train.py        unified trainer (both architectures share the same loop)
  evaluate.py     perplexity / long-range probe / generation samples
experiments/
  run_experiments.py   experiment orchestration (baseline / ablation / seeds / scale / probe)
  battery.sh           batch-run baselines + ablations + seeds + scaling
  battery2.sh          20M scaling comparison + long-sequence training comparison
  long_sequence.py     forward latency vs sequence length (O(N) verification) + long-seq training
  make_figures.py      generate all figures -> docs/figures/
  collect_results.py   aggregate all results into a Markdown table
tests/
  test_architecture.py  architecture self-check (chunk consistency / causality / gradient bound / depth stability, 63 items)
  test_runtime_model.py C whole-model decode consistency
  test_runtime_hf.py    Mamba-2 / DeltaNet / GLA external-weight integration
  test_rwkv7_gguf.py    GGUF loading / tensor alignment / RWKV-7 recurrence equivalence / token round-trip
  test_abi.py           C API stable ABI contract (version / capability bits / error codes / step equivalence)
  test_simd.py          AVX-512 build artifact + NEON cross-compile and disassembly
  test_integrate.py     Ollama / LM Studio integration (Modelfile / probing / directory layout)
  test_release.py       weight-release export (safetensors round-trip / blob header / sha256)
  test_smoke_pipeline.py end-to-end smoke (synthetic corpus -> training -> evaluation)
prepare_data.py   real corpus collection
build_corpus.py   tokenizer training + token-stream construction
docs/
  ARCHITECTURE.md  full mathematical derivation
  EXPERIMENT_REPORT.md  measured experiment report (including negative results)
  ABI.md           C API stable ABI versioning / capability bits / error codes
  figures/         experiment figures
```
---

## Originality Statement

SunnyNet is an independent, original architecture, with two points that need stating:

- Its state recurrence is mathematically related to linear-attention work (both are first-order
  linear recurrences); this is not glossed over;
- The gating construction (convex combination + log-uniform multi-scale + elementwise protection
  gate), cross-layer propagation (state relay + feature residual stream), activations,
  normalization, and initialization are all designed and empirically calibrated in-house;
- No code or structure was borrowed or ported from any existing project.

Four design conclusions were overturned and corrected by measurement during the architecture
design: softmax channel competition makes the gradient vanish with depth; the sign of the
spectral-radius layer-decay was set wrong; CenteredScaleNorm's `s_min` above the actual
activation scale degenerates normalization into a constant scaling; the original design's
exclusion of residuals actually caused the single largest quality loss. The corrections and the
data are recorded in sections 3/4/5 of docs/ARCHITECTURE.md. These records are kept to document
where each design constraint came from.

---

## Risks and Limitations

1. **Short-sequence disadvantage**: the O(N) advantage only materializes on long sequences. The
   batch=1 forward crossover is at T≈384 (6 layers) / T≈600 (4 layers); below that it is slower
   than a standard Transformer. This architecture is not suited to short-sequence scenarios.
2. **Quality is still below H3 on short sequences**: at T=256, 600 steps, same-protocol
   comparison, ppl 57.83 vs H3 53.93 (~7% behind).
   The comparison is unconverged, single seed, and learning rates were not tuned per
   architecture; it can only be read as an early sample-efficiency ranking.
3. **No absolute long-sequence advantage over peers either**: at T=4096 Hyena is still 1.9x
   faster and H3 is nearly tied; there is no length interval where it leads every peer. In the
   training regime at T=1024 it is 1st in speed and 2nd-lowest in memory — forward and
   training-regime rankings differ, so citations must carry their conditions.
4. **Early timing data was distorted**: without process isolation, same-process accumulated
   builds inflate readings by more than 2x, and the inflation factor varies by architecture.
5. **Long-range dependencies**: the rolling state's information capacity is bounded by the fixed
   dimension, with a theoretical upper bound below full attention. Large-span channels only
   guarantee "not forgotten", not "retrievable".
6. **Pseudo-convergence**: a falling loss / ppl does not imply reasoning or instruction-following
   ability.
7. **Corpus coverage**: the Chinese Wikipedia is unreachable from this machine's network, so the
   "encyclopedia" domain is missing; experiments cover only the three real domains: ancient
   texts / English literature / code.
8. **Scale**: 1.4M / 20M / 100M are measured. 0.5B can only run forward and a single SGD step
   (the 6GB GPU boundary state); there is no multi-step loss-decay trend and no 0.5B quality
   data at all.
9. **Ablations did not reach a conclusion**: 600 steps are unconverged, and disabling
   components lowered ppl instead — not evidence of component effectiveness.
10. **The same-complexity comparison is unconverged**: the seven-architecture comparison ran
    only 600 steps, and learning rates were not tuned per architecture, so small inter-arch
    gaps fall inside seed noise.
11. **`torch.compile` is unavailable**: this machine's MSVC lacks the Windows SDK headers, and
    Inductor's Windows branch hard-codes MSVC-style command lines; the GPU side also has no
    nvcc / CUDA toolkit / triton. The hand-written kernels bypass Inductor and are wired up
    independently.
12. **The boundary of the GPU conclusions**: the speedups are lower bounds with no CUDA-side
    optimization done; with no peer GPU comparison, they are not evidence of "competitive GPU
    performance".

See **[docs/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/EXPERIMENT_REPORT.md)** and **[safety_notice.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/safety_notice.md)** for details.
The model is an experimental research project; traditional-Chinese-medicine-related output must not be used as a basis for medical diagnosis.

---

## Sponsors

<a href="https://afdian.com/a/sunny-cx">
  <img src="https://img.shields.io/badge/afdian-sponsor-946CE6?style=for-the-badge" alt="afdian">
</a>

---

## License

[Apache License 2.0](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/LICENSE)

Copyright 2026 sunny-cx

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

---

## 快速开始

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

手写 AVX2 内核需 C 编译器（w64devkit / GCC）：
将 `SUNNY_W64` 指向 w64devkit 根目录，或把 `gcc` 加入 PATH。

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

完整数学推导见 **[docs/ARCHITECTURE.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/ARCHITECTURE.md)**。

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
实例化对齐，层数偏离基准不超过 ±1。完整数据见 [docs/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/EXPERIMENT_REPORT.md)。

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
- 与 O(N) 同行同场对照见下节：SunnyNet 解码**不是最快**，rwkv 全程第 1、快 1.9–2.6×。

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
| `.safetensors`（139 张量，绑定 head 已去重） | 421.5 MB | `12784fbfadf7` |
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
所有数字来自本机实测、逐格独立子进程。完整数据见 [docs/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/EXPERIMENT_REPORT.md) 第 23 节。

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
| **解码 ms/step（B=1）** | 第 5 → 第 2 | **rwkv** | 慢 1.9–2.6× |
| 长距离检索（代码域） | 与 TF 持平 | 并列 | — |
| 长距离检索（自然语言） | 接近 0 | — | 无优势 |
| 跨分布泛化 | 中文新域第 1 / 英文第 2 | sunnynet | — |

### 解码态七架构对照

口径：每个 (架构, T) 独立子进程；先逐 token 走满 T 步建立真实解码状态（不计时），
再测该长度下单步解码 ms；1 线程；2 轮取中位。

ms/step（越低越好）：

| 架构 | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| `rwkv` | **1.768** | **1.574** | **1.598** | **2.113** | **1.564** |
| `h3` | 2.517 | 2.524 | 2.487 | 2.969 | 3.227 |
| `transformer` | 2.550 | 2.667 | 3.426 | 5.049 | 6.219 |
| `mamba` | 2.686 | 2.742 | 2.573 | 2.773 | 3.205 |
| **`sunnynet`** | **3.279** | **4.067** | **3.086** | **3.331** | **2.854** |
| `hyena` | 3.346 | 3.583 | 3.662 | 5.101 | 4.530 |
| `linear_attn` | 4.061 | 3.452 | 4.599 | 3.676 | 3.624 |

读数：

- Transformer 的 O(t) 被复现（2.550 → 6.219 ms），T=4096 时 SunnyNet 快 2.18×。
- SunnyNet 单步成本大致平坦，排名随 T 改善（5→7→4→4→2），但**从未追上 rwkv**。
- **解码态最优是 rwkv**（1.57–2.11 ms），单步只有一条 `p*s + k*v` 逐元素链；
  SunnyNet 单步含竞争门控（3 次 sigmoid）、门控概率、扫描、反馈支路，算子数是它的数倍。
- 五个 O(N) 同行此前都无单步解码接口，本仓库为其实现 `init_state`/`step`，
  与整段前向数值一致性 maxdiff ≤ 4e-05。

### 差距的共同根因

落后的维度（解码速度、前向速度、L=6 训练速度、100M 质量、自然语言长距离检索）
收敛到同一个根因：**逐元素算子数量**。证据：训练步 4524 次 aten 调用；
减层加宽（等栈参数）训练快 12–17% 且 ppl −10.9%；层数对齐到 L=6 后同预算窄配置反慢 30–40%；
解码态 rwkv 靠单条逐元素链取胜。⇒ 有效方向只有减算子数，不是加宽、调 batch 或换矩阵库。

---

## 线性架构的 CPU 推理 Runtime

`sunnynet/runtime/` 是独立于 PyTorch 的 C 推理内核（AVX2/FMA + OpenMP），只依赖 ctypes
与 C 编译器。它实现通用一阶递推

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
与单算子同量级。B=1 单序列也不是本 Runtime 的目标场景（解码态最优是 rwkv）。

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
采样（temperature / top-k / top-p）在服务侧统一处理。

实测（本地启动后）：官方 `openai` SDK 的 `models.list`、chat 非流式、chat 流式、
legacy completions 四种调用全部通过；400（缺字段）/ 404（未知路径）错误处理正常。

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

### AVX-512 与 NEON 内核（`sunnynet/kernels/scan_avx512.c`、`scan_neon.c`）

AVX2/FMA 之外补两条 SIMD 后端源码，均含前向扫描、反向扫描、单步三个入口。

AVX-512（F/BW/VL）：用 `__mmask16` 做掩码尾处理，导出 CPUID 探测
（leaf 7 EBX bit16）与 ISA 字符串。**本机 CPU 实测 AVX512F=0**（i7-8750H，Coffee Lake），
故只做编译与产物验证，不做执行验证：

| 验证项 | 结果 |
|---|---|
| 编译 | `gcc -O3 -mavx512f -mavx512bw -mavx512vl -shared` 通过 |
| 反汇编 | 198 处 `zmm` 寄存器、16 处 `vfmadd132ps` |
| 加载保护 | 不支持时 `load_avx512()` 直接抛错，避免非法指令 |

NEON：无 ARM 工具链、无 QEMU（`wsl.exe` 被安全策略拦），改用 `zig cc` 真交叉编译：

```bash
pip install ziglang
python -c "from sunnynet.kernels.simd import cross_compile_neon; cross_compile_neon('<out.so>')"
```

| 验证项 | 结果 |
|---|---|
| 产物 | AArch64 ELF64（`e_machine=183`，`zig cc -target aarch64-linux-gnu`） |
| 反汇编 | capstone 确认 4-lane `fmla v3.4s, v2.4s, v4.4s` 及 `fmul`/`fsub`/`fadd` |

**本机无法执行 NEON**，真实执行验证放在 CI 的 `ubuntu-24.04-arm` runner（原生 aarch64）。

### 稳定 C ABI（`sunnynet/runtime/sn_abi.h`、`sn_abi.c`）

此前的 C 接口是内部用的，改结构即破坏调用方。新增一层稳定 ABI 隔离：

- 符号统一 `sn_abi_` 前缀，句柄不透明（`SnAbiRwkv7*`），调用方不依赖内部布局
- 版本 `SN_ABI_VERSION = (MAJOR<<16)|(MINOR<<8)|PATCH`，当前 **1.0.0**
- 能力位 `SN_CAP_*`、错误码 `SN_OK=0` / `SN_E_INVAL=-1` / `SN_E_UNSUPPORTED=-2` 等
- `sn_status` 统一为 `int32_t`，函数只返回状态码，不做异常抛出

ctypes 绑定在 `sunnynet/runtime/abi.py`，与内部句柄路径**逐位一致**
（`max|diff| = 0.0`）。契约测试 9 项全绿（版本协商、能力位、错误码、
`check(1)=OK` / `check(9)=E_UNSUPPORTED`、RWKV-7 步进等价）。详见 [docs/ABI.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/ABI.md)。

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
**AVX-512 与 NEON 内核源码（交叉编译 + 反汇编验证）**、
**OpenAI 兼容 HTTP 服务（`sunnynet-serve`）**、
**Ollama / LM Studio 接入（`sunnynet-integrate`）**、
**C API 稳定 ABI（`sunnynet/runtime/sn_abi.h`，版本化 + 能力位 + 错误码）**。

近期：权重发布（`sunnynet-release`）。

中期：补矩阵值递推（外积 / rank-1 校正）内核，把 RWKV-7 与 DeltaNet 的单步也搬进 C
（Mamba-2 与 GLA 已可复用现有递推）。

不做的：复刻 llama.cpp 的通用张量算子层（GGML 为 Transformer 设计，逐算子调度）；
打平 Transformers / vLLM（分别为模型仓库与 GPU 服务赛道）。

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
docs/ARCHITECTURE.md 第 3/4/5 节。保留这些记录是为了说明各设计约束的来源。

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

详见 **[docs/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/EXPERIMENT_REPORT.md)** 与 **[safety_notice.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/safety_notice.md)**。
模型为实验科研项目，中医相关输出不可作为医疗诊断依据。

---

## 赞助

<a href="https://afdian.com/a/sunny-cx">
  <img src="https://img.shields.io/badge/爱发电-赞助支持-946CE6?style=for-the-badge" alt="爱发电">
</a>

---

## 许可证

[Apache License 2.0](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/LICENSE)

Copyright 2026 sunny-cx