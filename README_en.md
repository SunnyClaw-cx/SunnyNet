# SunnyNet

[![tests](https://github.com/SunnyClaw-cx/SunnyNet/actions/workflows/tests.yml/badge.svg)](https://github.com/SunnyClaw-cx/SunnyNet/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/sunnynet-cx)](https://pypi.org/project/sunnynet-cx/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**English** · [中文](README.md)

An original general-purpose sequence architecture, not derived from Transformer, RNN, Mamba, CNN, or any existing design.

Repository: <https://github.com/SunnyClaw-cx/SunnyNet>

Built for CPU training: O(N) sequence complexity with a fixed-size rolling state, no token-to-token interaction, and no materialized activations across the sequence.

Licensed under Apache License 2.0.



---

### Overview

SunnyNet is built around a single recurrent primitive: a convex combination.

```
s_t = p_t ⊙ s_{t-1} + (1 - p_t) ⊙ u_t ,   p_t ∈ [1-δ_i, 1]
```

Both coefficients are non-negative and sum to one, so `|s_t| ≤ max(|s_{t-1}|, |u_t|)`. The state is bounded by construction and cannot diverge. Each channel's retention floor is tied to a memory span `H_i` laid out on a log scale over `[2, 8192]`, so one layer carries time scales from two tokens to eight thousand.

The design targets the real CPU bottleneck — memory bandwidth and cache behavior — not FLOPs. It avoids dense matmul, keeps the state dimension fixed, and passes only the state across chunks. Training a 100M-class model from scratch on CPU, instruction fine-tuning, and domain fine-tuning all work.

Full mathematical derivation: [docs/en/ARCHITECTURE.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/en/ARCHITECTURE.md).

### Core design constraints

| Constraint | Design response |
|---|---|
| Strictly O(N) sequence complexity | Rolling state + first-order linear recurrence, no token-to-token interaction |
| No activation materialized over the whole sequence | Fixed state dimension, only the state crosses chunk boundaries, backward is recomputed on demand |
| Avoiding large dense matmul | Low-rank factorization + depthwise separable conv + tridiagonal recurrent filter |

### Original components

| Component | Scheme | Key property |
|---|---|---|
| State update | Multi-timescale convex-combination rolling | State bounded, no divergence; multi-scale output gating |
| Nonlinearity | SoftWedge `f(z)=z+a·s(z)`, base `s(z)=½+½·z/(1+\|z\|)` | Derivative `f' ∈ (1, 1+a/2]`, nowhere zero, no gradient cliff |
| Normalization | CenteredScaleNorm | Denominator has a smooth floor `s_min`, near-constant vectors are not amplified (precondition for deep stacking) |
| Initialization | Spectral anchoring (QR orthogonal slices, σ_max anchored directly) | Condition number 1, flat forward-gain spectrum; ρ=1.0 calibrated by measurement |
| Cross-layer transfer | State relay + feature residual stream | State scale does not grow with depth; residual direct path carries the depth gradient (measured −28% ppl) |
| Channel competition | Intra-group pairwise competitive gating | Same semantics as softmax, but Jacobian is O(1) instead of O(1/d) |
| Parallel scan | Log-domain closed form | Algebraically equivalent to step-by-step recurrence; the block-head state can be recovered exactly from the block tail |

No reuse of ReLU / GELU / SiLU / Mish, LayerNorm / RMSNorm / InstanceNorm / GroupNorm, Xavier / Glorot / He / MSRA initialization, LSTM / GRU gating formulas, or SSM / HiPPO discretization.

### Measured results

All numbers are measured on one machine (i7-8750H, AVX2, PyTorch 2.14.1 CPU) with per-cell process isolation. Full data, including negative results, is in [docs/en/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/en/EXPERIMENT_REPORT.md).

| Metric | SunnyNet | Best peer | Setting |
|---|---|---|---|
| Training step, ms/token | 0.0953 | RWKV 0.0982 | T=256, batch 8, tied over 9 paired rounds |
| Quality, ppl | 57.83 (2nd of 7) | H3 53.93 | 600 steps, single seed, not converged |
| Decode, ms/step at T=4096 (torch path) | 2.89 | RWKV 1.67 | B=1, single sequence |
| Decode, ms/step (whole-model C runtime) | **1.65 at T=4096** | — | B=1, first of 8 at every length |
| Forward, ms/token at T=4096 | 0.0352 | Hyena 0.0184 | batch=1, pure forward |

**Training speed**

- Training step is statistically tied with RWKV (3–4 of 9 rounds faster, median paired ratio 1.042) and 16–41% faster than H3, Mamba, Hyena, and linear attention. The tie holds only at L=4; at L=6 the same parameter budget is 30–40% slower.
- Scaling with depth is the binding constraint on CPU. The step time is linear in the number of layers (`step_ms ≈ 32.9·L + 102.4` at d=128, T=256, B=8), and each layer adds a fixed cost of about 1100 aten calls of which `aten::mm` is only 1.9% — a per-layer operator-dispatch tax independent of width. At a fixed parameter budget, a deep-narrow model is worse on both axes than a shallow-wide one (L3/L4 beat L6/L8 in step time and in ppl). The fused AVX2 kernel cuts the per-layer call count from 1099 to 715; further fusion has little headroom because the tax is spread over five block stages. The actionable fix is configuration, not more kernels: keep depth in the 4–6 range and put the parameter budget into `inner`/`rank`. Preset `p100msw` (d1536/L6) is 4.2% faster than `p100mw` (d1408/L8) at matched stack parameters.
- Within the same architecture: operator fusion 19–41%, layer reduction with widening 12–17%, narrow residual stream with `p.clamp` dropped 17.7–20.7%.
- Hand-written fused kernel (competitive gating + gate probability + scan in one pass): 1.10–1.17× on the training step, loss bit-identical to torch — it changes speed, not quality. The inference-mode kernel shows no significant gain (B=1 tensors are too small; FFI cost is the same order as a single operator), so it is off by default.

**Inference**

- Single-step decode cost versus context: SunnyNet is flat (O(1), 2.87 → 2.90 ms, drift ±5%), a matched Transformer rises monotonically (O(t), 2.49 → 6.85 ms). Crossover T≈512, and SunnyNet is 2.36× faster at T=4096. This needs no kernel.
- Decode via the shipped whole-model C runtime is faster still: first of eight architectures at every length from T=256 to T=4096, 1.02–1.15× faster than RWKV and 4.2× faster than a parameter-matched Transformer at T=4096.
- Boundary: single sequence, B=1, randomly initialized models, 2-round median; batch decode is not measured and neither is KV-cache quantization or paging.

**Quality**

- After the architecture corrections (residual bypass + CSN floor), T=256 ppl fell from 93.82 to 57.83, 2nd of seven architectures (behind H3 53.93 only); the rest are Mamba 63.34 / RWKV 64.10 / Hyena 82.50 / Transformer 90.05 / linear attention 91.91.
- At 20M, layer reduction with widening cuts ppl another 10.9% (54.82) and moves SunnyNet to 1st of five in the same-protocol comparison.
- The scaling trend is not monotonic: 2nd at small, 1st at 20M, back to 2nd at 100M (109.47, 3.6% behind RWKV 105.66). "Larger scale, larger advantage" is not supported.

**Long sequences and memory**

- Forward: crossover with a matched Transformer at T≈384 (6-layer) or T≈600 (4-layer). At T=4096 it is 22.7× faster per token (batch=1, pure forward, 8 threads, 1.4M model).
- Training mode at T=1024: SunnyNet is 1st in speed (0.1319 ms/tok) and 2nd-lowest in peak memory (1812 MB, versus 3192 MB for the Transformer).
- Against O(N) peers in forward there is no stable leading interval — SunnyNet climbs from 6th to 3rd as T grows and never tops the table; the training-mode ranking differs (1st at T=1024).

**Cross-distribution**

On independently collected external data (trained on the original three domains): Chinese tech news ppl 321.08, 1st of 7 (23% unseen characters in that domain); English arXiv abstracts 35.58, 2nd of 7. OOD gap 1.480, the lowest tier (tied with Hyena). Absolute ppl is still high and this is 600 steps, single seed — the only defensible claim is that SunnyNet is not worse than same-complexity peers across distributions.

**Short sequences**

The T≈384 crossover holds only for batch=1 pure forward. In training mode (batch=8) the crossover moves below T<256: at T=256/384 SunnyNet is 1.5–2.4× faster than the Transformer. At T=128 it is still 10.4% slower (old d128) or 4.5% slower (narrow `smalln80`) — the narrow residual stream is a verified mitigation.

**GPU**

The architecture runs on CUDA with no CUDA-specific rewrite. At B=8: forward 8.0–15.2×, training step 9.9–12.5× (no CUDA-side optimization, so these are lower bounds). The local GPU (GTX 1060 6GB) is used only for correctness checks and speedup measurement; there is no peer GPU comparison.

### Six-architecture comparison

Compared against `h3`, `rwkv`, `mamba`, `hyena`, `linear_attn`, and `transformer`. All numbers are local measurements with per-cell process isolation. Full data: [docs/en/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/en/EXPERIMENT_REPORT.md) §23.

| Dimension | SunnyNet position | Best | Gap |
|---|---|---|---|
| Quality ppl (1.4M / 600 steps) | 2nd (57.83) | h3 53.93 | 7.2% behind |
| Quality ppl (20M / 200 steps) | **1st** (77.90) | sunnynet | — |
| Quality ppl (100M / 100 steps) | 2nd (109.47) | rwkv 105.66 | 3.6% behind |
| Training speed (L=4, matched stack) | tied with rwkv | rwkv (nominal) | paired test not significant |
| Training speed (L=6, layers aligned) | **behind** | rwkv | 30–40% slower |
| Training-mode memory (T=1024) | 2nd lowest | rwkv 1611MB | 12% higher, 43% below the Transformer |
| Forward ms/token (batch=1) | 6th → 3rd with T | hyena | never tops the table |
| Decode ms/step (B=1, torch path) | 7th of 8 | rwkv | 1.73–2.16× slower |
| **Decode ms/step (B=1, whole-model C runtime)** | **1st of 8** | sunnynet | 1.02–1.15× faster than rwkv |
| Long-range retrieval (code) | tied with the Transformer | — | — |
| Long-range retrieval (natural language) | near zero | — | no advantage |
| Cross-distribution | 1st on the Chinese new domain / 2nd on English | sunnynet | — |

The dimensions where SunnyNet lags or only ties — decode speed on the torch path, forward speed, L=6 training speed, 100M quality, natural-language long-range retrieval — converge on one root cause: **the number of elementwise operators**. Evidence: 4524 aten calls per training step; layer reduction with widening is 12–17% faster and 10.9% lower ppl at matched stack parameters; aligning layers to L=6 makes the same-budget narrow configuration 30–40% slower; RWKV's decode wins with a single elementwise chain. The productive direction is fewer operators, not wider layers, larger batches, or a different matmul library.

### Limitations

- Short sequences are a structural weakness: below T≈384 (batch=1 forward) a standard Transformer is faster. The cause is operator count, not FLOPs.
- No 0.5B quality number. On the 6 GB GPU available here, forward and a single SGD step run (5.0 GB peak); AdamW needs 7.6 GB and does not fit.
- The scaling trend is not monotonic: 2nd at 1.4M, 1st at 20M, 2nd at 100M.
- Long-range retrieval is verified only on structured text (code, far-bucket hit 0.13); on natural language it is near zero.
- No GPU kernel. The architecture runs on CUDA unmodified (8.0–15.2× forward on a GTX 1060), but there is no Triton or CUDA implementation.
- On the torch decode path, speed is not competitive against O(N) peers: RWKV decodes 1.73–2.16× faster at every length tested. The shipped whole-model C runtime reverses this ordering (1st at every length); the two paths must be quoted separately.
- Coherence ("incoherence") has two independent causes, both measured: a training-budget deficit (the external Kaggle run had 0.96 token/parameter against a Chinchilla suggestion of 20, a 20.8× gap) and a recall-capacity ceiling of the elementwise state (2 key-value pairs: recurrence 4.7% vs attention 93.4% accuracy). A same-budget controlled run shows the recurrence baseline at ppl 143.24 versus 140.40 for a Transformer with matched stack parameters — the gap is the budget, not the recurrence. Adding a semi-count associative-memory branch lowers zh ppl to 138.82 (−3.1%) and is the only configuration that improves generation coherence. Inter-layer dilution is a separate, depth-only effect: on a 100M checkpoint the input-specific energy share falls 0.226 → 0.035 from layer 0 to layer 7, and normalizing the state plus its feedback source raises 24-input argmax variety from 12 to 16 and lowers val ppl 200.51 → 198.33. Residual-stream normalization is falsified (ppl 263 → 625). The tail layer of that dilution is not yet resolved.

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

**RWKV-7 C kernel** (`sunnynet/runtime/rwkv7.c`): the whole decode step — embedding lookup, per-layer norm and time-mix (LoRA, `tanh`/`sigmoid`), per-head matrix state recurrence with the rank-1 correction, group-norm, channel-mix, final norm, LM head — runs in C in one call per step. Per-token logits agree with the numpy reference to 7.25e-05 (bf16 precision), argmax 12/12, and 120/120 over a 120-step run. Speed: numpy 366.36 → C single-thread 67.92 ms/step (5.39×, C wins 5/5 rounds), 8 threads 48.88 (7.49×).

Those multiples are **internal** (C against our own numpy), not an external comparison. So here is the same-weights comparison against llama.cpp's CPU backend (`experiments/_llamacpp_vs_sunny_st.py`, llama.cpp b11889):

| Threads | vs llama.cpp, same fp32 | vs llama.cpp, same 4-bit |
|---:|---|---|
| 1 | **0.90×** (1/7 rounds) | **0.89×** (0/7) |
| 6 | **0.58×** (0/5) | **0.71×** (1/5) |

**On the same weights and precision llama.cpp is faster** — about 1.1× single-thread, 1.4–1.7× at six threads. Single-thread both sit near the sequential read ceiling (12.5 / 13.7 GB/s); the gap is mainly multithread bandwidth utilization (12.5 vs 19.5 GB/s). Full boundaries in [docs/en/EXPERIMENT_REPORT.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/en/EXPERIMENT_REPORT.md) §24.11b.

**INT4 weight quantization** (`sunnynet/runtime/quant.c`): group-symmetric INT4, one fp32 scale per 64 consecutive weights, two values per byte. Quantized GEMV quantizes activations to INT8 on the same grouping and uses `maddubs` for the dot product. Blob size 729.1 → 269.0 MB, single-step decode 49.16 → 17.35 ms (2.83× end-to-end, 2.7× compression). The integer path is bit-identical to the explicit integer formula. Generation quality holds for English and degrades for Chinese and code — 4-bit needs calibration or mixed precision to preserve quality; only the "runs and is fast" result is claimed here. CLI: `--quant int4` with `--gguf-backend c`.

**Mamba-2**: the same runtime interface loads official Mamba-2 weights and runs per-token single-step recurrence without a PyTorch forward.

- Mamba-2 (`state-spaces/mamba2-130m`): each head state is a `(64, 128)` matrix, but the decay `dA = exp(dt·A)` is identical over the state dim and the outer product factorizes, so it reduces to an elementwise first-order recurrence. Against `Mamba2ForCausalLM` the whole forward agrees to 5.9e-04 (argmax 24/24); official fp32 itself deviates 3.8e-04 from fp64 while this implementation deviates 2.5e-04, so the difference is single-precision accumulation, not a formula error. Segmented continuation is bit-identical to the full run.

The whole forward pass runs in C via `--hf-arch mamba2 --blob`: per-token logits 1.678e-04, argmax 36/36. Speedup 1.37×, the only one not yet at the bandwidth ceiling — the gain tracks weight bytes over available bandwidth, and the 0.52 GB weight is small enough to leave headroom (a 1.37 GB weight is already at the single-thread sequential-read ceiling of 18.63 GB/s). Matrix-valued rank-1 recurrence (`S ← S + k ⊗ (·)`) is used by RWKV-7 and covered by a dedicated kernel (`sn_mat_scan`); Mamba-2 factorizes and reuses the elementwise recurrence.

**fp16 decode path**: truncating the five GEMV matrices and the LM head to fp16 at load time (small parameters stay fp32) drops bytes read per step from 467.6 to ~222.3 MB. Median 24.67 → 18.60 ms/token (1.33×, fp16 wins 11/11 rounds), logits max|diff| 4.181e-03 (relative 7.42e-04), argmax 64/64. The kernel reads 16 half values (32 B) at a time before converting, preserving load width. Use `--backend c --half`; fp32 remains the default and is bit-identical. fp16 is no longer bandwidth-bound (12.0 GB/s), so the bottleneck moves to type conversion.

**Decode parallelism scales with weight size**: small models keep weights in cache, so extra threads only add synchronization. Blobs ≤23 MB are fastest single-threaded, blobs ≥39 MB are fastest on physical cores (136 MB gains 1.29×). The CLI selects this from the blob size; `--threads` overrides. On the 100M-class blob (467.6 MB), row-major weight layout plus output-channel-parallel GEMV lifts the effective bandwidth from 6.10 to 18.18 GB/s and the single step from 76.60 to 25.72 ms (2.98×, C wins 5/5 rounds; logits max|diff| 2.766e-05, argmax 32/32). Parallelism takes physical cores, not logical ones: 6 cores 26.69 ms versus 12 logical cores 36.05 ms.

**AVX-512 and NEON**: source for both backends is included, each with forward scan, reverse scan and single step. AVX-512 is split as `scan_avx512.c` plus `core_avx512.c` / `step_avx512.c`, where one 512-bit register holds two groups of eight lanes and the in-group neighbour shift is a 16-lane `vpermps`; masked accesses handle a tail when `D` is not a multiple of 16. This CPU reports AVX512F=0 (i7-8750H, Coffee Lake), so only the build is verified — disassembling `core_avx512.c` alone gives 192 `zmm` occurrences and 10 `vfmadd*ps`; linking it with `step_avx512.c` gives 315 `zmm`, 44 `vfmadd*ps`, 16 `vpermps` and 33 masked moves — and the loader raises rather than executing an illegal instruction. Because neither an AVX-512 CPU nor an ARM device exists here, the indexing and in-group semantics of both sources are executed on x86 through same-named scalar shims (`experiments/_avx512_shim/`, `experiments/_neon_shim/`); they agree with the torch reference to 1.19e-07 and the NEON scan is bit-identical. That rules out mis-indexed channels that stay in bounds; it does not prove floating-point behaviour or throughput on real hardware. NEON is cross-compiled with `zig cc` to an AArch64 ELF64 and an ARMv7 ELF32; ARMv7 does not define `__ARM_FEATURE_FMA`, so the kernel falls back to `vaddq_f32 + vmulq_f32` (`vmla.f32` instead of `vfma.f32`).

**Stable C ABI** (`sunnynet/runtime/sn_abi.h`): symbols carry the `sn_abi_` prefix, handles are opaque, version is `(MAJOR<<16)|(MINOR<<8)|PATCH` at 1.0.0, with `SN_CAP_*` capability bits and `SN_OK` / `SN_E_INVAL` / `SN_E_UNSUPPORTED` codes. The ctypes binding in `sunnynet/runtime/abi.py` is bit-identical to the internal handle path (`max|diff| = 0.0`), and 9 contract tests pass.

**OpenAI-compatible HTTP server** (`sunnynet/serve.py`): zero-dependency (standard-library `ThreadingHTTPServer`), exposing `GET /v1/models`, `POST /v1/chat/completions` (with `stream: true`), `POST /v1/completions`, and `GET /health`. Backends are selected through an `Engine` abstraction (`HFEngine` / `GGUFEngine` / `GGUFCEngine` / `TorchEngine`); sampling is handled in the server. All four call styles of the official `openai` SDK were verified, and 400/404 handling is correct.

**Subcommands and the Ollama API server**: `cli.py` adds `chat` / `serve` / `convert` / `quantize` / `info`, with `-m/--model` dispatching on suffix (`.gguf` / `.pt` / HF directory). `chat` and `serve` default to the C kernel and export the blob on first use.

```bash
sunnynet chat   -m model.gguf --system "..." --ctx 8      # interactive, streaming
echo "Once upon a time" | sunnynet chat -m model.gguf     # piped single turn
sunnynet serve  -m model.gguf --port 8080
sunnynet convert <src.gguf> <dst.gguf> --arch rwkv7        # add sunnynet.* marks
sunnynet quantize <src.gguf> <dst.gguf> --type q4_0 --ppl  # quantize + perplexity
sunnynet info   -m model.gguf                              # arch / layers / vocab / tensors
```

`sunnynet/ollama.py` serves Ollama's **native** API (not the OpenAI shape) over
`GET /api/version`, `GET /api/tags`, `POST /api/generate`, `POST /api/chat`, with NDJSON streaming,
so an existing Ollama-API client only needs `OLLAMA_HOST` pointed here.

Measured locally via both the official `openai` SDK 3.36.0 and raw HTTP: all four OpenAI calls and
all five Ollama endpoints return 200, and an invalid `temperature` returns 400 instead of dropping the
connection. GGUF to C kernel single step: numpy fp32 530.58 / C fp32 86.55 / C INT4 28.21 ms/token
(191M test artifact, B=1). Items pending on real hardware are listed in [docs/en/ROADMAP.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/en/ROADMAP.md).

**Docker image** (`Dockerfile`, `python:3.11-slim`): the build stage installs gcc and compiles the kernels into a Linux `.so` (the PyPI wheel carries prebuilt libraries for win-x86_64 only, so Linux always builds from source); the runtime stage keeps `libgomp1` and no compiler. Weights are mounted at run time:

```bash
docker build -t sunnynet .
docker run --rm -p 8080:8080 -v "$PWD/sunnynet-small-zh:/models" sunnynet \
    --ckpt /models/sunnynet-small-zh.pt --tokenizer /models/tokenizer.json \
    --host 0.0.0.0 --port 8080
```

On a tag push, `.github/workflows/docker.yml` builds and pushes `ghcr.io/sunnyclaw-cx/sunnynet` (linux/amd64 only; the kernels target x86-64-v3).

**Ollama / LM Studio** (`sunnynet/integrate.py`): both run llama.cpp and need GGUF with a supported architecture. RWKV-7 GGUF is recognized directly. Three pitfalls are recorded: an explicit `TEMPLATE` is required (Ollama's built-in `rwkv-world` template wraps another `User:`/`Assistant:` layer and the output degenerates into repetition); `Error: could not locate ollama app` actually means the server is not running; and `OLLAMA_MODELS` must match the server's. Local probes bypass the system HTTP proxy.

CLI:

```bash
python -m sunnynet.cli --ckpt runs/small.pt --prompt "..." --max-new 80 --stats
python -m sunnynet.cli --ckpt runs/small.pt --backend c --half --prompt "..."   # pure-C decode, fp16
python -m sunnynet.cli --gguf model.gguf --gguf-backend c --quant int4 --prompt "..."
python -m sunnynet.cli --hf-dir <model dir> --hf-arch mamba2 --prompt "..."
sunnynet-serve --ckpt runs/small.pt --port 8080
sunnynet-integrate ollama model.gguf --name sunnynet-rwkv7 --run --smoke
sunnynet-release runs/small.pt --out-dir dist/release
```

The loading entry point is also callable from Python:

```python
from sunnynet.runtime import load_model
m = load_model("mamba2", "<model dir>")     # or "rwkv7"
st = m.reset(m.init_state())
logits = m.step(token, st)                  # O(1) per step
```

**Roadmap.** Existing assets: the original architecture, the AVX2 fused kernel (training step 1.10–1.17×), the generic recurrent C runtime (9.9× median against a vectorized baseline), whole-model C single-step decode (B=1 single thread 1.53–2.05×), cross-architecture mapping verification, GGUF read plus RWKV-7 weights (numerically equivalent to the official `.pth`), the RWKV-7 whole-model C kernel (5.39× over numpy, 7.49× at 8 threads), INT4 quantization (2.83× end-to-end), official Mamba-2 weights (numerically aligned, coherent greedy generation), AVX-512 and NEON kernel source (compiled, plus shim-executed equivalence), one-command GGUF into the C kernel (`sunnynet chat -m x.gguf`, blob exported and cached automatically), the `chat` / `serve` / `convert` / `quantize` / `info` subcommands, GGUF quantization with perplexity, the OpenAI-compatible server, a native Ollama API server, Ollama / LM Studio integration, and the stable C ABI. Near term: weight release via `sunnynet-release`. Mid term: a matrix-valued rank-1 recurrence kernel to bring RWKV-7 single-step into C (Mamba-2 can reuse the existing recurrence). Not planned: replicating llama.cpp's general tensor-operator layer, or matching Transformers / vLLM head-on. Verification protocols, items pending real hardware, and what was left out are in [docs/en/ROADMAP.md](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/docs/en/ROADMAP.md).

### Originality

SunnyNet is an independent original architecture, with two points stated plainly:

- Its state recurrence is mathematically related to linear-attention work (both are first-order linear recurrences); this is not avoided.
- The gating construction (convex combination + log-uniform multi-scale + per-element guard gate), the inter-layer transfer (state relay + feature residual stream), the activation, the normalization, and the initialization are all designed and calibrated here.
- No code or structure from any existing project was referenced or ported.

Four design conclusions were overturned by measurement and corrected: softmax channel competition drove the gradient to zero with depth; the sign of the spectral-radius depth decay was inverted; CenteredScaleNorm's `s_min` was above the actual activation scale, degrading normalization to a constant scale; and excluding the residual caused the largest single quality loss. The corrections and their data are in `docs/en/ARCHITECTURE.md` §3/4/5, kept to show where each design constraint came from.

### Quick start

Run the released small model without training anything:

```bash
pip install sunnynet-cx                                     # pulls torch + numpy
hf download SunnyClaw-cx/SunnyNet --local-dir sunnynet-small-zh   # ~21 MB
sunnynet --ckpt sunnynet-small-zh/sunnynet-small-zh.pt \
    --tokenizer sunnynet-small-zh/tokenizer.json --prompt "子曰：" --max-new 60
```

`hf` ships with `huggingface_hub`. The checkpoint is a 1.4M-parameter model (dim 128, 4 layers, vocab 8192) trained 16000 steps on the bundled Chinese corpus (val ppl 139.4); it emits classical Chinese. Add `--backend c --blob sunnynet-small-zh/sunnynet-small-zh.runtime.bin` to decode through the C runtime instead of torch.

### Linux servers: use the Docker image

The prebuilt wheel on PyPI only carries win-x86_64 `.so` files; on Linux the sdist needs a local gcc. The simplest path is the image (the build stage compiles the kernels with gcc, the runtime stage has no compiler):

```bash
docker run --rm -p 8080:8080 -v "$PWD/models:/models" \
    ghcr.io/sunnyclaw-cx/sunnynet \
    --ckpt /models/xxx.pt --tokenizer /models/tokenizer.json --host 0.0.0.0
```

Or build locally: `docker build -t sunnynet .` (Docker required). linux/amd64 only; the kernels target x86-64-v3. Separately, `manylinux` wheels are built by `.github/workflows/wheels.yml` via cibuildwheel, so installing those needs no compiler at all.

To build the environment from scratch and train your own:

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1 (CPU build is enough, no GPU needed), numpy.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pytest

# published as sunnynet-cx, imported as sunnynet
pip install sunnynet-cx          # or: pip install -e .  from source
```

The wheel from PyPI ships the prebuilt native libraries (active on x86-64 with AVX2+FMA), so the C kernels work with no local C compiler. Other platforms or instruction sets fall back to a source build, which is the only case that needs a compiler.

```bash
python prepare_data.py                              # collect real corpora
python build_corpus.py --vocab-size 8192            # train BPE, build token streams
python train/train.py --preset small --steps 800 --batch-size 8 --seq-len 256
python train/train.py --preset proto20m --steps 5000 --batch-size 4 --seq-len 512
python train/evaluate.py --ckpt runs/<yours>.pt --gen \
    --out experiments/results/eval_sunnynet.json
```

The validation split is file-level, so adjacent fragments of the same text never straddle train and validation.

### Repository layout

```
sunnynet/
  ops.py           original operators: poly_sigmoid / softwedge / CenteredScaleNorm /
                   multi-scale decay / log-domain parallel scan / generic gated scan
  layers.py        SunnyBlock: state update unit + recurrent filter + local conv + nonlinear branch
  kernels/         hand-written AVX2 kernels (scan / fused core / inference-mode step) + auto-build loader
                   scan_avx512.c  AVX-512F/BW/VL kernels (masked tails + CPUID detection)
                   scan_neon.c    ARM NEON kernels (zig cross-compile, executed on CI aarch64)
                   simd.py        AVX-512 / NEON loading and cross-compile layer
  linear_baselines.py  O(N) same-complexity peers: linear attention / RWKV / Mamba / Hyena / H3
  model.py         SunnyNet body (state (B,L,D), relayed layer by layer)
  init.py          spectral-anchored initialization
  cli.py           command-line inference (ckpt -> tokenizer -> autoregressive decode; --backend torch/c;
                   --gguf for external RWKV-7 weights; --hf-dir for Mamba-2)
  serve.py         OpenAI-compatible HTTP server (standard library, --host / --port)
  integrate.py     Ollama / LM Studio integration (Modelfile generation + probe + GGUF placement)
  release.py       weight release export (safetensors + runtime blob + model card + sha256)
  runtime/         standalone C inference runtime
    recur.c         generic first-order recurrence kernel (scan / reverse / single step / INT8 quantization)
    sunnynet.c      whole-model single-step decode (B=1, no intermediate tensors)
    rwkv7.c         RWKV-7 whole-model single-step kernel
    quant.c         group-symmetric INT4 weight GEMV
    sn_abi.h/.c     stable ABI layer (versioned + capability bits + error codes + opaque handles)
    abi.py          ABI ctypes binding
    model.py        ctypes binding
    gguf.py         GGUF read-only loader (f16/bf16/f32 + quantized dequantization)
    rwkv7.py        RWKV-7 single-step recurrence (numpy reference path, reads GGUF weights)
    mamba2.py       Mamba-2 single-step recurrence
    mamba2_c.py     Mamba-2 whole-model C kernel binding
  baseline.py      same-parameter Transformer baseline (for comparison only, not part of the architecture)
  tokenizer.py     BPE tokenizer written from scratch
  profiling.py     real performance probes (peak memory / memory traffic / throughput)
train/
  train.py         unified trainer (both architectures share one loop)
  evaluate.py      perplexity / long-distance probe / generation samples
experiments/       experiment scripts and result JSON
tests/             architecture invariants, C-runtime equivalence, external-weight integration, ABI, SIMD
docs/
  ARCHITECTURE.md        full mathematical derivation
  EXPERIMENT_REPORT.md   measured experiment report (negative results included)
  ABI.md                 version policy / capability bits / error codes of the stable C ABI
  ROADMAP.md             roadmap boundaries, per-item verification record, pending items
  figures/               experiment figures
```

### Sponsor

<a href="https://afdian.com/a/sunny-cx">
  <img src="https://img.shields.io/badge/Afdian-sponsor-946CE6?style=for-the-badge" alt="Afdian">
</a>

### License

[Apache License 2.0](https://github.com/SunnyClaw-cx/SunnyNet/blob/main/LICENSE)

Copyright 2026 sunny-cx
