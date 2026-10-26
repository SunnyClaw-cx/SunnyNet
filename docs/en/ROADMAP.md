*[中文版](../cn/ROADMAP.md)*
# Roadmap and verification record

This file records the six boundaries of the CPU inference roadmap, the state of each deliverable,
and draws a hard line between **measured on this machine** and **waiting for the target hardware**.
Numbers that have no verification environment here are marked as pending and are not claimed publicly.

## TL;DR

| Item | State | Evidence |
|---|---|---|
| CPU only (AVX2/AVX-512/NEON), no CUDA | Held | no CUDA source, no `torch.cuda` |
| Linear architectures only, no standard Transformer | Held | no `MultiheadAttention` |
| Foreign models first, own model later | Held | GGUF / external-weight path precedes weight release |
| No GUI / training framework / multimodal | Held | no tkinter/PyQt/gradio/streamlit dependency; a static-text Space is a demo, not a GUI |
| GGUF layer, CLI, OpenAI server, quantization, Ollama wrapper | Verified locally | §2, JSON in `experiments/roadmap_verify/` |
| AVX-512 / NEON kernels | Code + semantic equivalence verified, **speed pending** | §3 |
| Raspberry Pi / Apple Silicon / RK3588 | **No verification environment** | §4 |

## 1. The boundaries

1. CPU only. AVX2 + FMA is the verified baseline; AVX-512 and NEON ship as source plus a loader.
   No CUDA backend — general GPU inference (vLLM / TensorRT-LLM) belongs to other projects.
2. Linear complexity only, and only two: RWKV-7 and Mamba-2. Standard Transformers are out of scope.
   Two architectures that are fastest on CPU beat six that are supported but not fastest — the latter
   is a toy collection, the former is a standard. GLA / DeltaNet / HGRN2 were integrated and left a
   full record of measurements (`EXPERIMENT_REPORT.md`); all three carry weights ≥1.37 GB whose C
   speedup is already at the bandwidth ceiling (1.01× / 1.10× / 1.29×), so they leave the support list.
3. Order: make loading and running foreign weights fast first. Every speed number below comes from
   foreign weights (RWKV-7 191M, Mamba-2 130M).
4. No GUI, no training framework, no multimodal, no chat UI or Oobabooga plugin.
   `serve.py` / `ollama.py` expose HTTP only.
   Relationship to an HF Space: a Space is a **demo entry point**, not a GUI product — static
   HTML/JS only (copyable commands, recorded output), no gradio/streamlit, no interactive chat UI,
   so it does not violate this rule. The self-check grep below guards that line.
5. No bespoke weight format: GGUF (or safetensors + HF standard) is the authoritative format long
   term; the internal `.runtime.bin` is a local cache only.
6. Keep the hot path out of Python/PyTorch: the moat is a C kernel with no PyTorch.

Self-check (reproducible):

```bash
grep -rn "MultiheadAttention\|torch.cuda\|import tkinter\|PyQt\|gradio\|streamlit" sunnynet/
```

## 2. Verified on this machine

Verification scripts land their output in `experiments/roadmap_verify/`; the numbers below are read
from that output, not typed by hand.

### 2.1 GGUF in one command, running on the C kernel

```
sunnynet chat -m experiments/gguf/rwkv7-191M-world-bf16.gguf --prompt "Once upon a time"
```

`ensure_rwkv7_blob()` caches the blob by mtime: the first call exports it, later calls load it directly.

Test artifact: `rwkv7-191M-world-bf16.gguf`, 368.47 MiB, 342 tensors, 12 layers, d_model 768, vocab 65536.

| Backend | ms/token | Protocol |
|---|---:|---|
| numpy fp32 | 530.58 | 8 tokens, warm 2 |
| C fp32 | 86.55 | 32 tokens, warm 8 |
| C INT4 | 28.21 | 64 tokens, warm 16 |

INT4 / fp32 = 3.07×; INT4 / numpy = 18.81×. Engine build including blob export: 3.3–3.5 s.

INT4 is the B=1 single-sequence path on this 12-logical-core box. This matches the earlier finding:
the multiple tracks weight bytes over available bandwidth, not the implementation language.

**Both multiples are internal comparisons** (C against our own numpy, INT4 against our own fp32), not against an external engine.
The same-weights comparison against llama.cpp is in §2.1b — that is the external statement.

### 2.1b Same-weights comparison against llama.cpp

One `rwkv7-191M-world-bf16.gguf`, llama.cpp b11889 CPU backend vs the SunnyNet C kernel. Full data in
`EXPERIMENT_REPORT.md` §24.11b. This box throttles, so ratios are per-round paired.

| Threads | Same fp32 | Same 4-bit | Verdict |
|---:|---|---|---|
| 1 | SunnyNet 0.90× (1/7) | SunnyNet 0.89× (0/7) | llama.cpp slightly faster |
| 6 | SunnyNet 0.58× (0/5) | SunnyNet 0.71× (1/5) | llama.cpp clearly faster |

**On the same weights and precision llama.cpp is faster**, about 1.1× single-thread and 1.4–1.7× at six threads. The earlier
"5.39× over numpy" is an internal multiple and is not a claim of leading an external engine. Single-thread both sit near the
sequential read ceiling (12.5 / 13.7 GB/s); the difference is mainly multithread bandwidth utilization (12.5 vs 19.5 GB/s).

### 2.2 OpenAI-compatible server

Official `openai` SDK 3.36.0, `base_url` pointing at the local port.

| Call | Result |
|---|---|
| `models.list()` | 200, `rwkv7-c` |
| `completions.create()` | 200 |
| `chat.completions.create()` | 200 |
| `chat.completions.create(stream=True)` | 200, SSE chunks |
| `temperature="abc"` | 400 `参数类型错误: could not convert string to float` |

Fix on record: the streaming response carried `Connection: keep-alive` with no `Content-Length`.
Under HTTP/1.0 a client reading to EOF hangs forever (`TimeoutError` observed). Switching to
`Connection: close` plus `close_connection = True` ends the stream properly. The NDJSON streams in
`ollama.py` had the same defect and were fixed the same way.

### 2.3 Ollama native API

| Endpoint | Result |
|---|---|
| `GET /api/version` | 200 `{"version": "0.1.0"}` |
| `GET /api/tags` | 200, `rwkv7-c` |
| `POST /api/generate` (non-stream) | 200, `eval_count` matches `num_predict`, `done=true` |
| `POST /api/generate` (stream) | 200, NDJSON line by line, last line `done=true` |
| `POST /api/chat` (non-stream / stream) | 200 / 200 |

### 2.4 Quantization toolchain

```
sunnynet quantize <src.gguf> <dst.gguf> --type q4_0 --ppl --gguf-backend c
```

| Type | Output size | vs source | Tensors |
|---|---:|---:|---|
| Q8_0 | 335.17 MiB | 1.099× | 169 quantized / 173 kept |
| Q4_0 | 268.12 MiB | 1.374× | 169 quantized / 173 kept |

Q4_0 perplexity (512 tokens, C backend): 67.89 → 73.82, +8.73%.

K-quant (Q4_K_M / Q5_K_M / Q6_K) is read-only in gguf-py 0.19.0. The command fails loudly and lists
the writable types rather than silently downgrading:

```
类型 q4_k_m 在 gguf-py 里只能读不能写 (quantize 未实现 K-quant)。
可写类型: q4_0, q4_1, q5_0, q5_1, q8_0
```

### 2.5 convert markings

```
sunnynet convert <src.gguf> <dst.gguf> --arch rwkv7
```

The written GGUF carries `sunnynet.arch` / `sunnynet.version` / `sunnynet.kernel_c` /
`sunnynet.source_sha256`, used to check that a cached blob matches the current weights.

## 3. AVX-512 and NEON: code verified, speed pending

### 3.1 Facts on this machine

| Fact | Value |
|---|---|
| CPU | Intel i7-8750H (Coffee Lake), AVX512F=0 |
| AVX-512 compile | passes (`core_avx512.c` + `step_avx512.c` linked into `sunnynet_core512.dll`) |
| AVX-512 load | runtime CPUID reports unsupported, loader refuses instead of executing an illegal instruction |
| NEON cross-compile AArch64 | `sunnynet_neon.aarch64-linux-gnu.so`, 22264 B, ELF machine 0xb7 |
| NEON cross-compile ARMv7 | `sunnynet_neon.arm-linux-gnueabihf.so`, 881104 B, ELF machine 0x28 |
| Symbols in both | `step_core_neon_f32` / `core_neon_fwd_f32` / `scan_neon_f32` / `sn_neon_isa` present |
| qemu / docker / WSL | all unavailable (WSL blocked by local security policy) |

### 3.2 Running the real source on x86 via scalar shims

There is no AVX-512 CPU and no ARM device here, but the **indexing and in-group semantics** of the
kernel source can still be executed on x86: drop in same-named shims for `immintrin.h` / `arm_neon.h`,
letting `__m512` / `float32x4_t` degrade to 16-element / 4-element arrays. Element-wise semantics match
real SIMD, and the in-group permutation still uses the index vector from the source.

```bash
python experiments/_avx512_shim/_run.py /f/devtools/w64devkit/bin/gcc
python experiments/_neon_shim/_run.py  /f/devtools/w64devkit/bin/gcc
```

| Kernel | max\|diff\| vs torch reference | Coverage |
|---|---:|---|
| AVX-512 core fwd (D=32, full 16-wide loop) | 1.19e-07 | two groups in one register |
| AVX-512 core fwd (D=24, 8-wide tail) | 5.96e-08 | odd NG tail path |
| AVX-512 step core (D=32 / D=24) | 1.19e-07 / 5.96e-08 | as above |
| AVX-512 dw step | 0.0 | 16-lane element-wise + masked tail |
| NEON scan | 0.0 (bit-identical) | 4-lane scan |
| NEON core fwd | 1.19e-07 | group of 8 split into two 4-lane blocks |
| NEON step core | 1.19e-07 | inference-mode single step |

This path only rules out defects where the index is wrong but still in bounds — for instance, if
AVX-512's `idx[half*8+i] = half*8 + ((i+step)&7)` were not partitioned per half, the upper 8 lanes
would read another group's channels and the numbers would still be finite. It does **not** prove
floating-point behaviour or throughput on the real hardware.

Wired into the suite as `tests/test_simd.py::test_avx512_indexing_matches_reference` and
`test_neon_indexing_matches_reference`.

Pitfall: the first shim wrote `_mm512_fmadd_ps(a,b,c)` as `a + b*c` (correct is `a*b + c`). The symptom
was that `u_out` and `p_out` already matched while only `s_new` was wrong, which invites a
misdiagnosis of the scan branch.

### 3.3 FMA fallback on ARMv7

`vfmaq_f32` needs NEONv2. Preprocessor probes on both targets:

| Target | `__ARM_FEATURE_FMA` | `SN_FMA` expands to |
|---|---|---|
| `aarch64-linux-gnu` | defined | `vfmaq_f32` |
| `arm-linux-gnueabihf` (+`-mfpu=neon-vfpv4`) | not defined | `vaddq_f32 + vmulq_f32` |

So 32-bit ARMv7 takes the non-fused path: one fewer rounding, and the encoding is `vmla.f32`
rather than `vfma.f32`.

### 3.4 Pending on real hardware (must not be claimed)

| Target | Original acceptance criterion | Missing |
|---|---|---|
| AVX-512 | 1.5–1.8× faster at equal clock, single-token latency another 30% below AVX2 | no AVX-512 CPU |
| Apple Silicon | M2/M3 running 1.6B Q4 at 30+ tok/s | no Mac |
| Raspberry Pi 5 / RK3588 | RWKV-7 0.4B offline voice assistant under 200 ms | no ARM device, no qemu |
| Ollama comparison | 30%+ faster than the old path | ollama not installed here |
| llama.cpp comparison | 1.6B Q4 30%+ faster than llama.cpp | no llama.cpp source; the test artifact is 191M, not 1.6B |

Until the matching hardware exists, these keep only their source and compile evidence.

## 4. Not done

- On-device code-completion model (original roadmap item 6): conflicts with the established
  "no new training" position, not started.
- The Ollama wrapper covers the API layer only (`sunnynet/ollama.py`). Making
  `ollama run rwkv7:1.6b` route into SunnyNet automatically requires either modifying Ollama itself
  or a custom model definition; not done.
- `sunnynet quantize` supports only the types gguf-py can write. K-quant would need a hand-written
  quantizer; not done.

## 5. Reproduction commands

```bash
export PATH="/f/devtools/w64devkit/bin:$PATH"

# GGUF -> C kernel
python -m sunnynet.cli info   -m experiments/gguf/rwkv7-191M-world-bf16.gguf
python -m sunnynet.cli chat   -m experiments/gguf/rwkv7-191M-world-bf16.gguf --prompt "Once upon a time"

# OpenAI-compatible server
python -m sunnynet.cli serve  -m experiments/gguf/rwkv7-191M-world-bf16.gguf --port 8080

# quantization + perplexity
python -m sunnynet.cli quantize <src.gguf> /tmp/q4.gguf --type q4_0 --ppl --gguf-backend c

# cross-compile
PYTHON=<venv python> bash scripts/build_arm.sh aarch64-linux-gnu
PYTHON=<venv python> bash scripts/build_arm.sh arm-linux-gnueabihf

# kernel semantic equivalence (real source, executed on x86)
python experiments/_avx512_shim/_run.py $(command -v gcc)
python experiments/_neon_shim/_run.py  $(command -v gcc)
```

Regression: `python -m pytest -q` → 166 passed, 42 skipped.