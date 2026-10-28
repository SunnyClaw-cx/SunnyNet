*[中文版](CHANGELOG.md)*

# Changelog

## Unreleased

### Associative memory: delta write, write-key alignment, and numerical boundary
- `AssocMemory` gains `assoc_write='delta'`: `S_t = p⊙S_{t-1} − β k_t(k_tᵀS_{t-1}) + β(v_t⊗k_t)`,
  subtracting the key's old association before writing the new value, so it overwrites a key exactly.
  The soft (convex-combination) write reads back mostly the old value on overwrite recall
  (relative error 0.75–0.92); delta reads back the new value (0.06).
- Chunked form `ops.chunk_delta_scan` (WY: `W=(I+L)⁻¹diag(β)(V−K S_in)`, L lower triangular), matching
  the sequential delta to 1e-16 in fp64 (any chunk size, with or without decay). The delta rule and the
  WY representation belong to DeltaNet / FLA.
- `_safe_chunk_delta` caps the block length at 32: in fp32, chunk=64 produces non-finite decay gradients
  in the backward pass (anomaly detector points to `AminBackward0`), diverging in 2/3 seeds (steps 237/249);
  with the cap, 3/3 pass.
- The write key is taken from the previous position (`assoc_shift`): with k/v/q all from the current
  position the memory can only express token → token associations and cannot store (key → value). This is
  why early recall experiments left both write rules at chance (0.02 vs. 1/32); after the fix, MQAR gives
  off 0.033 / soft 0.936 / delta 0.961.
- Training-step cost (`experiments/_assoc_ab_speed.py`, subprocess per cell, interleaved rounds): the
  chunked delta write is *cheaper* than the soft write — soft materializes the outer product into a
  (B,T,slots·m) tensor and scans it, so log/cumsum/exp all run at width slots·m (cumsum alone is 38%
  of soft's time), while delta keeps the state at (slots,m) and only solves one C×C triangular system
  per block. delta/soft training step = 0.84–0.89× (32×16) / 0.58–0.65× (64×32) / 0.51–0.62× (128×64),
  at a parameter cost of one `W_b` (+516, +0.034%).
- **`assoc_decay='scalar'`**: the chunked delta accepts only a scalar decay, and the code collapsed the
  per-channel head `W_g`'s $(B,T,m)$ output to $(B,T)$ with `amin` — taking the fastest-forgetting
  channel, driving the block's peak $1/\prod d$ condition number to **9.92e16** at chunk=64 (against
  1.78 for `amax`) while keeping only 1 of 16 channels. Replaced by a dedicated `W_gd(dim→2)` that
  outputs (keep-rate, write strength) directly with no reduction. $p$ follows the same H
  parameterization as the per-channel head ($p \ge 1-1/h_{\min} = 0.5$); a bare $\sigma$ lets $p$ fall
  to 0.01 and diverges (measured: it moved the collapse from seed 7 to seed 2024 — now fixed). After
  the fix: **7,160 fewer parameters than soft (−0.47%)**, the MQAR seed collapse is gone
  (seed 7: 0.0141 → 0.9484), overwrite recall and ppl do not regress, training step 0.867× and decode
  0.955× (4 of 5 paired rounds).
- `_safe_chunk_delta` now derives the block length from the actual decay ($d_{\min}^{-L} \le 2^{50}$).
  The headroom was $2^{80}$, 30 bits wider than the measured safety level ($2^{57}$–$2^{62}$), and
  returned $L$=24/34 for $p$=0.1/0.2, both of which produce non-finite decay gradients in measurement.
  At $2^{50}$ every $p\in[0.05,0.9]$ is stable. On the production path ($p \ge 0.5$) by_cond is always
  $\ge 50 > \text{cap}$, so the block length is unchanged.
- Block length 32 is set by stability, not by speed: across 22 cross-process readings the median ratio
  is 0.94× for 32/64 and 0.91× for 32/128, while 64's own spread (34.55 ms) dwarfs the 3.8 ms median
  gap, so all block lengths tie on speed and no ranking is resolvable (ranking them inside one process
  yields false differences). Stability is unambiguous: at $p$=0.5 the length-64 block already yields
  non-finite decay gradients and $L$=128 is NaN in the forward pass. `cap=32` stays.
- 4 new invariant tests: scalar-decay forward/step agreement, per-channel head not built, block length
  shrinking with decay.
- 1 new guard test: the reversed block length must satisfy $d_{\min}^{-L} \le 2^{50}$, pinning the
  headroom against regression.
- 15 new invariant tests: shift forward/step agreement, chunk invariance, causality, overwrite, block cap.

### Gate attribution and originality notes
- `experiments/_gate_originality.py`: the operator-level Jacobian spectral norm rises with d
  (softmax 0.32→0.39, no 1/d decay); model-level gradients do not decay with depth. The earlier claim of
  "softmax Jacobian's 1/d factor kills gradients" does not hold and has been corrected.
- Gate-form quality comparison (dim=128, 4 layers, 600 steps, zh_ancient, 3 seeds): pairwise 326.61 /
  softmax 327.96 / scalar 325.27 / none 369.42. The three gates differ far less than seed noise or the
  no-gate gap: the presence of a gate is what matters; the form differences are within noise.
- `docs/{cn,en}/ORIGINALITY.md` (new): prior work listed component by component, with the claimable scope marked.
- README / ARCHITECTURE / CHANGELOG wording of "an original architecture that is not Transformer / RNN"
  did not match measurement; changed to a linear-recurrence-family statement.

## v0.3.0

### The cost mechanism of depth scaling and new presets
- Training step scales linearly with layer count: fixed `d=128`, T=256/B8/6 threads, 3 interleaved rounds measured
  `step_ms ≈ 32.9·L + 102.4`. Each layer adds a fixed ~**1100 aten calls**, of which `aten::mm`
  accounts for only 1.9%, elementwise and view operators account for 95.7%, and `mm` is only 21 calls per layer
  ⇒ **each layer is an operator dispatch tax independent of width**.
- Fused kernels lower the per-layer slope from 1099 to 715 calls. Segment-wise attribution (replacement experiments give the upper bound of gains) shows this tax
  is spread across `scan`/`readout`/`compete`/`local`/`emit` five segments (after zeroing each, per-layer drops to 0.65–0.74×),
  no single segment can claim it alone ⇒ the gains from further fusion are limited.
- Under the same stack parameters (365,844), 6 `(L, dim)` families scanned, 900 steps 2 seeds: `L3 wide` fastest
  (0.0995 ms/token), `L4 wide` (d121) best quality (zh ppl 247.2), `L6`/`L8` three families
  worse on **both axes**, speed and quality ⇒ depth does not buy quality under the same parameter budget.
- New preset `p100msw` (d1536/i2624/r1152/L6, stack 94,470,942, aligned with `p100mw` +0.68%):
  isolated A/B (B4×T256, 3 interleaved rounds) faster than `p100mw` (d1408/L8) by **4.2%** (3/3 rounds better),
  layer count back to the same tier. **Do not pile parameters onto depth—this is configuration discipline.**

### Locating and fixing inter-layer input dilution
- Per-branch common-component attribution (`experiments/_dilution_source.py`): in the common component accumulated layer by layer in deep models,
  the `W_c(y)` branch accounts for about **96%**, a constant independent of input that drifts slowly, self-reinforced through the residual stream
  (state DC → residual stream DC → `norm_in` → `u` → `y` → `W_c(y)`).
  The previously read "state collapse 0.50" from raw pairwise cosine is a phantom of the common component; after mean removal it is 0.
- `state_norm` changed to also normalize the feedback source `sfb` (previously only covered `sn`), plugging the DC leak back through the feedback branch:
  on `deepl8` at 6000 steps, 24-input argmax diversity **12 → 16**, val ppl 200.51 → **198.33**.
- Residual-stream normalization `h_norm` falsified by measurement (ppl 263 → 625, 24 inputs collapse to 1 prediction), kept as a control.

### DeltaNet full-model C kernel
- `runtime/deltanet.c`: the entire forward pass in C (three-way depthwise short convolution + per-head L2 + delta rule +
  RMSNorm + SwiGLU), weights exported to an fp32 blob by `experiments/_export_deltanet.py`.
  Python side `sunnynet.runtime.deltanet_c.SnDeltaNet`, CLI `--hf-arch deltanet --blob`.
- Against the numpy reference path, 36 tokens: logits max|diff| **1.335e-05**, argmax **36/36**,
  repeated runs and reruns after reset bitwise 0.0, prefill and step-by-step 0.0.
- Speed single step 78.69 → **71.24 ms/token (1.10×, 5/5 rounds better)**. The reason for only 1.10× is that
  **both paths hit the memory bandwidth wall**: the C side's effective bandwidth is 19.18 GB/s, already above the single-thread sequential read limit
  of 18.63 GB/s. For memory-bound full-model decoding, the upper bound of C-ification gains is set by bandwidth.
- Thread count set to physical cores (6 threads 64.33 ms / 21.24 GB/s, better than 12 logical cores 68.55 ms);
  default takes physical cores automatically. `tests/test_deltanet_c.py` 7 items.

### Mamba-2 full-model C kernel
- `runtime/mamba2.c`: the entire forward pass in C (single-way depthwise short convolution + elementwise state recurrence + RMSNorm,
  no L2/dual norm/SwiGLU MLP, decay broadcast along the state dim so no matrix multiply needed), weights exported by
  `experiments/_export_mamba2.py` to a 516 MB fp32 blob.
  Python side `sunnynet.runtime.mamba2_c.SnMamba2`, CLI `--hf-arch mamba2 --blob`.
- Against the numpy reference path, 36 tokens: logits max|diff| **1.678e-04**, relative **6.307e-06**,
  argmax **36/36**. Absolute error an order of magnitude larger than DeltaNet; per-layer table comparison
  (`experiments/_mamba2_layer_probe.py`) confirms all 24 layers' relative errors are within 4e-08 ~ 1.1e-06
  and do not grow with layer count ⇒ single-step reduction order noise, not cumulative, not a formula error.
- Speed single step 47.36 → **34.59 ms/token (1.37×, 5/5 rounds better)**. The multiplier higher than DeltaNet's
  1.10× is likewise due to bandwidth: weights 0.52 GB are only 38% of DeltaNet's 1.37 GB,
  neither path reaches the 18.63 GB/s limit. `tests/test_mamba2_c.py` 7 items.

### GLA full-model C kernel
- `runtime/gla.c`: the entire forward pass in C (low-rank decay projection + elementwise state recurrence + output gate +
  SwiGLU MLP), weights exported by `experiments/_export_gla.py` to a 1.37 GB fp32 blob (source bf16).
  Python side `sunnynet.runtime.gla_c.SnGLA`, CLI `--hf-arch gla --blob`.
- Against the numpy reference path, 36 tokens: logits max|diff| **1.431e-05**, relative **1.081e-06**,
  argmax **36/36**, repeated and reset bitwise 0.0, prefill and step-by-step 0.0.
- Speed single step 74.92 → **74.15 ms/token (1.01×, 4/5 rounds better)**. Weights 1.37 GB on the same order as DeltaNet,
  theoretical lower bound `1.37 GB ÷ 18.63 GB/s ≈ 73.5 ms`, the median readings of both paths already touch this line
  ⇒ no margin left for C-ification. The three architectures jointly confirm: **gains are decided by "weight bytes / available bandwidth"**.
- New in-layer stage probe `sn_gla_probe` (10 stages), locating that in `gla_scan` all heads read
  segment 0 of `v` instead of their own value segment (single-step readout differs by 96%, yet all projections are correct).
  When `head_k_dim ≠ head_v_dim` this error **does not go out of bounds**, it only silently computes wrong. `tests/test_gla_c.py` 8 items.

### Matrix-valued rank-1 recurrence kernel
- New `sn_mat_scan` (`runtime/recur.c`, declaration in `recur.h`): unified form
  `p=S·u; S=S⊙w+v⊗k−p⊗b; y=S·r`, passing `NULL` for a parameter skips it, covering the matrix state layers of RWKV-7 and
  DeltaNet. Per-head serial, vectorized along the `C` dim with width 8 + scalar tail.
- Python side `sunnynet.runtime.mat_step(S, w=, u=, v=, k=, b=, r=)`, `S` as `(NH,R,C)`
  contiguous fp32, updated in place, returns `y`.
- Numerical contract `tests/test_mat_scan.py` 11 cases all pass, relative error on the order of 1e-7.
- Operator-level 8×128×128 is **4.25×** of numpy; but measured to account for only **2.2%** of a DeltaNet single step,
  no end-to-end speedup can be claimed from this—the remaining time is per-layer GEMV.

### Prebuilt libraries and source consistency
- `build_prebuilt()` writes `_prebuilt/<platform>/sources.sha256` (source content hashes grouped by library),
  `locate()` compares before loading; when sources changed but were not recompiled it no longer silently hits the old library, falling back to source compilation directly
  (actually hit during development: `sn_mat_scan` was already compiled into the source yet reported `AttributeError`).
- Compares content hashes instead of mtime—pip unpacking refreshes source mtime to the unpack time, and judging by time would always judge stale.
  Isolated install measured all libraries hit, `prebuilt_stale()` is False. Library count increased from three full-model libraries
  (DeltaNet / Mamba-2 / GLA) to 9.
- Contract tests `tests/test_native_prebuilt.py` (7 items).

### Two Windows defects at build time
- `build_prebuilt()` originally called each module's `_compile` directly, writing to the shared `<pkg>/_build/<dll>`;
  once that file has been loaded by this process, Windows refuses to overwrite it and compilation fails (the full test suite reports
  3 `RuntimeError`s in an order-dependent way, yet running that file alone passes). Changed to compile to a temp directory then copy; when the output dll is occupied
  `PermissionError` is caught per library, without dragging down the rest.
- Six `_compile` sites used `subprocess.run(..., text=True)` decoding as UTF-8, whereas under Windows
  gcc diagnostics are output in the system code page (GBK), and paths containing non-ASCII throw `UnicodeDecodeError` masking the real
  error (only `abi.py` tolerated it separately). Unified through `_native.run_cc()` collecting bytes then tolerantly decoding.
- Each module gains `_link_cmd(gcc, dst)`, `_compile` only writes to the target path; contract tests assert every
  module provides this interface.
- `deltanet_c._recommended_threads()` no longer does its own `CDLL` on `_build/sunnynet_model.dll`
  (bypassing prebuilt lookup, and the trigger of the above item), changed to reuse `runtime.model.recommended_threads()`.
- `prebuilt_stale()`'s decision has an in-process cache, not cleared after `build_prebuilt()` rebuilds, causing "change source
  → rebuild → load" to keep reading a stale conclusion (equivalent to still falling back to source compilation). Clears the cache at the end of rebuild.
- Contract tests originally hardcoded the artifact name `sunnynet_runtime.dll`, while manifest keys carry a platform suffix (Linux is
  `.so`), causing a `KeyError` on `ubuntu-latest`; Windows locally happened to match so it never surfaced.
  Changed to take `DLL.name` from the module.

### fp16 weight decode path
- New `sn_load_h()`: at load time truncates the 5 GEMV matrices and the LM head to fp16, small parameters remain fp32;
  single-step actual reads 467.6 → about 222.3 MB. Kernel `gemv_h` reads 16 halves (32 B) at a time then converts to fp32 for
  FMA, keeping the same load width as fp32.
- `p100mw` single step 24.67 → **18.60 ms/token** (**1.33×**, 11 interleaved rounds 11/11 better).
  logits max|diff| 4.181e-03, relative deviation 7.42e-04, argmax 64/64.
- Python side `SnModel(..., half=True)`, CLI `--backend c --half`. fp32 is still the default, bitwise unchanged.

## v0.2.0

### Inference runtime memory access refactor
- Full-model C decoding weights transposed to row-major `(out,in)` at `sn_load`, GEMV changed to parallel
  over output channels, reading each row contiguously. Export format unchanged, published blobs need no re-export.
- `p100mw` (105.36M) single-step decode 76.60 → **25.72 ms/token** (**2.98×**, 5 interleaved rounds
  5/5 better), effective bandwidth 6.10 → 18.18 GB/s. Against pre-optimization logits max|diff| 2.766e-05,
  argmax 32/32.
- Parallelism takes physical cores rather than logical cores (decode is memory-bound): 6 cores 26.69 ms vs 12 logical cores 36.05 ms.
  New `sn_recommended_threads()` and `sunnynet.runtime.model.set_threads()`;
  CLI automatically uses the physical core count when `--backend c` and `--threads` is unspecified.

### External weights
- GLA integration (`sunnynet/runtime/gla.py`): the gate is only on the key dim, the decay can be decomposed into row-wise scaling
  ⇒ elementwise recurrence, no matrix-valued layer needed. Against an independent reference implementation max|diff| 2.05e-05, argmax 12/12,
  segmented continuation bitwise 0.0. CLI `--hf-arch gla`.

### Serving and integration
- OpenAI-compatible HTTP service (`sunnynet/serve.py`, standard library zero-dependency): `/v1/models`,
  `/v1/chat/completions` (including SSE streaming), `/v1/completions`, `/health`.
  Verified through the official `openai` SDK's four call styles. Command `sunnynet-serve`.
- Ollama / LM Studio integration (`sunnynet/integrate.py`): `sunnynet-integrate` generates
  the Modelfile and LM Studio config. RWKV-7 GGUF really imported through Ollama and generated successfully
  (`family=rwkv7`, `runner=llamacpp`).

### C API stable ABI
- `sunnynet/runtime/sn_abi.h` / `sn_abi.c`: `sn_abi_` prefix isolation, opaque handles,
  capability bits, error codes, `SN_ABI_VERSION 1.0.0`. Bitwise consistent with the internal handle path (max|diff| 0.0).
  Contract tests 9 items. See `docs/cn/ABI.md`.

### SIMD backends
- AVX-512F/BW/VL kernels (`sunnynet/kernels/scan_avx512.c`): compiles,
  disassembly confirms 198 `zmm` / 16 `vfmadd132ps`. This machine's CPU does not support it, only artifact verification is done.
- ARM NEON kernels (`sunnynet/kernels/scan_neon.c`): `zig cc` cross-compiles to AArch64 ELF64,
  capstone confirms 4-lane `fmla`. Execution verification is placed on CI's `ubuntu-24.04-arm` runner.

### Weight release
- `sunnynet-release` (`sunnynet/release.py`): exports safetensors (config and training summary
  written into metadata), runtime blob, model card and sha256 manifest.
- Trainer supports `--device cuda`, checkpoints uniformly written to disk as CPU tensors.
- `p100mw` trained 2000 steps on a GTX 1060 6GB (1.024M tokens), val ppl 9983 → 120.09,
  0.503 ms/token. Release assets `.pt` 467.7 MB / `.safetensors` 421.5 MB /
  `.runtime.bin` 467.6 MB, load-and-generate passes on all three paths: torch, C Runtime, OpenAI service.

### Tests
- 121 passed / 10 skipped (new `test_abi.py`, `test_simd.py`, `test_integrate.py`,
  `test_release.py`).

## v0.1.0

First usable version.

### Original architecture
- SunnyNet: a general sequence architecture trained on CPU (not Transformer / RNN / Mamba / CNN).
  CenteredScaleNorm, intra-group pairwise channel competition, log-domain parallel scan, multi-scale decay.
- Presets `proto20mw` (d640/i1280/r384/L6) and `p100mw` (d1408/i1920/r1024/L8).
- Invariant self-checks (`tests/test_architecture.py`): fp64 single step vs whole segment, segmented continuation, chunk invariance, causality.

### Hand-written kernels
- AVX2/FMA fused kernels (`sunnynet/kernels/`), training step 1.10–1.17×.
- General first-order linear recurrence C Runtime (`sunnynet/runtime/recur.c`), median 9.90× over the torch vectorized baseline.

### Inference runtime
- Full-model C single-step decode (`sunnynet/runtime/sunnynet.c`): B=1 single-thread T=256/1024/2048 is 1.53×/2.05×/1.97×.
- GGUF read-only (`sunnynet/runtime/gguf.py`), external weight dequantization matches the official `.pth` tensor by tensor.
- RWKV-7 full-model C kernel (`sunnynet/runtime/rwkv7.c`): 5.39× over the numpy reference, 7.49× with 8 threads;
  single-step logits max|diff| 7.25e-05, argmax all identical over 120 steps.
- INT4 grouped weight quantization (`sunnynet/runtime/quant.c`): end-to-end 2.83×, size 729→269 MB.
- External weight integration: Mamba-2 (`runtime/mamba2.py`) and DeltaNet (`runtime/deltanet.py`)
  per-token single-step recurrence, without going through the PyTorch forward pass. Mamba-2 against the `transformers` official implementation
  argmax 24/24; DeltaNet greedy continuation coherent ("The capital of France is the city of Paris…").

### Command line
- `sunnynet` / `python -m sunnynet.cli`: `--ckpt` (SunnyNet), `--gguf` (external weights),
  `--hf-dir` (Mamba-2 / DeltaNet), `--backend c`, `--gguf-backend c`, `--quant {fp32,int4}`.
- `sunnynet.runtime.load_model(arch, path)`: loads external weights by architecture name.

### Distribution
- Release name `sunnynet-cx`, import name `sunnynet`, command `sunnynet`.
  On PyPI `sunnynet` already belongs to another project (network packet capture middleware), the same name cannot be registered.
- Release goes through GitHub Actions Trusted Publisher (OIDC, no long-lived token);
  tag `v*` triggers build → twine check → publish.

### Known limits
- Kernels require AVX2/FMA and a C compiler (auto-compiled on first call).
- INT4 quality: English holds, Chinese and code degrade, not calibrated.
- Mamba-2 / GLA currently only have the numpy reference path (single-thread 45.7 / 85.1 ms/token), not yet C-ified.
  The DeltaNet full model is already in C (`sn_dn_step`), but is limited to 1.10× by memory bandwidth.
- DeltaNet's C path needs a pre-exported 1.37 GB fp32 blob (`--blob`), not distributed with the package.
- The 0.5B preset is a marginal state on this machine's 6GB GPU, no quality numbers released.