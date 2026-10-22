# Changelog

## 未发布

### fp16 权重解码路径
- 新增 `sn_load_h()`：载入时把 5 个 GEMV 矩阵与 LM head 截断为 fp16，小参数仍 fp32；
  单步实读 467.6 → 约 222.3 MB。内核 `gemv_h` 一次读 16 个 half（32 B）后转 fp32 做
  FMA，保持与 fp32 相同的加载宽度。
- `p100mw` 单步 24.67 → **18.60 ms/token**（**1.33×**，11 轮交错 11/11 占优）。
  logits max|diff| 4.181e-03、相对偏差 7.42e-04、argmax 64/64。
- Python 侧 `SnModel(..., half=True)`，CLI `--backend c --half`。fp32 仍是默认，逐位不变。

## v0.2.0

### 推理 Runtime 访存重构
- 整模型 C 解码的权重在 `sn_load` 时翻成行主序 `(out,in)`，GEMV 改为按输出通道
  并行、每行连续读。导出格式不变，已发布的 blob 无需重导。
- `p100mw`（105.36M）单步解码 76.60 → **25.72 ms/token**（**2.98×**，5 轮交错
  5/5 占优），有效带宽 6.10 → 18.18 GB/s。对优化前 logits max|diff| 2.766e-05、
  argmax 32/32。
- 并行度取物理核而非逻辑核（解码访存受限）：6 核 26.69 ms vs 12 逻辑核 36.05 ms。
  新增 `sn_recommended_threads()` 与 `sunnynet.runtime.model.set_threads()`；
  CLI 在 `--backend c` 且未指定 `--threads` 时自动采用物理核数。

### 外部权重
- GLA 接入（`sunnynet/runtime/gla.py`）：门控只在 key 维，衰减可分解为按行缩放
  ⇒ 元素级递推，不需要矩阵值层级。对独立参考实现 max|diff| 2.05e-05、argmax 12/12、
  分段续算逐位 0.0。CLI `--hf-arch gla`。

### 服务与集成
- OpenAI 兼容 HTTP 服务（`sunnynet/serve.py`，标准库零依赖）：`/v1/models`、
  `/v1/chat/completions`（含 SSE 流式）、`/v1/completions`、`/health`。
  官方 `openai` SDK 四种调用方式验证通过。命令 `sunnynet-serve`。
- Ollama / LM Studio 接入（`sunnynet/integrate.py`）：`sunnynet-integrate` 生成
  Modelfile 与 LM Studio 配置。RWKV-7 GGUF 经 Ollama 真实导入并生成通过
  （`family=rwkv7`、`runner=llamacpp`）。

### C API 稳定 ABI
- `sunnynet/runtime/sn_abi.h` / `sn_abi.c`：`sn_abi_` 前缀隔离、不透明句柄、
  能力位、错误码、`SN_ABI_VERSION 1.0.0`。与内部句柄路径逐位一致（max|diff| 0.0）。
  契约测试 9 项。详见 `docs/ABI.md`。

### SIMD 后端
- AVX-512F/BW/VL 内核（`sunnynet/kernels/scan_avx512.c`）：编译通过，
  反汇编确认 198 处 `zmm` / 16 处 `vfmadd132ps`。本机 CPU 不支持，只做产物验证。
- ARM NEON 内核（`sunnynet/kernels/scan_neon.c`）：`zig cc` 交叉编译出 AArch64 ELF64，
  capstone 确认 4-lane `fmla`。执行验证放在 CI 的 `ubuntu-24.04-arm` runner。

### 权重发布
- `sunnynet-release`（`sunnynet/release.py`）：导出 safetensors（config 与训练摘要
  写入 metadata）、runtime blob、模型卡与 sha256 清单。
- 训练器支持 `--device cuda`，checkpoint 统一以 CPU tensor 落盘。
- `p100mw` 在 GTX 1060 6GB 上训练 2000 步（1.024M token），val ppl 9983 → 120.09，
  0.503 ms/token。发布资产 `.pt` 467.7 MB / `.safetensors` 421.5 MB /
  `.runtime.bin` 467.6 MB，torch、C Runtime、OpenAI 服务三条路径加载生成均通过。

### 测试
- 121 passed / 10 skipped（新增 `test_abi.py`、`test_simd.py`、`test_integrate.py`、
  `test_release.py`）。

## v0.1.0

首个可用版本。

### 原创架构
- SunnyNet：CPU 上训练的通用序列架构（非 Transformer / RNN / Mamba / CNN）。
  CenteredScaleNorm、组内成对通道竞争、对数域并行扫描、多尺度衰减。
- 预设 `proto20mw`（d640/i1280/r384/L6）与 `p100mw`（d1408/i1920/r1024/L8）。
- 不变量自检（`tests/test_architecture.py`）：fp64 单步 vs 整段、分段续算、分块不变性、因果性。

### 手写内核
- AVX2/FMA 融合内核（`sunnynet/kernels/`），训练步 1.10–1.17×。
- 通用一阶线性递推 C Runtime（`sunnynet/runtime/recur.c`），对 torch 向量化基线中位 9.90×。

### 推理 Runtime
- 整模型 C 单步解码（`sunnynet/runtime/sunnynet.c`）：B=1 单线程 T=256/1024/2048 为 1.53×/2.05×/1.97×。
- GGUF 只读（`sunnynet/runtime/gguf.py`），外部权重反量化与官方 `.pth` 逐张量一致。
- RWKV-7 整模型 C 内核（`sunnynet/runtime/rwkv7.c`）：对 numpy 参考 5.39×，8 线程 7.49×；
  单步 logits max|diff| 7.25e-05，120 步 argmax 全一致。
- INT4 分组权重量化（`sunnynet/runtime/quant.c`）：端到端 2.83×，体积 729→269 MB。
- 外部权重接入：Mamba-2（`runtime/mamba2.py`）与 DeltaNet（`runtime/deltanet.py`）
  逐 token 单步递推，不经 PyTorch 前向。Mamba-2 对 `transformers` 官方实现
  argmax 24/24；DeltaNet 贪心续写连贯（"The capital of France is the city of Paris…"）。

### 命令行
- `sunnynet` / `python -m sunnynet.cli`：`--ckpt`（SunnyNet）、`--gguf`（外部权重）、
  `--hf-dir`（Mamba-2 / DeltaNet）、`--backend c`、`--gguf-backend c`、`--quant {fp32,int4}`。
- `sunnynet.runtime.load_model(arch, path)`：按架构名加载外部权重。

### 分发
- 发布名 `sunnynet-cx`，导入名 `sunnynet`，命令 `sunnynet`。
  PyPI 上 `sunnynet` 已属他人项目（网络抓包中间件），同名不可注册。
- 发布走 GitHub Actions 的 Trusted Publisher（OIDC，无需长期令牌）；
  tag `v*` 触发 build → twine check → publish。

### 已知边界
- 内核需 AVX2/FMA 与 C 编译器（首次调用自动编译）。
- INT4 质量：英文保持，中文与代码有退化，未配校准。
- Mamba-2 / DeltaNet / GLA 目前只有 numpy 参考路径（单线程 45.7 / 91.5 / 85.1 ms/token），
  尚未 C 化；矩阵值 rank-1 递推（DeltaNet、RWKV-7）现有内核表达不了。
- 0.5B 预设在本机 6GB 显卡为边界态，未发布质量数字。
---

# Changelog

## Unreleased

### fp16 weight decode path
- Added `sn_load_h()`: at load time the 5 GEMV matrices and the LM head are truncated to fp16, smaller
  parameters stay fp32; per-step actual reads drop 467.6 -> ~222.3 MB. The kernel `gemv_h` loads
  16 halves at a time (32 B) then converts to fp32 for FMA, keeping the same load width as fp32.
- `p100mw` single step 24.67 -> **18.60 ms/token** (**1.33x**, 11/11 interleaved rounds won).
  Logits max|diff| 4.181e-03, relative deviation 7.42e-04, argmax 64/64.
- Python side `SnModel(..., half=True)`, CLI `--backend c --half`. fp32 remains the default and is
  bit-identical.

## v0.2.0

### Inference runtime memory-access restructuring
- The whole-model C decode weights are flipped to row-major `(out,in)` at `sn_load` time; GEMV is
  parallelized over output channels with contiguous reads per row. The export format is unchanged,
  so already-published blobs need no re-export.
- `p100mw` (105.36M) single-step decode 76.60 -> **25.72 ms/token** (**2.98x**, 5/5 interleaved
  rounds won), effective bandwidth 6.10 -> 18.18 GB/s. Logits max|diff| 2.766e-05 vs the
  pre-optimization version, argmax 32/32.
- Parallelism uses physical rather than logical cores (decode is memory-bound): 6 physical cores
  26.69 ms vs 12 logical cores 36.05 ms. New `sn_recommended_threads()` and
  `sunnynet.runtime.model.set_threads()`; the CLI automatically adopts the physical core count when
  `--backend c` is set and `--threads` is not.

### External weights
- GLA integration (`sunnynet/runtime/gla.py`): the gate acts only on the key dimension and the
  decay factors into per-row scaling, so it is an elementwise recurrence, no matrix-valued layer
  needed. Against the independent reference implementation max|diff| 2.05e-05, argmax 12/12,
  segmented continuation bit-identical 0.0. CLI `--hf-arch gla`.

### Service and integration
- OpenAI-compatible HTTP service (`sunnynet/serve.py`, stdlib only, zero dependencies):
  `/v1/models`, `/v1/chat/completions` (incl. SSE streaming), `/v1/completions`, `/health`.
  All four official `openai` SDK call patterns verified. Command: `sunnynet-serve`.
- Ollama / LM Studio integration (`sunnynet/integrate.py`): `sunnynet-integrate` generates the
  Modelfile and LM Studio configuration. RWKV-7 GGUF imported through Ollama and generation
  verified (`family=rwkv7`, `runner=llamacpp`).

### Stable C API ABI
- `sunnynet/runtime/sn_abi.h` / `sn_abi.c`: `sn_abi_` symbol prefix isolation, opaque handles,
  capability bits, error codes, `SN_ABI_VERSION 1.0.0`. Bit-identical to the internal handle path
  (max|diff| 0.0). 9 contract tests. See `docs/ABI.md`.

### SIMD backends
- AVX-512F/BW/VL kernel (`sunnynet/kernels/scan_avx512.c`): compiles, disassembly confirms 198
  `zmm` uses / 16 `vfmadd132ps`. The host CPU does not support it, so only the artifact was
  verified.
- ARM NEON kernel (`sunnynet/kernels/scan_neon.c`): cross-compiled via `zig cc` to an AArch64
  ELF64; capstone confirms the 4-lane `fmla`. Execution verification runs on the CI
  `ubuntu-24.04-arm` runner.

### Weight release
- `sunnynet-release` (`sunnynet/release.py`): exports safetensors (config and training summary
  written to metadata), the runtime blob, a model card, and a sha256 manifest.
- The trainer supports `--device cuda`; checkpoints are stored uniformly as CPU tensors.
- `p100mw` trained 2000 steps on a GTX 1060 6GB (1.024M tokens), val ppl 9983 -> 120.09,
  0.503 ms/token. Released assets: `.pt` 467.7 MB / `.safetensors` 421.5 MB /
  `.runtime.bin` 467.6 MB; loading and generation verified through all three paths: torch, C
  runtime, and the OpenAI service.

### Tests
- 121 passed / 10 skipped (added `test_abi.py`, `test_simd.py`, `test_integrate.py`,
  `test_release.py`).

## v0.1.0

First usable version.

### Original architecture
- SunnyNet: a general-purpose sequence architecture trained on CPU (not Transformer / RNN /
  Mamba / CNN). CenteredScaleNorm, intra-group pairwise channel competition, log-domain parallel
  scan, multi-scale decay.
- Presets `proto20mw` (d640/i1280/r384/L6) and `p100mw` (d1408/i1920/r1024/L8).
- Invariant self-checks (`tests/test_architecture.py`): fp64 single step vs full pass, segmented
  continuation, chunk invariance, causality.

### Hand-written kernels
- AVX2/FMA fused kernel (`sunnynet/kernels/`), 1.10-1.17x on the training step.
- General first-order linear recurrence C runtime (`sunnynet/runtime/recur.c`), 9.90x median vs
  the torch vectorized baseline.

### Inference runtime
- Whole-model C single-step decode (`sunnynet/runtime/sunnynet.c`): 1.53x/2.05x/1.97x at
  T=256/1024/2048 with B=1 single thread.
- GGUF read-only loader (`sunnynet/runtime/gguf.py`); external weight dequantization is
  tensor-identical to the official `.pth`.
- RWKV-7 whole-model C kernel (`sunnynet/runtime/rwkv7.c`): 5.39x vs the numpy reference, 7.49x
  with 8 threads; single-step logits max|diff| 7.25e-05, argmax identical over 120 steps.
- INT4 grouped weight quantization (`sunnynet/runtime/quant.c`): 2.83x end-to-end, size
  729 -> 269 MB.
- External weight integration: Mamba-2 (`runtime/mamba2.py`) and DeltaNet
  (`runtime/deltanet.py`) per-token single-step recurrence without a PyTorch forward. Mamba-2
  matches the official `transformers` implementation with argmax 24/24; DeltaNet greedy
  continuation is coherent ("The capital of France is the city of Paris...").

### Command line
- `sunnynet` / `python -m sunnynet.cli`: `--ckpt` (SunnyNet), `--gguf` (external weights),
  `--hf-dir` (Mamba-2 / DeltaNet), `--backend c`, `--gguf-backend c`,
  `--quant {fp32,int4}`.
- `sunnynet.runtime.load_model(arch, path)`: loads external weights by architecture name.

### Distribution
- Release name `sunnynet-cx`, import name `sunnynet`, command `sunnynet`.
  `sunnynet` on PyPI belongs to another project (a network-capture middleware), so the name is
  unavailable.
- Releases go through the GitHub Actions Trusted Publisher (OIDC, no long-lived tokens);
  a `v*` tag triggers build -> twine check -> publish.

### Known limits
- The kernels need AVX2/FMA and a C compiler (compiled automatically on first use).
- INT4 quality: English is preserved, Chinese and code degrade; no calibration applied.
- Mamba-2 / DeltaNet / GLA have only the numpy reference path so far (single-threaded
  45.7 / 91.5 / 85.1 ms/token), not yet ported to C; the matrix-valued rank-1 recurrence
  (DeltaNet, RWKV-7) cannot be expressed by the existing kernels.
- The 0.5B preset is at the boundary state on this machine's 6GB GPU; no quality numbers
  published.
