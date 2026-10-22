# CPU 训练速度剩余空间

机器：i7-8750H（Coffee Lake，6C12T，AVX2，无 AVX-512 / 无 VNNI / 无 AMX）
torch 2.14.1+cpu，MKL 2026.1 + oneDNN v3.12 已启用，`capability=AVX2`。
数字来自 `experiments/` 下本机实测，每格独立子进程。

## 四个方向的判定

| 方向 | 判定 | 依据 |
|---|---|---|
| 算子融合 | 可做，已落地 | w64devkit（GCC 16.2.0）可用，手写 AVX2+FMA 内核已编译验证（§21/§22）；训练步提速 1.10–1.17× |
| 矩阵效率 | 基本没有空间 | 实际尺寸 GEMM 已达 238 GFLOPS（峰值 296 的 80%）；`ms/token` 对 batch 不敏感 |
| SIMD | 可做，见"算子融合" | 同上，编译器已具备 |
| 量化 | 反向：只会更慢 | fp16/bf16 matmul 0.57–0.59 GFLOPS vs fp32 296 GFLOPS（慢约 500×）；INT8 1.44 GFLOPS（慢约 200×） |

## 一、训练步构成（T=256/B=8/dim=80/L=4，8 线程）

```
前向 49.2 ms (24.8%)  |  CE 14.5 ms (7.3%)  |  反向 113.1 ms (56.9%)  |  完整步 198.7 ms
aten 算子调用：4524 次/步
```

反向占 56.9%，是最大单块；前向不到 1/4。热点 top（self CPU 占比）：

| 算子 | 占比 | 调用次数 |
|---|---|---|
| `aten::mm` | 35.2% | 63 |
| `_log_softmax_backward_data` | 7.1% | 1 |
| `aten::mul` | 6.6% | 232 |
| `_log_softmax` | 6.0% | 1 |
| `aten::cat` | 4.6% | 44 |
| `aten::fill_` / `add` / `mkldnn_convolution` | 3.9 / 2.8 / 2.6% | 51 / 69 / 4 |

`mm` 已是高效核，其余是分散的小算子（4524 次调用）——减算子数是唯一有效方向。

## 二、矩阵效率：没有空间

| 形状 | GFLOPS |
|---|---|
| 实际训练尺寸（8192×80×160，即 B=32 折叠） | 237.9 |
| dim=128 旧配置（8192×128×256） | 274.2 |
| 方阵 1024³ fp32（峰值） | 295.8 |

判据：`ms/token` 随 batch 几乎不变——若被小形状拖累，加大 batch 应显著降低 per-token 成本。

| B | ms/step | ms/token | 吞吐 |
|---|---|---|---|
| 4 | 110.6 | 0.10803 | 9257 tok/s |
| 8 | 185.3 | 0.09049 | 11051 tok/s |
| 16 | 424.3 | 0.10358 | 9654 tok/s |
| 32 | 860.7 | 0.10507 | 9518 tok/s |

⇒ 不是矩阵效率问题，是随 token 线性增长的逐元素/调度开销。加大 batch 不能省 per-token 成本。
（B=8 取 0.0905 ms/token，与既有记录 0.0953 一致。）

## 三、量化：在本机是反向优化

| 精度 | matmul GFLOPS | 相对 fp32 |
|---|---|---|
| fp32 | 295.8 / 282.9 / 235.7（n=1024/512/256） | 1× |
| fp16 | 0.593（n=512） | 慢约 477× |
| bf16 | 0.569（n=1024） | 慢约 520× |
| int8（`torch._int_mm`） | 1.443（1024³） | 慢约 205× |

原因：AVX2 无原生 FP16/BF16 运算、Coffee Lake 无 VNNI，torch 走模拟/回退路径。

数值上也不可行：扫描累积的 fp32 相对误差 4.7e-07（T=2048 仍为 4.5e-07，分块递推稳定），
而 bf16 达 2.75%。量化会同时击穿收敛与 `tests/test_architecture.py` 的不变量。

## 四、算子融合：编译器已具备

| 工具 | 检测结果 | 实测 |
|---|---|---|
| MSVC `cl.exe` | 存在但缺 SDK | 确实存在（14.39.33519），缺 Windows SDK 头文件 |
| `clang` / `gcc` / `g++` | 系统 PATH 无 | w64devkit（GCC 16.2.0）自带 ninja/make/cmake/ar/ld + mingw-w64 完整头文件，不需 Windows SDK |

链路：C 源 → `gcc -O3 -mavx2 -mfma -shared` → `.dll` → `ctypes` 调用，
前向/反向均与 PyTorch autograd 数值一致。

- **已落地**：CSN 走 `F.layer_norm`（oneDNN 融合核，前向 19.5×/反向 4.46×）；
  扫描两级并行；竞争门控 + 门控概率 + 扫描融成单趟内核（§22）。
- **实测结果**：scan 内核比 torch 快 4.3–22×（算子级）；融合内核接入模型后
  训练步 1.10–1.17×（§22.3/§22.6）。
- **推理态内核收益不显著**（B=1 张量太小，ctypes FFI 与单算子同量级），默认关闭。

注意：算子级微基准的优势会被端到端摊薄进噪声（已出现两次），
微基准不得单独作为采纳依据。

## 五、线程数（未定论）

既有 `_train_bench.py` 固定 8 线程。实测真实训练步：

| 线程 | 中位 ms/step | 轮间极差 |
|---|---|---|
| 4 | 168.5 | ±15.2% |
| 6 | 193.4 | ±8.4% |
| 8 | 183.3 | ±15.6% |
| 12 | 225.4 | ±7.7% |

12 线程明确最差（+23%）——与纯 matmul 的结论相反（方阵 matmul 12 线程反而 394 GFLOPS 最高）。
训练步由小张量主导，多线程只增同步开销。
4 线程名义比 8 快 8.8%，但落在 ±15% 轮间极差内，不足以判定。
若改动线程默认值，既有全部对照数据须重测。

## 六、结论

CPU 训练速度已接近本机架构下的结构性上限。剩余空间不在矩阵、不在量化，
而在反向的 4524 次小算子调度。编译器具备后，这条路已被走到融合内核，
实测训练步 1.10–1.17×，边际收益递减。

若有量级变化，瓶颈在硬件：AVX2-only 的 2018 年移动 CPU 既不支持低精度加速，
也不支持量化加速——这两条在现代服务器（AVX-512/VNNI/AMX）上结论会不同。
---

# Remaining headroom in CPU training speed

Machine: i7-8750H (Coffee Lake, 6C12T, AVX2, no AVX-512 / no VNNI / no AMX)
torch 2.14.1+cpu, MKL 2026.1 + oneDNN v3.12 enabled, `capability=AVX2`.
Numbers come from local measurements under `experiments/`, each cell in an isolated subprocess.

## Verdicts on the four directions

| Direction | Verdict | Basis |
|---|---|---|
| Operator fusion | Doable, already implemented | w64devkit (GCC 16.2.0) available; hand-written AVX2+FMA kernels compiled and verified (§21/§22); training step 1.10-1.17x faster |
| Matrix efficiency | Essentially no headroom | GEMMs at real sizes already reach 238 GFLOPS (80% of the 296 peak); `ms/token` is insensitive to batch |
| SIMD | Doable, see "operator fusion" | Same as above, the compiler is in place |
| Quantization | Backwards: it would be slower | fp16/bf16 matmul 0.57-0.59 GFLOPS vs fp32 296 GFLOPS (~500x slower); INT8 1.44 GFLOPS (~200x slower) |

## 1. Training step composition (T=256/B=8/dim=80/L=4, 8 threads)

```
forward 49.2 ms (24.8%)  |  CE 14.5 ms (7.3%)  |  backward 113.1 ms (56.9%)  |  full step 198.7 ms
aten operator calls: 4524/step
```

Backward is 56.9%, the largest single block; forward is under 1/4. Top hotspots (self CPU share):

| Operator | Share | Call count |
|---|---|---|
| `aten::mm` | 35.2% | 63 |
| `_log_softmax_backward_data` | 7.1% | 1 |
| `aten::mul` | 6.6% | 232 |
| `_log_softmax` | 6.0% | 1 |
| `aten::cat` | 4.6% | 44 |
| `aten::fill_` / `add` / `mkldnn_convolution` | 3.9 / 2.8 / 2.6% | 51 / 69 / 4 |

`mm` is already an efficient kernel; the rest are scattered small operators (4524 calls) -
reducing the operator count is the only effective direction.

## 2. Matrix efficiency: no headroom

| Shape | GFLOPS |
|---|---|
| Real training size (8192x80x160, i.e. B=32 folded) | 237.9 |
| dim=128 old configuration (8192x128x256) | 274.2 |
| Square 1024^3 fp32 (peak) | 295.8 |

Criterion: `ms/token` barely changes with batch - if small shapes were the drag, a larger batch
should reduce the per-token cost noticeably.

| B | ms/step | ms/token | throughput |
|---|---|---|---|
| 4 | 110.6 | 0.10803 | 9257 tok/s |
| 8 | 185.3 | 0.09049 | 11051 tok/s |
| 16 | 424.3 | 0.10358 | 9654 tok/s |
| 32 | 860.7 | 0.10507 | 9518 tok/s |

=> Not a matrix-efficiency problem, but elementwise/scheduling overhead that grows linearly with
tokens. A larger batch does not save per-token cost.
(B=8 gives 0.0905 ms/token, consistent with the recorded 0.0953.)

## 3. Quantization: a backwards optimization on this machine

| Precision | matmul GFLOPS | Relative to fp32 |
|---|---|---|
| fp32 | 295.8 / 282.9 / 235.7 (n=1024/512/256) | 1x |
| fp16 | 0.593 (n=512) | ~477x slower |
| bf16 | 0.569 (n=1024) | ~520x slower |
| int8 (`torch._int_mm`) | 1.443 (1024^3) | ~205x slower |

Reason: AVX2 has no native FP16/BF16 operations, Coffee Lake has no VNNI; torch takes
emulation/fallback paths.

Numerically it is not viable either: the fp32 relative error of the scan accumulation is 4.7e-07
(still 4.5e-07 at T=2048, chunked recurrence is stable), while bf16 reaches 2.75%.
Quantization would break both convergence and the invariants in `tests/test_architecture.py`.

## 4. Operator fusion: the compiler is in place

| Tool | Detection result | Measured |
|---|---|---|
| MSVC `cl.exe` | present but SDK missing | does exist (14.39.33519), missing Windows SDK headers |
| `clang` / `gcc` / `g++` | not in system PATH | w64devkit (GCC 16.2.0) ships ninja/make/cmake/ar/ld + full mingw-w64 headers, no Windows SDK needed |

Chain: C source -> `gcc -O3 -mavx2 -mfma -shared` -> `.dll` -> `ctypes` call,
forward/backward both numerically consistent with PyTorch autograd.

- **Implemented**: CSN goes through `F.layer_norm` (oneDNN fused kernel, forward 19.5x /
  backward 4.46x); two-level parallel scan; competition gate + gate probability + scan fused
  into a single-pass kernel (§22).
- **Measured results**: scan kernel is 4.3-22x faster than torch (operator level); with the
  fused kernel wired into the model the training step is 1.10-1.17x (§22.3/§22.6).
- **The inference-regime kernel benefit is not significant** (B=1 tensors are too small, ctypes FFI
  is on the same scale as a single operator), so it is off by default.

Note: operator-level microbenchmark advantages get diluted into noise end-to-end (this has
happened twice); microbenchmarks must not be the sole basis for adoption.

## 5. Thread count (undecided)

The existing `_train_bench.py` is fixed at 8 threads. Measured real training step:

| Threads | Median ms/step | Inter-round spread |
|---|---|---|
| 4 | 168.5 | ±15.2% |
| 6 | 193.4 | ±8.4% |
| 8 | 183.3 | ±15.6% |
| 12 | 225.4 | ±7.7% |

12 threads is clearly the worst (+23%) - the opposite of the pure-matmul conclusion (square
matmul actually peaks at 394 GFLOPS with 12 threads).
The training step is dominated by small tensors; more threads only add synchronization overhead.
4 threads is nominally 8.8% faster than 8, but that falls inside the ±15% inter-round spread,
not enough for a verdict.
If the default thread count is changed, every existing comparison dataset must be re-measured.

## 6. Conclusion

CPU training speed is close to the structural ceiling of this machine's architecture. The
remaining headroom is not in matrices, not in quantization, but in the 4524 small-operator
scheduling calls of the backward pass. Now that a compiler is available, that path has been
carried as far as the fused kernel; the measured training-step gain is 1.10-1.17x, with
diminishing marginal returns.

For an order-of-magnitude change the bottleneck is hardware: an AVX2-only 2018 mobile CPU
supports neither low-precision acceleration nor quantization acceleration - on modern servers
(AVX-512/VNNI/AMX) the conclusion for those two would be different.
