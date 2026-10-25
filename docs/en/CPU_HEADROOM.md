*[中文版](../cn/CPU_HEADROOM.md)*
# Remaining headroom for CPU training speed

Machine: i7-8750H (Coffee Lake, 6C12T, AVX2, no AVX-512 / no VNNI / no AMX)
torch 2.14.1+cpu, MKL 2026.1 + oneDNN v3.12 enabled, `capability=AVX2`.
Numbers come from local measurements under `experiments/`, each cell in an independent subprocess.

## Verdict on four directions

| Direction | Verdict | Basis |
|---|---|---|
| Operator fusion | Feasible, already landed | w64devkit (GCC 16.2.0) available, hand-written AVX2+FMA kernels compile and are verified (§21/§22); training step speedup 1.10–1.17× |
| Matrix efficiency | Essentially no headroom | GEMM at real sizes already reaches 238 GFLOPS (80% of the 296 peak); `ms/token` is insensitive to batch |
| SIMD | Feasible, see "operator fusion" | Same as above, the compiler is available |
| Quantization | Reverse: only slower | fp16/bf16 matmul 0.57–0.59 GFLOPS vs fp32 296 GFLOPS (about 500× slower); INT8 1.44 GFLOPS (about 200× slower) |

## 1. Training step composition (T=256/B=8/dim=80/L=4, 8 threads)

```
forward 49.2 ms (24.8%)  |  CE 14.5 ms (7.3%)  |  backward 113.1 ms (56.9%)  |  full step 198.7 ms
aten operator calls: 4524 per step
```

Backward accounts for 56.9%, the largest single block; forward is under 1/4. Top hotspots (self CPU share):

| Operator | Share | Call count |
|---|---|---|
| `aten::mm` | 35.2% | 63 |
| `_log_softmax_backward_data` | 7.1% | 1 |
| `aten::mul` | 6.6% | 232 |
| `_log_softmax` | 6.0% | 1 |
| `aten::cat` | 4.6% | 44 |
| `aten::fill_` / `add` / `mkldnn_convolution` | 3.9 / 2.8 / 2.6% | 51 / 69 / 4 |

`mm` is already an efficient kernel; the rest are scattered small operators (4524 calls) — reducing operator count is the only effective direction.

## 2. Matrix efficiency: no headroom

| Shape | GFLOPS |
|---|---|
| Real training size (8192×80×160, i.e. B=32 folded) | 237.9 |
| dim=128 old config (8192×128×256) | 274.2 |
| Square 1024³ fp32 (peak) | 295.8 |

Criterion: `ms/token` is nearly constant across batch — if small shapes were the drag, increasing batch should markedly lower per-token cost.

| B | ms/step | ms/token | Throughput |
|---|---|---|---|
| 4 | 110.6 | 0.10803 | 9257 tok/s |
| 8 | 185.3 | 0.09049 | 11051 tok/s |
| 16 | 424.3 | 0.10358 | 9654 tok/s |
| 32 | 860.7 | 0.10507 | 9518 tok/s |

⇒ Not a matrix-efficiency problem, but elementwise/scheduling overhead growing linearly with tokens. Increasing batch does not reduce per-token cost.
(B=8 gives 0.0905 ms/token, consistent with the existing record of 0.0953.)

## 3. Quantization: a reverse optimization on this machine

| Precision | matmul GFLOPS | Relative to fp32 |
|---|---|---|
| fp32 | 295.8 / 282.9 / 235.7 (n=1024/512/256) | 1× |
| fp16 | 0.593 (n=512) | about 477× slower |
| bf16 | 0.569 (n=1024) | about 520× slower |
| int8 (`torch._int_mm`) | 1.443 (1024³) | about 205× slower |

Cause: AVX2 has no native FP16/BF16 arithmetic, Coffee Lake has no VNNI, and torch takes an emulation/fallback path.

Numerically it is also infeasible: the fp32 relative error accumulated over the scan is 4.7e-07 (still 4.5e-07 at T=2048, the chunked recurrence is stable),
whereas bf16 reaches 2.75%. Quantization would break both convergence and the invariants in `tests/test_architecture.py`.

## 4. Operator fusion: the compiler is available

| Tool | Detection result | Measurement |
|---|---|---|
| MSVC `cl.exe` | Present but missing SDK | Indeed present (14.39.33519), missing Windows SDK headers |
| `clang` / `gcc` / `g++` | Not on system PATH | w64devkit (GCC 16.2.0) bundles ninja/make/cmake/ar/ld + complete mingw-w64 headers, no Windows SDK needed |

Chain: C source → `gcc -O3 -mavx2 -mfma -shared` → `.dll` → `ctypes` calls,
forward and backward both match PyTorch autograd numerically.

- **Landed**: CSN uses `F.layer_norm` (oneDNN fused kernel, forward 19.5×/backward 4.46×);
  the scan is two-level parallel; competition gating + gate probability + scan are fused into a single-pass kernel (§22).
- **Measured results**: the scan kernel is 4.3–22× faster than torch (operator level); after the fused kernel is wired into the model,
  the training step is 1.10–1.17× (§22.3/§22.6).
- **Inference-mode kernel gains are not significant** (B=1 tensors are too small, ctypes FFI is on the same order as a single operator), off by default.

Note: advantages from operator-level microbenchmarks are diluted into noise end to end (this has happened twice),
and microbenchmarks must not be used alone as grounds for adoption.

## 5. Thread count (not settled)

The existing `_train_bench.py` fixes 8 threads. Measured on real training steps:

| Threads | Median ms/step | Run-to-run spread |
|---|---|---|
| 4 | 168.5 | ±15.2% |
| 6 | 193.4 | ±8.4% |
| 8 | 183.3 | ±15.6% |
| 12 | 225.4 | ±7.7% |

12 threads is clearly worst (+23%) — the opposite of the pure-matmul conclusion (for square matmul, 12 threads is instead the highest at 394 GFLOPS).
Training steps are dominated by small tensors, and more threads only add synchronization overhead.
4 threads is nominally 8.8% faster than 8, but that falls within the ±15% run-to-run spread and is not enough to decide.
If the thread default changes, all existing comparison data must be re-measured.

## 6. Conclusion

CPU training speed is already near the structural ceiling of this machine's architecture. The remaining headroom is not in matrix work,
not in quantization, but in the 4524 small-operator dispatches of the backward pass. With the compiler available, that path has been
taken through to fused kernels, measured at 1.10–1.17× on the training step, with diminishing marginal returns.

For an order-of-magnitude change, the bottleneck is hardware: an AVX2-only 2018 mobile CPU supports neither low-precision acceleration
nor quantization acceleration — both of these conclusions differ on modern servers (AVX-512/VNNI/AMX).