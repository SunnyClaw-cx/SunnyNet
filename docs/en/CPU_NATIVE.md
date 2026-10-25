*[中文版](../cn/CPU_NATIVE.md)*

# Design Principles for CPU-Native Models

This document answers one question: **are mainstream architectures built for GPU? What should a CPU-native architecture look like?**

Conclusion first: yes, and it is quantifiable. But the reason is not the commonly assumed "lack of compute".

All numbers are measured on this machine (i7-8750H / AVX2 / torch 2.14.1+cpu; GTX 1060 6GB / torch 2.7.1+cu118),
scripts in `experiments/_machine_balance.py`, `_cpu_gpu_crossover.py`, `_cache_cliff.py`,
`_device_affinity.py`, `_transcendental_cost.py`.

---

## 1. Machine balance: the two devices are almost identical (falsifies a common claim)

"CPU is memory-bound, GPU is compute-bound" is a common narrative. Measurement does not support it:

| Device | fp32 GEMM | Copy bandwidth | Machine balance |
|---|---|---|---|
| i7-8750H (6C12T, AVX2) | 216.1 GFLOPS | 11.5 GB/s | **18.7 FLOP/byte** |
| GTX 1060 6GB | 2878.2 GFLOPS | 153.1 GB/s | **18.8 FLOP/byte** |

The FLOP/byte of the two is **almost identical**. GPU has 13× the compute and also 13× the bandwidth, so the algebra cancels.
⇒ **The bottleneck is not "which kind of resource" but "whether the problem size can fill that resource".**

---

## 2. The real divergence: problem size

GEMM size sweep (GFLOPS):

| n | CPU | GPU |
|---|---|---|
| 64 | **23.5** | 9.0 |
| 128 | **215.2** | 74.9 |
| 256 | 227.6 | **573.5** |
| 1024 | 258.7 | **1168.1** |
| 4096 | 322.3 | **3032.3** |

**At n ≤ 128, CPU wins** (n=128: 215 vs 75, 2.9×).

B=1 single-sequence GEMV (decode-state workload, µs/token):

| dim | CPU | GPU |
|---|---|---|
| 256 | **19.1** | 47.7 |
| 512 | **18.6** | 47.7 |
| 1024 | 42.6 | **42.4** |
| 2048 | 598.4 | **109.9** |
| 4096 | 2746.5 | **421.9** |

Two facts:
1. **GPU has a launch floor of about 47 µs** (dim 256 and 512 are the same value), independent of small sizes;
2. **for B=1 decode at dim ≤ 512, CPU is about 2.5× faster**.

GPU's strength is spreading large matrices across its parallelism; once the work per step is smaller than the launch overhead, it loses.

---

## 3. Cache cliff

L2 = 1536 KB, L3 = 9216 KB (i7-8750H). Single-thread streaming effective bandwidth against working set:

| Working set | Effective bandwidth |
|---|---|
| 256 KB | 18.8 GB/s |
| 512 KB | 24.6 GB/s |
| 2048 KB | 26.7 GB/s |
| 4096 KB | 15.4 GB/s ← cliff |
| 262144 KB | 11.0 GB/s |

**A working set above about 3–4 MB leaves the high-bandwidth region.** This is the only "local resource" on the CPU side —
accessing data in L3 costs far less than in DRAM.

---

## 4. Device affinity: mainstream architectures are indeed more "GPU-native"

Parameter-aligned at the same count (stack params about 366K), same batch inputs, forward ms/token.
**GPU multiple = CPU time ÷ GPU time**; the larger it is, the more that architecture benefits from GPU:

| Architecture | B,T | CPU | GPU | GPU multiple |
|---|---|---|---|---|
| sunnynet | 1,256 | 0.0545 | 0.0484 | 1.13× |
| sunnynet | 1,2048 | 0.0279 | 0.0095 | 2.94× |
| sunnynet | 8,256 | 0.0366 | 0.0060 | 6.10× |
| sunnynet | 8,1024 | 0.0426 | 0.0030 | 14.20× |
| transformer | 1,256 | 0.0685 | 0.0240 | 2.85× |
| transformer | 1,2048 | 0.3344 | 0.0322 | **10.39×** |
| transformer | 8,256 | 0.0521 | 0.0057 | 9.14× |
| transformer | 8,1024 | 0.2000 | 0.0169 | 11.83× |

Key readings:

- **Long-context single sequence** (B=1,T=2048): Transformer's GPU multiple is **10.39×**,
  SunnyNet only **2.94×**. The gap comes from the parallel shape of O(T²) attention — GPU is extremely good at it,
  CPU extremely bad at it. **This is the measurable form of "built for GPU".**
- **Real crossover**: at B=1/T=2048, **SunnyNet on CPU (0.0279) is faster than Transformer on GPU
  (0.0322) by 1.15×**. This is the only scenario where "16G of memory catches up with an 8G graphics card" is supported by measurement.
- Training step (60 steps measured): on CPU, sunnynet 0.129 vs transformer 0.173 ms/tok (SN 1.34× faster);
  after switching to GPU, 0.024 vs 0.021 (Transformer 1.14× faster).
  **The CPU advantage disappears when switching to GPU** — exactly the definition of "affinity".

---

## 5. A falsified premise

A common argument: "CPU has no hardware sigmoid, so the architecture should switch to a rational approximation to save transcendental functions".
Measurement on this machine does not support it:

| Width | torch.sigmoid | poly_sigmoid | Multiple |
|---|---|---|---|
| 256 | 0.224 ms | 0.375 ms | 0.60× |
| 640 | 0.594 ms | 4.426 ms | 0.13× |
| 2048 | 2.677 ms | 17.369 ms | 0.15× |

`poly_sigmoid` (addition/absolute value/division only) is **1.7–7.7× slower than torch.sigmoid**:
AVX2 indeed has no hardware sigmoid, but torch's vectorized implementation is already good enough,
and `abs` + element-wise division is not cheap on this machine.

⇒ **Activation functions must not be designed on the grounds that "transcendental functions are expensive on CPU".** That premise is false on this machine.

(This conclusion is limited to AVX2 on this machine; machines with AVX-512 or hardware high-precision reciprocal need re-measurement.)

---

## 6. The CPU-native design principles that follow

Each is checkable, with its check method.

| Principle | Basis | Check |
|---|---|---|
| **Compress the working set into L2/L3 (≤ 3 MB)** | §3 cliff | measure layer-wise activation + resident weights |
| **Keep matrix dimensions near the n ≤ 128 CPU sweet spot** | §2 | GEMM size sweep |
| **Single-step latency must be pushed below 47 µs to matter** | §2 GPU launch floor | B=1 single-step timing |
| **Target scenario is B=1, long context, small dimension** | §4 real crossover | head-to-head against the counterpart architecture on GPU |
| **Do not save cost via transcendental functions** | §5 falsified | activation function microbenchmark |
| **Reducing the operator count takes priority over widening matrices** | CPU_HEADROOM §1 (4524 aten calls/step) | training step profiling |

---

## 7. SunnyNet against the current state

| Principle | SunnyNet | Verdict |
|---|---|---|
| O(N), no pairwise token interaction | rolling state + first-order recurrence | Satisfied |
| Avoid giant dense matrices | low-rank decomposition + depthwise separable convolution + tridiagonal filtering | Satisfied |
| Bounded state | convex combination, `\|s_t\| ≤ max(\|s_0\|,\|u_t\|)` | Satisfied |
| Does not store the whole activation | reversible recomputation | Satisfied |
| Working set ≤ 3 MB | not measured | **To verify** |
| Dimension in the n ≤ 128 sweet spot | `proto20mw` d640 is too large | **To verify** |
| Single step < 47 µs | 1.4M ckpt single step 2.49–2.19 ms | **Not met** (B=1 small model) |

SunnyNet already satisfies most structural principles, but **items 6–7 (scale placement) are unverified**.
Past optimization focused on "reducing operators", without systematic calibration against "cache residency" and the "dimension sweet spot".

---

## 8. Next steps (ordered by strength of evidence)

1. **Measure working-set residency**: done, see §9.
2. **Calibrate the dimension sweet spot**: done, see §9 — **no sweet spot found**.
3. **B=1 long-context specialization**: this is the only region where CPU beats GPU, so the target scenario
   is worth tightening explicitly to it, rather than pursuing general-purpose speed.
4. Add operator-level verification of the `p_min` verdict (see the applicability boundary of the §5 conclusion).

Not doing: architectural rewriting on the GPU side (no nvcc/triton on this machine, and not a project goal).

---

## 9. Two follow-up measurements

### 9.1 Working-set residency (`experiments/_working_set.py`)

Even the smallest preset **exceeds the cache-efficient region once activations are counted**:

| Preset | Matrix weights | Single-layer activation + apportioned weights | Verdict |
|---|---|---|---|
| small (d128/L4) | 1.38 MB | 10.35 MB | exceeds efficient region |
| smalln80 (d80/L4) | 1.39 MB | 7.85 MB | exceeds efficient region |
| proto20mw (d640/L6) | 63.81 MB | 68.99 MB | overflows L3 |

The matrix weights themselves (1.38 MB) **fit in L2**, but at `B=8/T=256`, **a single-layer activation (6.9–9 MB) already blows past L3**.
⇒ The bottleneck of the current configuration is the **activation working set**, not the weights. Reducing B/T or block-wise recomputation is the lever for aligning with the §6 principles.

### 9.2 Dimension sweet spot (`experiments/_dim_sweetspot.py`, 300 steps / T=256 / B=8)

| dim | L | Stack params | ms/token | ppl |
|---|---|---|---|---|
| 64 | 4 | 756,628 | **0.0856** | 96.94 |
| 96 | 4 | 1,150,292 | 0.1057 | 86.50 |
| 128 | 4 | 1,677,076 | 0.1410 | 82.17 |
| 192 | 4 | 2,908,820 | 0.2315 | **78.10** |

Time rises **monotonically** with d, ppl falls **monotonically** with d — **there is no non-monotonic sweet spot**.

⇒ **Falsified**: "a CPU-specific optimal dimension exists". The n≤128 GEMM sweet spot of §2 only shows that a **single matrix**
is efficient at that size; it does not mean the whole model should be at that dimension — the parameter budget dominates (larger d means more parameters,
better quality and also slower). Dimension is a pure trade-off, not an extremum that can be optimized.