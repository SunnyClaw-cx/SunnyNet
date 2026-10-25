*[中文版](../cn/CPU_WAY_OUT.md)*

# The Way Out for CPU: Verdicts on Four Candidate Routes

`CPU_NATIVE.md` established "what a CPU-native architecture should satisfy". This document answers its next question:
**to actually do training and inference on CPU, which route is viable, which is blocked, and where the boundary lies.**

Each route is given as mechanism, measurement, verdict. All numbers are measured on this machine (i7-8750H / AVX2 / torch 2.14.1+cpu;
GTX 1060 6GB / torch 2.7.1+cu118), each cell in a separate process, performance experiments run exclusively.

Verdict first (TL;DR):

| # | Route | Verdict | Basis |
|---|---|---|---|
| 1 | **Sparse / conditional compute** | **Viable, and the only structurally advantaged region for CPU** | B=1 decode CPU dividend 2.5–3.2×, GPU from E8 on instead 0.64–0.91× |
| 2 | Cache residency (compress working set into L2/L3) | **Blocked** (does not materialize in the training step) | Working set 1–8 MB flat throughout, no L3 cliff |
| 3 | Operator decomposition / small GEMM | **Blocked** (CPU does not tolerate decomposition either) | Decomposition loss CPU 1.30× vs GPU 1.07× |
| 4 | B=1 long context | **Viable** (proven to beat GPU-Transformer) | Crossover T≈2048, at T=8192 faster by 2.97× |

Two more are **falsified**: `poly_sigmoid` is not faster than `torch.sigmoid` (see `CPU_NATIVE.md` §5);
"CPU branches / gather are cheap" does not hold (see §3.1 of this document).

---

## 1. Route One: Sparse / Conditional Compute (Viable)

### 1.1 Phenomenon

The same workload (d=1024, hidden=4096), dense MLP and E-expert MoE (1 expert per token,
effective FLOPs = 1/E). The **absolute time ratio** of CPU / GPU:

| Sparsity | 2048 token | 8192 token | CPU sparse dividend | GPU sparse dividend |
|---|---|---|---|---|
| Dense | 11.63× | 13.15× | 1.0× | 1.0× |
| E2 | 9.65× | 11.13× | 1.9× | 1.8× |
| E4 | 8.53× | 10.60× | 3.5× | 3.1× |
| E8 | 6.03× | 8.59× | 7.5× | 4.9× |
| E16 | 3.01× | 6.71× | 11.5× | 5.9× |
| E32 | **1.27×** | **3.88×** | **14.5×** | 4.3× |

Two readings:

- **Sparsity compresses the CPU / GPU gap from 13.15× to 3.88×**; at 2048 token, E32 compresses it to **1.27×**.
  CPU and GPU times are almost even.
- The sparse dividend CPU receives is 3.4× that of GPU (E32 @ 8192 token: 14.5× vs 4.3×).

### 1.2 Mechanism: not "branches are cheap" but "the relative efficiency of small GEMM"

The initial hypothesis was "CPU has out-of-order execution and cheap branches, so gather is cheap". **Falsified by measurement**:

| Primitive | CPU | GPU | CPU/GPU ratio |
|---|---|---|---|
| Dense GEMM 1024² | 12.63 ms | 1.47 ms | **8.6×** |
| Column gather | 0.709 ms | 0.039 ms | 18.2× |
| Boolean mask | 1.737 ms | 0.456 ms | 3.8× |
| `index_add` | 0.875 ms | 0.123 ms | 7.1× |
| `topk` | 12.73 ms | 3.97 ms | 3.2× |

The CPU/GPU ratio of control-flow primitives (gather 18.2×) is **higher** than that of dense GEMM (8.6×) — gather on CPU is relatively
**more expensive**, not cheaper. The second hypothesis, "CPU tolerates operator decomposition better", is likewise falsified:

| Split into K small GEMMs | CPU loss | GPU loss |
|---|---|---|
| K=1 | 1.000× | 1.000× |
| K=8 | 1.054× | 0.801× |
| K=32 | **1.302×** | **1.074×** |

The finer the split, the **larger the loss on CPU** (1.30× vs 1.07×).

The real mechanism lies in the **GEMM efficiency curve against row count M** (N=4096, K=1024, i.e. the weight block of a single expert):

| M (tokens routed to that expert) | CPU GFLOPS | CPU% of peak | GPU GFLOPS | GPU% of peak |
|---|---|---|---|---|
| 16 | 77.3 | **31.4%** | 204.2 | **6.8%** |
| 32 | 118.8 | 48.2% | 531.2 | 17.6% |
| 64 | 153.0 | 62.1% | 923.8 | 30.5% |
| 128 | 227.3 | 92.3% | 709.1 | 23.4% |
| 1024 | 225.6 | 91.6% | 1732.6 | 57.3% |
| 4096 | 246.2 | 100% | 3024.2 | 100% |

- CPU from M=16 to 4096: 77.3 → 246.2 GFLOPS, **3.2×**;
- GPU over the same interval: 204.2 → 3024.2 GFLOPS, **14.8×**.

**Sparsity produces "many and small" GEMMs. The relative efficiency of CPU at small M is 4.65× that of GPU**
(M=16: 31.4% / 6.8%), so CPU realizes more dividend. This is not "branches are cheap";
it is that CPU **degrades slowly** when there are not enough rows to fill the SIMD width.

### 1.3 Placement: where sparsity belongs

The sparse dividend is realized to different degrees in different scenarios (forward, d=1024):

| Scenario | token count | CPU dividend E8 | CPU dividend E16 | GPU dividend E8 | GPU dividend E16 |
|---|---|---|---|---|---|
| **B=1 decode** | 1 | **2.53×** | **3.19×** | **0.91×** (slower) | **0.64×** (slower) |
| Prefill 1k | 1024 | 4.51× | 5.92× | 4.15× | 2.67× |
| Prefill 8k | 8192 | 6.84× | 9.14× | 5.08× | 6.31× |
| Training B8×T256 | 2048 | 5.48× | 7.71× | 3.57× | 2.57× |

**B=1 decode is the decisive region**: CPU has a 2.5–3.2× dividend, **GPU from E8 on instead gets slower** (0.91×, 0.64×).
In single-sequence decode the launch overhead of each expert kernel cannot be amortized on GPU, whereas on CPU it can.

### 1.4 End-to-end verdict: not penetrated (recorded as is)

**An operator-level dividend is not an end-to-end dividend** (the third precedent of this kind in this repository). A real training loop
(d=256 / L=4 / T=256 / B=8 / 40 steps, same parameter count, same token stream):

| Structure | Params | ms/token | loss |
|---|---|---|---|
| Dense MLP | 4,199,936 | **0.4362** | **5.71** |
| MoE (E=8) | 4,208,128 | 0.4932 | 6.72 |

End-to-end **0.884× (13% slower)**, and quality is worse. The cause is that the indexing and
`index_add` overhead of the per-expert loop is amplified in the backward pass.

⇒ **Verdict**: the sparse route is **viable, but its placement is the pure forward path** (B=1 decode, prefill),
not training backward. To enter training, the routing overhead must first be eliminated from the eager loop (fused grouped GEMM /
custom kernel), otherwise it does not materialize end-to-end.

---

## 2. Route Two: Cache Residency (Blocked)

`CPU_NATIVE.md` §3 measured a single-thread streaming bandwidth cliff at 3–4 MB (26.7 → 15.4 GB/s),
from which "compressing the working set into cache" was conjectured to be a lever. **It does not materialize in the training step**:

| Config | Activation estimate | ms/token |
|---|---|---|
| B8 × T64 | 0.98 MB | 0.1223 |
| B8 × T128 | 1.97 MB | 0.2202 |
| B8 × T256 | 3.93 MB | 0.1298 |
| B8 × T512 | 7.86 MB | 0.1318 |
| B4 × T512 | 3.93 MB | 0.1229 |
| B2 × T1024 | 3.93 MB | 0.1206 |

The working set grows from 0.98 MB to 7.86 MB (crossing the cliff point), and ms/token stays **flat at 0.12–0.13 throughout**,
with **no cliff**. The 0.22 at T=128 is a scheduling artifact of chunk alignment, not a cache effect.

⇒ **Verdict**: the bottleneck of the training step is **per-token operator scheduling** (consistent with `CPU_HEADROOM.md` §1:
4524 aten calls/step, backward accounting for 56.9%), not cache residency. The cache argument holds only for **single-thread streaming memory access**
and has no predictive power for a real training step.

---

## 3. Route Three: Operator Decomposition (Blocked)

See the decomposition experiment in §1.2: on CPU, splitting to K=32 loses 1.30×, on GPU only 1.07×.
**CPU does not tolerate small-operator decomposition** — the opposite of the intuition that "CPU suits many small tasks".
(This also explains why fused kernels pay off on CPU: §22 measured 1.10–1.17× on the training step.)

---

## 4. Route Four: B=1 Long Context (Viable)

`CPU_NATIVE.md` §4 already gives that at B=1/T=2048, CPU-SunnyNet beats GPU-Transformer by 1.15×.
This document completes the frontier (same stack params d=256/L=4, both sides use the single-step + KV cache path):

| T | SunnyNet single step | Transformer single step | Transformer/SunnyNet |
|---|---|---|---|
| 128 | 5.10 ms | 3.09 ms | 0.606× |
| 512 | 4.68 ms | 3.67 ms | 0.784× |
| **2048** | 4.96 ms | 5.06 ms | **1.021× (crossover)** |
| 4096 | 6.59 ms | 13.01 ms | 1.976× |
| 8192 | 7.11 ms | 21.13 ms | **2.971×** |

- SunnyNet single step is essentially flat (O(1), 4.6–7.1 ms);
- Transformer rises linearly with T (O(t), 3.1 → 21.1 ms);
- **crossover T ≈ 2048**; beyond it the recurrent architecture leads steadily, and at T=8192 it is 2.97× faster.

⇒ **Verdict**: viable. This is the only region where "CPU beats GPU" has been proven, and it should be the primary target scenario for CPU-native models.

---

## 5. Root Cause of the Quality Problem: State Collapse and Recall Capacity

The output of an external Kaggle run was reported as "incoherent". After verification, two independent causes were distinguished.

### 5.1 The external run's accounting (CodeDoes/SunnyNet-P100M-Kaggle)

- 102,126,662 parameters, 24,000 steps × B8 × T512 = **98.3M token**, val loss 2.4759 / **ppl 11.89**;
- ran on **Kaggle P100 (GPU)**, not CPU;
- budget ratio: 98.3M / 102M = **0.96 token/parameter**, Chinchilla suggests 20 ⇒ **shortfall 20.8×**;
- only the single domain `en_literature` was run, whose training set is only **4.34M token** ⇒ the data was chewed over about **22 times**.

### 5.2 "Incoherence" is state collapse, not random error

Among 40 samples, multiple mutually unrelated inputs produced the same stretch of text:

| Input | Output fragment |
|---|---|
| `Once upon a time,` | and was horning to the days ... |
| `The old house stood` | ne with the carching of the **scape of the visic of the place, of the eyes** |
| `At the edge of the sea,` | ... **scape of the visic of the place, of the eyes** |
| `The captain turned and` | suref a smoke of the **scape of the visic of the place, of the eyes** |

In 8 sampled cases, `"scape of the"` appears 5 times and `"visic of the place"` 4 times.
**Different inputs converging to the same trajectory = attractor collapse**, not per-token sampling noise.

### 5.3 On-machine reproduction: collapse accumulates with depth

`experiments/_state_collapse.py` (small preset, zh_ancient 400 steps) measures the pairwise cosine
similarity of layer-wise states across different inputs:

| Layer | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Pairwise similarity | 0.308 | 0.212 | 0.338 | **0.526** |

From layer 1 to layer 4, states for different inputs become increasingly alike. The random-initialization baseline is only **0.0043**,
ruling out the "measurement noise" explanation. **The deeper the layers, the heavier the collapse** — the 14-layer p100m is far more severe than
the 4-layer one, consistent with the external observation.

### 5.4 Architecture-level ceiling: exact recall capacity

`experiments/_recall_capacity.py`: sequence `k1 v1 ... kn vn kq`, predicting `v_q`
(key-value vocab 64, d=64, 800 steps):

| # pairs n | Recurrent acc | Attention acc |
|---|---|---|
| 2 | **0.0469** | **0.9336** |
| 4 | 0.1211 | 0.6094 |
| 8 | 0.1133 | 0.2578 |
| 16 | 0.0430 | 0.0742 |

Recurrence reaches only **4.7%** at 2 pairs (random 1/64 = 1.6%), attention **93.4%**.
SunnyNet's state is a **per-channel scalar** (small has 128 numbers per layer, `init_state` shape `(1,4,128)`),
so its capacity is inherently limited. By contrast: DeltaNet / RWKV-7 use **per-head matrix-valued state**, an order of magnitude larger in capacity.

⇒ **Verdict**: incoherence has two independent sources — (a) insufficient training budget (shortfall 20.8×);
(b) the **recall capacity ceiling of element-level state**. The former is solved by adding data; the latter requires changing the state form.

---

## 6. Conclusion: The Shape of the CPU Way Out

Combining the verdicts of the four routes, the design target of a CPU-native model converges to:

**`O(N)` recurrent backbone + sparse forward + element-level state upgrade + oriented to B=1 long context**

| Design item | Basis | Status |
|---|---|---|
| O(N) recurrence (no pairwise token interaction) | §4 crossover T≈2048 | Satisfied (SunnyNet) |
| Long-context single sequence as primary target | §4 T=8192 faster by 2.97× | Target to be tightened |
| Forward sparsity (MoE / top-k) | §1.3 B=1 decode CPU 2.5–3.2×, GPU instead slower | To be implemented |
| Training backward **does not** use sparsity | §1.4 end-to-end 0.884× | Discipline |
| **State upgraded from element-level to matrix-valued** | §5.4 recall 4.7% vs 93.4% | **Architecture change pending** |
| Add an "anti-collapse" mechanism within the layer | §5.3 similarity rises to 0.53 with layer | **Architecture change pending** |

The two most urgent (P0):

1. **State form**: element-level → matrix-valued (one d×d submatrix per head). DeltaNet has already verified feasibility;
   the cost on CPU is that state read/write grows from `dim` to `n_head × d_head²`, in exchange for higher recall capacity.
2. **Anti-collapse**: introduce an input path within the layer that does not accumulate with depth (e.g. direct cross-layer injection,
   or a normalization constraint on the state feedback term). The layer-wise similarity of §5.3 is a free diagnostic metric and can be wired directly into tests.

---

## 7. Reproduction

```sh
# Sparse affinity (run once each on CPU / GPU)
python -X utf8 experiments/_sparse_affinity.py --device cpu
python -X utf8 experiments/_sparse_affinity.py --device cuda

# Sparse dividend vs scale
python -X utf8 experiments/_sparse_dividend.py --device cpu

# Mechanism attribution: control-flow primitives / decomposition tolerance / GEMM M curve
python -X utf8 experiments/_route_mechanism.py --device cpu
python -X utf8 experiments/_decomp_tolerance.py --device cpu
python -X utf8 experiments/_gemm_m_curve.py --device cpu

# Landing zone: which scenario realizes the sparse dividend
python -X utf8 experiments/_sparse_where.py --device cpu

# End-to-end verdict
python -X utf8 experiments/_sparse_e2e.py --device cpu --steps 40

# Cache curve
python -X utf8 experiments/_cache_curve.py --preset smalln80

# Decoding frontier
python -X utf8 experiments/_decode_frontier.py --device cpu

# Quality root cause
python -X utf8 experiments/_recall_capacity.py
python -X utf8 experiments/_state_collapse.py --preset small --steps 400
```

Boundary: all conclusions are limited to this machine (an AVX2-only 2018 mobile CPU).
On machines with AVX-512 / VNNI / AMX, both small-M GEMM efficiency and the sparse dividend need re-measurement.