*[中文版](../cn/EXPERIMENT_REPORT.md)*

# SunnyNet Experiment Report

All data comes from measurements on this machine; no numbers are quoted from papers and no extrapolated estimates are made. Every number has a corresponding raw record in  
`experiments/results/*.json`. Items that were not measured, cannot be measured, or contain confounding factors are  
marked directly in the text, without substituting estimates for measured values.

---

## TL;DR

| Conclusion                                                      | Evidence                                            | Confidence               |
| ------------------------------------------------------- | --------------------------------------------- | ----------------- |
| For Transformer (**forward only**): SunnyNet is faster once T≳384 (at T=4096 it is **22.69×** faster per token) | Process-isolated forward latency measurements (§2.4)                             | High (architectural, inherent)           |
| **Training speed (forward+backward) lags all O(N) peers by 20–52%** (old `dim=128` configuration, T=256→T=4096 with no crossover) | batch=8 training step, process isolation (§13)                            | High (IQR 5–15%)        |
| **After the narrow residual stream + `p.clamp` removal, training speed matches rwkv** (nominally 3.0% faster at T=256, 3.7% slower at T=512), with a steady 16–41% lead over h3/mamba/hyena/linear | 5 independent subprocess rounds + 9 rounds of per-round paired head-to-head (§16)           | Medium (paired test not significant, recorded as statistically tied) |
| **`p.clamp` is provably unreachable**: measured p∈[0.5902,0.99994], the lower bound is 1209× above `p_floor`, forward pass bit-identical before removing clamp | 20-step sampling + fp64 equivalence check (§16.1)                     | High                 |
| **The speed tie holds only for `L=4`**: when the layer count is aligned to rwkv's `L=6`, the same-budget narrow configuration is instead 30–40% slower; the mechanism is localized as **a fixed tax of ~1100 operator dispatches per layer**, avoidable with a "shallow and wide" configuration (§28) | Layer-count alignment (§16.5) + cost law and same-budget family comparison (§28) | High |
| **After the architecture corrections, T=256 quality rises to second among seven architectures** (ppl 93.82→~56.8, exceeding RWKV/Mamba/Hyena, still below H3 53.93) | Residual + CSN correction, 600 steps, multiple seeds (§12)                       | Medium (unconverged / mostly single seed)      |
| Before the correction, **at T=256 it was below same-complexity O(N) peers** (H3's ppl was 0.57× SunnyNet's) | Seven-architecture same-protocol comparison, 600 steps                              | Corrected by §12           |
| Forward speed has **no leading interval** against the peers (the draft's "only for 512<T<2048" contradicts this table's data, and at T=1024 it ranks 5th); among the seven architectures SunnyNet improves with T: 6th→5th→5th→4th→3rd, **never reaching the top** | Seven-architecture forward scaling measurements (§10.6)                             | Medium (IQR 4–26%)      |
| **In training mode (batch=8, with backward) at T=1024 SunnyNet ranks 1st in speed** (0.1319 vs transformer 0.5569) and 2nd in peak memory (1812MB, 43% below transformer) | Seven-architecture training-mode audit, independent subprocess per cell (§17)                        | High                 |
| **Cross-domain (external new datasets) it is not worse than the peers**: Chinese tech news ppl **321.08, 1st of 7** (unk 23% in that domain), English arxiv abstracts **35.58, 2nd of 7**; OOD gap 1.480, tied with hyena for the lowest | External independent collection (arxiv/Xinhua, etc.), 600 steps of training on the original three domains (§18)              | Medium (single seed / unconverged)        |
| **The short-sequence inversion holds only for batch=1 forward**: in training mode (batch=8) the crossover shifts down to T<256; at T=256/384 SunnyNet is 1.5–2.4× faster than TF; at T=128 it is still 4.5–10.4% slower (4.5% for the narrow configuration) | Three configurations × three lengths × 3 independent subprocess rounds (§19)                      | Medium (memory load 84%)        |
| At 1.4M and short sequences, quality is **statistically indistinguishable** from Transformer                     | 1200 steps, single seed, val gap 1.10%, ppl gap 4.84%, **both below seed noise** | Medium (single seed)            |
| **At 20M scale, quality rises to first among five architectures** (ppl 77.90, exceeding rwkv/mamba/h3/hyena), and the speed deficit narrows to 8–37% | Stack-parameter alignment + 200 steps + isolated speed measurement (§14.6)                    | Medium (200 steps, unconverged)        |
| **At 100M scale, quality ranks only second** (ppl 109.47, about 3.6% behind rwkv 105.66) — the 20M "first place" does not carry over | p100mw stack alignment + 100 steps (§14.8)                        | Medium (100 steps, unconverged)        |
| **At 100M, the unified lr=3e-3 makes mamba diverge (step 37) and hyena's gradient explode (gn≈41.5)** | Divergence detection + re-run with a reduced lr (§14.8)                          | High                 |
| ~~At 20M and T=512, it loses on both quality and speed~~                                | The old conclusion came from the pre-correction stack plus unbalanced layer counts, see §14.6 | Corrected               |
| The ablation experiment **cannot** serve as a criterion for component effectiveness                                     | 600 steps, unconverged; switching components off is instead better                             | High                 |
| Early timing data was distorted (no process isolation)                                     | Model construction accumulated within one process inflates readings by up to 2× and the factor varies by architecture (§2.4)                  | High                 |
| **CSN uses the oneDNN fused kernel + a two-level parallel scan**: training step 19–41% faster (more at larger T), with quality improving at the same time (ppl 92.04→84.37) | 600 steps, three seeds + independent subprocess per cell (§14.1)                     | High                 |
| **Under parameter alignment, SunnyNet's layer count far exceeds the peers** (20M: L10 vs 5–7; 100M: L14 vs 7–9), a deviation from its own fairness discipline | meta device counts (§14.2)                         | High                 |
| **Layer reduction with widening (L10→L6, widened dim)**: stack parameters ±0.1%, training step **12.1%** faster, ppl **−10.9%**, memory −14% | 5 independent subprocess rounds + 400-step quality A/B (§14.3–14.4)            | High                 |
| **Upper bound on the gain from writing the scan / competition gating as fused kernels is 8.8%** (replacement experiment); realized gain 1.10–1.17× | Replacement experiment zeroing both cost segments (§14.7) + w64devkit kernel measurements (§22.3) | High |
| **On GPU these two segments account for a smaller share (scan 1.85% / gating 2.55%) → less headroom for fusion on GPU** | CUDA kernel-level profiling (§15.5)                          | High                 |
| **0.5B (scale05b) cannot be instantiated on CPU fp32; GPU forward/SGD run, while AdamW and SGD with momentum both OOM** | Ladder + multi-step probes (§15.3/15.4)                            | High                 |
| **0.5B multi-step loss shows no downward trend** (30 steps, Δ=−0.005, step-to-step range 0.72) → no quality data | Real-corpus SGD probe (§15.4)                              | High                 |
| **Competition gating roll→index_select**: faster in the microbenchmark (0.717×/0.891×), no end-to-end difference → not adopted | Two-platform end-to-end A/B (§15.6/15.7)                          | Medium                 |
| `torch.compile` is **unavailable**: MSVC is present but Windows SDK headers are missing + `reg.exe` is blocked by the sandbox; `torch.jit` segfaults | Local measurements (§13.4)                                    | High                 |
| **Custom kernels cannot be compiled on the GPU side either**: no nvcc / no CUDA toolkit / no triton, cpp_extension lacks setuptools | GPU venv measurements (§15.5)                              | High                 |
| **The inference advantage lies in the architecture, not the kernels**: single-step decoding is flat for SunnyNet (O(1), drift ±5%) and rises with context for Transformer (O(t), 2.49→6.85ms); crossover at T≈512, 2.36× faster at T=4096; no gain from kernel fusion at short context | Independent subprocess per cell, filling T steps first, then measuring single steps (§22.10) | Medium (B=1, single sequence) |
| CPU cache hit rate is **not measurable on this machine**                                      | No PDH Cache counters, only page-fault rate as a proxy                      | —                 |

Core conclusions: fusion + layer reduction with widening make the same architecture both faster and better; after the narrow residual stream + `p.clamp` removal, training speed matches rwkv (statistically tied) and holds a steady lead over h3/mamba/hyena/linear.

- **Quality**: after the §12 corrections, T=256 ppl ranks **second** among seven architectures (57.83, behind only H3 53.93);
  after layer reduction with widening at 20M scale ppl **drops a further 10.9%** (54.82 vs 61.55, 400 steps, unconverged),
  and in the 20M same-protocol comparison it **rises to first among five architectures** (§14.6).
- **Forward speed**: faster than Transformer once T≳384 (22.69× at T=4096, re-verified);
  against peers that are also O(N) there is **no stable leading interval** — it climbs from 6th to 3rd as T grows (§10.6),
  and never reaches the top (hyena is always first).
- **Training-mode memory**: at T=1024 peak RSS is **1812 MB**, the **2nd-lowest tier** among seven architectures,
  **43%** below Transformer (3192 MB), in the same tier as the O(N) peers (§17).
- **Training speed**: operator fusion gives 19–41% on **the same architecture**; layer reduction with widening is **12–17% faster again**;
  the **narrow residual stream (dim 128→80) + `p.clamp` removal (§16)** is **17.7–20.7% faster again** at equal stack parameters —
  this brings the training step at T=256 **to first place** (0.0953 ms/tok, nominally 3.0% faster than rwkv),
  at T=512 the gap to rwkv is 3.7% and the per-round paired test is not significant ⇒ recorded as "statistically tied with rwkv" per discipline;
  it holds a steady 16–41% lead over h3/mamba/hyena/linear_attn. rwkv leads only nominally, and "surpassing" it was **not achieved**.
  The measured upper bound on the gain from fused kernels is only 8.8% (§14.7), a lower priority than the changes already landed.
- **The boundary of the tie**: it holds only for `L=4`; when the layer count is aligned to `L=6`, the same-budget narrow configuration is 30–40% slower (§16.5).
  The mechanism is in §28: a fixed operator-dispatch tax per layer, and at the same budget "shallow and wide" is better on both the speed and the quality axis.
  And at T≥1024 the peers' advantage remains (§13's conclusion still holds under the old configuration).
- **Scaling**: 20M and 100M run and have been compared; **0.5B cannot be instantiated on CPU fp32, and on GPU only forward and
  the plainest SGD run (AdamW/momentum SGD both OOM), with multi-step loss showing no decrease → no 0.5B quality conclusion**.
- **Positioning of the GPU side**: this machine's GPU is used for **correctness verification and speedup measurement** (forward 8.0–15.2×, training step
  9.9–12.5×, without any CUDA-side optimization, so these are lower bounds). It is **not** used for kernel-level optimization — there is no nvcc / no CUDA
  toolkit / no triton, and on GPU the scan and gating account for under 5%, leaving less headroom for fusion than on CPU (§15.5).
- **The scaling trend is not monotonic**: at 20M quality is first (§14.6), but **at 100M it falls back to second** (§14.8,
  3.6% behind rwkv). The available evidence supports only "quality is competitive around 20M, and at 100M it is in the same tier as rwkv";
  it does not support "the larger the scale, the larger the advantage".

Applicability depends on the target task's sequence length and on the trade-off between quality and speed.

---

## 1. Experimental setup

### 1.1 Hardware and environment

| Item       | Value                                |
| ------- | -------------------------------- |
| CPU     | 8 threads (`torch.set_num_threads(8)`) |
| Memory      | 15.87 GB physical                      |
| Precision      | Native FP32 (no FP16/BF16)             |
| PyTorch | CPU build                            |
| Operating system    | Windows 11                       |

> **Measurement-contamination statement**: during the experiments, two Python processes **unrelated to this project** were running on this machine  
> (`recall_attrib.py`, `minimind/verify_flow.py`); they have no relation to SunnyNet.  
> The concurrent load was relatively stable across the baseline-comparison, ablation, and seed stages; the load was confirmed during the latency-comparison measurements.  
> All timing numbers are affected by this, **absolute values should be treated as upper bounds**, while the **relative comparisons** between architectures still hold  
> (both were measured alternately under the same load). This is a known limitation.

### 1.2 Corpus (really collected, not synthetic)

Sources: Project Gutenberg Chinese classics + English literature, TheAlgorithms/Python source code.  
The tokenizer is a **from-scratch BPE implementation** (vocabulary 8192).

| Domain                       | Training tokens       | Validation tokens      | Training files  | Validation files |
| ----------------------- | -------------- | ------------- | ----- | ---- |
| Chinese classics `zh_ancient`       | 39,987,804     | 2,183,213     | 345   | 22   |
| English literature `en_literature`    | 4,343,482      | 86,836        | 15    | 1    |
| Python code `code_python` | 1,254,856      | 79,005        | 1,267 | 80   |
| **Total**                  | **45,586,142** | **2,349,054** | 1,627 | 103  |

- The validation set is split at the **file level** (`val_ratio=0.06`), so adjacent fragments of the same text never straddle the training/validation boundary,
  ruling out information leakage.
- **Missing domain**: Chinese Wikipedia is unreachable from this machine's network, so there is **no "encyclopedia" domain** and only 3 real domains are covered.
  Compared with the five domains required by the specification ("encyclopedia/novel/news/code/classical Chinese") this is **incomplete**, and it is marked as such here.

### 1.3 Comparison baseline

Standard Transformer (`sunnynet/baseline.py`), with `d_model`/`d_ff` solved by **closed-form formulas**
to match SunnyNet's **compute-stack parameter count** (embedding layer excluded, since SunnyNet has no positional embedding):

| Preset         | SunnyNet total parameters | SunnyNet stack parameters | Transformer stack parameters | Relative deviation       |
| ---------- | ------------ | ------------ | --------------- | ---------- |
| `small`    | 1,413,908    | 365,332      | 361,952         | 0.93%      |
| `proto20m` | 20,955,698   | 16,761,394   | 16,762,160      | **0.005%** |

> The `proto20m` name was corrected once: the early configuration (d=384/r=224/di=640/L=8) measured only 10.35M parameters,
> which did not match the name; it was changed to d=512/r=320/di=896/L=10, measuring 20,955,698 parameters.

### 1.4 Learning-rate calibration

Each architecture was swept over 3 learning-rate levels (260 steps), taking each one's best:

| Learning rate   | SunnyNet val loss | Transformer val loss |
| ----- | ----------------- | -------------------- |
| 0.001 | **4.8858**        | 5.0614               |
| 0.003 | **4.7720** ← best   | 4.7989               |
| 0.006 | 4.7822            | **4.7744** ← best      |

→ SunnyNet uses lr=0.003, Transformer uses lr=0.006. **Each is compared at its own best learning rate**, so the baseline is not disadvantaged.

---

## 2. Core experiment: measured verification of the O(N) advantage

This is the architecture's only claim, and it can only be tested with data. Method: fix the model (`small`),  
vary the input sequence length, and measure the time of a single forward pass.

![Sequence scaling](../figures/seq_scaling.png)

### 2.1 Data

| Sequence length T | SunnyNet | Transformer | Time ratio (TF/Sunny) |
| ------ | -------- | ----------- | -------------- |
| 256    | 45.6 ms  | 17.5 ms     | 0.38×          |
| 512    | 80.3 ms  | 62.5 ms     | 0.78×          |
| 1024   | 147.7 ms | 167.0 ms    | **1.13×**      |
| 2048   | 215.8 ms | 542.1 ms    | **2.51×**      |
| 4096   | 319.9 ms | 4041.8 ms   | **12.63×**     |

### 2.2 Conclusion

1. **The O(N) claim holds**: SunnyNet's time grows approximately linearly with T (45.6→319.9ms; T grows 16× while the time grows 7×,  
   and the slope actually flattens as T grows — the fixed overhead is amortized). Transformer explodes as O(T²) (17.5→4041.8ms; T grows 16× while the time grows 230×).
2. **Crossover at T≈834**: below this sequence length SunnyNet is slower; above it SunnyNet is faster.
3. **Source of the short-sequence disadvantage**: at T=256 Transformer takes only 17.5ms while SunnyNet takes 45.6ms.  
   The reason is that SunnyNet's per-token rolling-state computation involves many element-wise operations (competition gating, scan, recurrent filtering),  
   so its constant factor is higher than a single dense matmul. At short sequences, Transformer's parallel matmul efficiency wins.

> **This result is the key one**: any claim that "SunnyNet is uniformly faster" contradicts this data.  
> The accurate statement is: SunnyNet trades a constant-factor disadvantage at short sequences for a linear-complexity advantage at long sequences.

### 2.3 The crossover point depends on batch size

The latency measurements in §2.1 were **batch=1, forward only**. In **batch=8 training** (§3), at T=256 SunnyNet is instead  
8.4% faster (0.2831 vs 0.3089 ms/tok) — **the opposite of the batch=1 forward conclusion**.

| Measurement condition              | Which is faster at T=256                        |
| ----------------- | --------------------------------- |
| Forward, batch=1        | Transformer (17.5 vs 45.6 ms)      |
| Training (forward+backward), batch=8 | SunnyNet (0.2831 vs 0.3089 ms/tok) |

**Reason**: at batch=1 the matrices are extremely small, and SunnyNet's many element-wise operators are dominated by **PyTorch's per-operator dispatch overhead**;  
as batch grows, matmul efficiency improves, while Transformer's attention computation grows linearly with batch,  
which **moves the crossover point toward smaller T**.

→ **Conclusion correction**: the crossover **T≈834 is only the value for batch=1 forward** and is not a universal constant.  
It varies with batch, with whether backward is included, and with the thread count. Decisions should be based on measurements at the target task's real batch/length combination,  
not by applying a single number directly.

### 2.4 Measurement-discipline correction (re-measured after process isolation; values differ from the original §2.1)

**Finding**: the absolute timings in §2.1 are not reproducible. When the same model is measured in the same process "after several models have been constructed",
the time can be **more than 2×** that of a clean process (measured at T=4096: 325 ms vs 776–822 ms).
The reason is that heap fragmentation and allocator state degrade as model constructions accumulate within a process, and **the inflation factor differs between architectures**.
This distorts all inter-architecture time ratios in §2.1 and in earlier versions.

**Correction procedure**: each `(architecture, sequence length)` cell is started in its own subprocess, and only one model is constructed within that process.
Script `experiments/on_scaling.py`. Re-measured results (batch=1, forward, 8 threads, 9 repetitions, median):

| Sequence length T | SunnyNet | Transformer | Time ratio (TF/Sunny) | IQR |
|---|---|---|---|---|
| 128  | 14.0 ms | 7.5 ms | 0.53× | 19% / 26% |
| 256  | 19.5 ms | 13.5 ms | 0.69× | 3.5% / 15% |
| 512  | 26.4 ms | 40.8 ms | **1.55×** | 12% / 6% |
| 768  | 34.6 ms | 87.3 ms | **2.53×** | 9% / 3% |
| 1024 | 50.0 ms | 141.2 ms | **2.83×** | 6% / 7% |
| 1536 | 58.0 ms | 315.4 ms | **5.44×** | 13% / 4% |
| 2048 | 98.1 ms | 638.2 ms | **6.51×** | 7% / 6% |
| 4096 | 148.7 ms | 3373.4 ms | **22.69×** | 4% / 3% |

**Differences from §2.1 and their explanation**:

| Item | §2.1 (contaminated) | §2.4 (process isolation) | Note |
|---|---|---|---|
| T=256 time ratio | 0.38× | 0.69× | Contamination inflates SunnyNet more |
| Crossover | T≈834 | **T≈384** (6-layer TF) / **T≈600** (4-layer TF) | The crossover moves with the depth of the comparison model |
| T=4096 time ratio | 12.63× | **22.69×** | Contamination compressed the real gap |

**Reason for keeping the original values**: the **directional conclusions** of §2.1 and §2.3 still hold (slower at short sequences, faster at long sequences,
the crossover moves with batch/depth), but their **absolute values and the 12.63× should be treated as lower bounds**.
When both the README's "12.6×" and this table's "22.7×" are put on the process-isolation convention, the latter is closer to the true value;
but both hold only under the conditions under which each was measured, and **are not citable once detached from those conditions**.

**Re-verification of 22.69× (2026-10-20, re-measured after the fusion changes)**:

| Item | Value |
|---|---|
| Original data | `experiments/results/seq_scaling.json`: at T=4096, 148.66 ms vs 3373.42 ms = **22.69×**, IQR 3.94% / 2.69% |
| Re-verification data | `experiments/results/verify_22x.json`: 131.9 ms vs 3709.9 ms = **28.12×** (current post-fusion code), IQR 10.6% / 14.6% |
| Conditions (must be cited together with the number) | **batch=1, forward only (no backward), 8 threads, 1.4M small model** |

⇒ **The number is real and reproducible**, and after the fusion changes the ratio rises rather than falls (22.69 → 28.12, because fusion speeds up SunnyNet's
forward pass more than Transformer's). But it is a forward-latency ratio, not a training-speed ratio —
for a batch=8 training step it still lags the O(N) peers (§13).

**Another note on measurement conventions**: §2.4's `22.69×` and §10.6's "forward leads the peers only for 512<T<2048"
**do not contradict each other**. The former benchmarks against a scalar-attention Transformer (O(T²), where the gap is amplified as T² at T=4096),
while the latter benchmarks against linear-class architectures that are also O(N) (where the gap is determined by constant factors, and larger T works against SunnyNet).

---

## 3. Quality comparison: can it match?

Speed is not the only metric. Quality is compared under the same token budget (1200 steps × batch 8 × T 256 ≈ 2.45M tokens).

![Baseline comparison](../figures/speed_compare.png)

![Training curves](../figures/loss_curves.png)

| Metric                 | SunnyNet      | Transformer | Gap              |
| ------------------ | ------------- | ----------- | --------------- |
| Validation loss               | 4.3367        | **4.2895**  | +1.10%          |
| Validation perplexity              | 76.46         | **72.93**   | +4.84%          |
| Per-token time (T=256) | **0.2831 ms** | 0.3089 ms   | SunnyNet 8.4% faster |
| Peak memory               | 896 MB        | 879 MB      | +1.9%           |
| Mean gradient norm             | 0.8445        | 0.4838      | —               |

**Reading**:

- At small scale and short sequences, quality is close but **slightly worse**: val loss 1.10% higher, ppl 4.84% higher.
- **But this 4.84% gap cannot establish that Transformer is better**: under the same configuration the ppl range across seeds is 6.32 (§6),
  which is **greater than** the 3.53 between the two architectures. The baseline comparison ran only a single seed, so the difference falls within the noise level.  
  The strict conclusion is: the two are statistically indistinguishable at small scale and short sequences, and a multi-seed comparison is needed to settle it (see the improvement items in §8).
- On speed, "SunnyNet 8.4% faster" was recorded at the time, **and that number is void**: it came from the pre-correction stack and without process isolation (§2.4).
  Under the process-isolated re-measurement of §13, at T=256 the training step has SunnyNet **1.10× ahead of Transformer** (0.1858 vs 0.2041 ms/token),
  the direction is the same but the magnitude should follow §13; **against mamba/h3/hyena it lags at all lengths**.

> **Convention correction**: §3's "0.2831 ms per token" and §13's 0.1858 ms are not comparable —
> the former is an unisolated measurement (several models already constructed in the same process), the latter an independent subprocess.
> **Under the same configuration, unisolated measurement inflates the reading by about 50%**, which is the same origin as the 2×-level inflation recorded in §2.4.

---

## 4. 20M-scale comparison

### 4.1 Single-step training cost (measured, batch 4, T=512)

| Architecture          | Single-step time        | Per-token time   | Total parameters        |
| ----------- | ----------- | ------------ | ---------- |
| SunnyNet    | 3.755 s     | 1.834 ms     | 20,955,698 |
| Transformer | **2.200 s** | **1.074 ms** | 22,086,960 |

> **Warning**: at 20M and T=512, SunnyNet is **71% slower per step** (1.834 vs 1.074 ms/tok).  
> There are two reasons: (a) T=512 is still **below the crossover point of 834**; (b) at 20M scale SunnyNet's per-token element-wise operators  
> (there is still a dense d→inner multiply after the low-rank projection) have their constant-factor disadvantage amplified.
>
> This means: SunnyNet's applicable range is determined by sequence length, not by model scale.  
> Scaling up the model does not let it overtake on short sequences.

### 4.2 20M training-quality comparison (300 steps)

> **Note**: 300 steps is far from convergence for a 20M model (val is still falling fast: 5.73→5.55→5.49),  
> so the values here serve only to confirm "training proceeds normally, gradients do not explode, memory stays under control",  
> and **do not serve as a quality-comparison conclusion**.

| Architecture          | Total parameters        | Stack parameters        | val loss   | ppl        | ms/token  | Peak memory    | Mean gradient norm | wall   |
| ----------- | ---------- | ---------- | ---------- | ---------- | --------- | ------- | ------ | ------ |
| SunnyNet    | 20,955,698 | 16,761,394 | 5.4883     | 241.84     | 3.1642    | 2579 MB | 1.846  | 1987 s |
| Transformer | 22,086,960 | 16,762,160 | **5.0955** | **163.29** | **2.281** | 1689 MB | 0.779  | 1430 s |

**At 20M and T=512, SunnyNet loses on both quality and speed**: ppl is **48.1%** higher (241.84 vs 163.29),  
each token is **38.7%** slower (3.164 vs 2.281 ms), and memory is **52.7%** higher (2579 vs 1689 MB).  
This is a clearly unfavorable result.

**Cause analysis**:

1. T=512 is **below the crossover point of 834**, so SunnyNet's O(N) advantage has not yet unfolded and its constant-factor disadvantage is fully exposed.
2. SunnyNet's per-token operator chain is long (normalization → convolution → low-rank projection → competition gating → scan → output feedback),  
   and at 20M dimensions the dispatch and memory-access cost of each operator is amplified; Transformer, by contrast, is a few large matrix multiplies.
3. The higher memory (2579MB) comes from the scan needing to keep the whole state sequence `(B,T,d)`, whereas Transformer's activations are  
   released layer by layer by PyTorch.

**Conclusion**: SunnyNet is a long-sequence-specific solution, not a more resource-efficient general replacement.  
On short sequences, scaling up the model scale only amplifies its disadvantage.

**Training-health reading** (SunnyNet):

- The gradient norm fluctuates between 0.84–1.93, **neither exploding nor vanishing**, which is acceptable at 10 layers of depth.
- Peak memory 2579 MB, **no paging risk** (this machine has 15.87 GB).
- val loss decreases monotonically (5.7325 → 5.5545 → 5.4883), training is stable.

**Measurement-contamination note**: SunnyNet's ms/token was recorded three times as 2.527 → 2.477 → **3.164**,  
rising markedly in the last segment, which is caused by **intensified CPU contention during the measurement**, not by the model slowing down.

---

## 5. Ablation experiment

![Ablation](../figures/ablation.png)

600 steps, `small` preset, components switched off one at a time:

| Variant            | val loss   | ppl       | ms/token |
| ------------- | ---------- | --------- | -------- |
| full (complete)      | 4.5681     | 96.36     | 0.4528   |
| no_decay_gate | **4.5539** | **95.01** | 0.3231   |
| no_circulant  | 4.5663     | 96.19     | 0.3259   |
| no_local_taps | 4.5722     | 96.76     | 0.2605   |
| no_current    | 4.5531     | 94.92     | 0.4878   |
| none (all off)      | 4.6058     | 100.06    | 0.2063   |

**Three problems with the ablation experiment**:

- Switching off a single component gives ppl that is **instead lower** (e.g. no_current 94.92 < full 96.36),  
  which in principle cannot happen for a sufficiently trained model.
- The reason is that **600 steps is far from convergence** (see the training curves; loss is still falling at 1200 steps).  
  Before convergence, a smaller model capacity "overfits more slowly", and the ppl reading is dominated by noise.
- Therefore this ablation experiment provides no evidence for the effectiveness of any component; it only shows that "all off (none) is indeed the worst" (100.06),
  i.e. these components **do contribute computation** (though the direction is undetermined).
- Ablations should be run on a sufficiently converged checkpoint, or at least at several times the current step count.  
  This is an **improvement item**, not a usable conclusion.

### 5.1 Valid readings on the speed side

The ablation's speed data is trustworthy (it does not depend on convergence):

| Variant            | ms/token | Relative to full   |
| ------------- | -------- | --------- |
| full          | 0.4528   | 1.00×     |
| no_local_taps | 0.2605   | **0.58×** |
| no_current    | 0.4878   | 1.08×     |
| no_decay_gate | 0.3231   | 0.71×     |
| none          | 0.2063   | 0.46×     |

→ The depthwise separable convolution (local_taps) and the protection gate (decay_gate) are the **main sources of cost**,  
accounting for about 42% of full. This gives a clear direction for subsequent optimization.

---

## 6. Seed stability

600 steps, 3 random seeds:

| Seed             | val loss            | ppl          |
| -------------- | ------------------- | ------------ |
| 1234           | 4.5681              | 96.36        |
| 2024           | 4.5984              | 99.32        |
| 3407           | 4.6316              | 102.68       |
| **Mean ± sample standard deviation** | **4.5994 ± 0.0318** | 99.45 ± 3.16 |

→ The coefficient of variation is about **0.69%**, the fluctuation across seeds is small, and the results are **reproducible**.  
The ppl range across different seeds (96.36→102.68, **a difference of 6.32**) is **greater than** the ppl difference between the two architectures (76.46 vs 72.93, **a difference of 3.53**),  
which shows that a comparison conclusion from a single run is fragile and that multi-seed repetition is necessary. The baseline comparison ran only  
a single seed (seed=1234), and its 4.84% ppl gap **falls within the seed-noise level**, so the quality difference between the two architectures at small scale  
**should be judged "statistically indistinguishable"**, rather than Transformer being better.

---

## 7. Long-sequence training comparison (direct verification of the crossover point)

§2's crossover conclusion was derived from **forward latency**. To verify that it also holds in **real training**,  
a training comparison was run at T=1024 (**above the crossover point of 834**) (200 steps, batch 4, `small` preset).

![Long-sequence training comparison](../figures/longseq_train.png)

| Metric         | SunnyNet      | Transformer | SunnyNet advantage |
| ---------- | ------------- | ----------- | ----------- |
| Per-token time | **0.2623 ms** | 0.5445 ms   | **2.08× faster** |
| Validation loss       | **5.1320**    | 5.3702      | 4.4% better      |
| Validation perplexity      | **169.35**    | 214.92      | **21.2% better** |
| Peak memory       | **1269 MB**   | 2008 MB     | **36.8% saved** |
| Stack parameters        | 365,332       | 361,952     | Aligned (0.9% apart)  |

**This is the most important positive result**: at T=1024, SunnyNet is **simultaneously** better than an equally-parameterized Transformer  
on all three dimensions of speed, quality, and memory. The three advantages are **consistent in direction and sizable in magnitude**,  
not noise (the 21.2% ppl advantage is far larger than the ~6% seed noise).

**Consistency with §2's crossover prediction**: forward latency predicted SunnyNet 1.13× faster at T=1024,  
while the measured training (with backward) is 2.08× faster. **The direction is consistent and the magnitude is larger** (in backpropagation attention's  
O(T²) cost is heavier than in the forward pass). The crossover conclusion is therefore more credible.

**Comparison with §4**: at T=512 SunnyNet loses on both ( §4), at T=1024 it wins on all three (§7).  
**The conclusions are self-consistent: applicability is determined entirely by sequence length, with the boundary around T≈834.**

---

## 8. Long-distance dependency probe

> **Architecture-version note**: the §8 checkpoint is
> `data/ck_probe_sunnynet.pt`; its state_dict measured to contain only `blocks.N.delta`,
> **not** `theta` / `delta_max` / `delta_min` ⇒ **it comes from the old architecture before the §12 corrections**
> (the current architecture cannot load this ckpt, and reports unexpected/missing keys).
> Therefore the §8 values do not represent the retrieval capability after the corrections (residual bypass + CSN floor + learnable theta),
> and re-measuring requires retraining on the current architecture. The version was not marked before; it is corrected here.

**Probe design**: find **repeatedly occurring token fragments** in the real validation corpus. If a fragment occurred before,  
a model that can really use long-range context should rank the token that followed that fragment previously at top-1.  
Hit rates are bucketed by "distance to the most recent occurrence".

Two key lengths are supported (`--ngram`): `1` = single-token predecessor; `4`/`8` = multi-token fragments.  
The longer the key, the more a hit indicates true long-range retrieval rather than local smoothing.

### 8.1 Discriminative-power self-check

| Model | code_python / ngram=4 / far bucket hit rate |
|---|---|
| **Untrained** (random initialization) | **0.0** |
| After 800 training steps (SunnyNet) | **0.1208** |
| After 800 training steps (Transformer) | **0.1208** |

The untrained model scores 0.0 in all buckets → **the probe does measure learned retrieval ability**, not noise.

### 8.2 Comparison results (budget: 800 steps / T=1024 / batch 4 / `small` preset)

| Key length | Domain | Distance bucket | SunnyNet | Transformer |
|---|---|---|---|---|
| 1 | code_python | near | 0.1796 | 0.1802 |
| 1 | code_python | mid (17–256) | 0.1166 | 0.1184 |
| 1 | code_python | far (257–2048) | 0.0096 | **0.0129** |
| 4 | code_python | near | 0.3961 | 0.3943 |
| 4 | code_python | mid | 0.1143 | 0.1143 |
| 4 | code_python | **far** | **0.1208** | **0.1208** |
| 8 | code_python | near | **0.4578** | **0.4578** |
| 8 | code_python | mid | **0.1093** | **0.1093** |
| 8 | code_python | **far** | **0.1341** | 0.1311 |
| 1 | zh_ancient | near | 0.0166 | 0.0008 |
| 4 | en_literature | far | 0.0 | **0.003** |

### 8.3 Reading

1. Long-range retrieval really exists, and is significant on structured text: in the code domain's far bucket (distance 257–2048)
   the hit rate at ngram=8 is **0.134**, far above random (vocabulary 8192 → about 0.0001 at random).
2. **The longer the key, the higher the hit rate** (near bucket 0.18 → 0.40 → 0.46): the longer the repeated fragment, the more reliable the hit,
   showing that the model is doing fragment-level matching, consistent with the behavioral characteristics of long-range retrieval.
3. **The two architectures perform comparably on repeated patterns**: the ngram=4 mid/far bucket values are **completely identical** (0.1143/0.1208).
   This is consistent in direction with §7's training comparison (SunnyNet's quality slightly worse, 0.6%).
4. **Long-distance hit rates are near 0 in natural-language domains**: consistent with the risk assessment in §9 — the rolling state
   `H_max=8192` only guarantees "not forgotten", not "retrievable". Long-distance dependencies in natural language
   are harder than repeated patterns in code, and this architecture **shows no advantage** here.

**Boundary**: the `ultra` bucket (distance >2048) has 0 samples under all configurations and **no data**, so
§8 draws no conclusion about that interval. The probe runs within a T=1024 window; capability over longer spans was not measured.

---

## 9. Risks and limitations

| Risk            | Status      | Note                                             |
| ------------- | ------- | ---------------------------------------------- |
| **Long-distance dependency capability**   | **Partially verified** | Long-distance hit 0.13 on structured text (code), near 0 on natural language; no samples in the `ultra` bucket (§8) |
| **Pseudo-convergence**       | **Present**  | Falling ppl ≠ reasoning/instruction capability; small-scale experiments cannot be extrapolated to 0.5B               |
| **Ablation conclusions**      | **Invalid**  | 600 steps, unconverged, differences within noise (§5)                            |
| **Short-sequence disadvantage**     | **Confirmed (forward-only convention)** | Batch=1 pure forward is worse than Transformer for T<384 (§2); **does not hold in training mode (batch=8)** — at T=256 it is already faster than Transformer and ties rwkv (§16/§19) |
| **Short sequences behind O(N) peers** | **Partially corrected** | At T=256 ppl is still below H3, but after the §12 corrections it goes from 93.82→57.83 (second among seven); on speed the §16 narrow configuration ties rwkv |
| **Early timing data distorted** | **Corrected** | §2.1 did not use process isolation; absolute values are inflated and the inflation factor differs between architectures, see §2.4 |
| **No absolute advantage over peers at long sequences** | **Confirmed** | At T=4096 Hyena is 1.9× faster and H3 roughly ties; per-cell rank goes 6th→3rd, and **there is no length at which it leads all peers** (§10.6) |
| **T=1024 memory/speed** | **Measured; the doubt does not hold** | Training-mode speed 1st, peak memory 2nd-lowest tier (1812MB, 43% below TF) (§17) |
| **Cross-distribution generalization** | **No worse than peers (measured)** | External new domains: Chinese news 1st/7, English arxiv 2nd/7, OOD gap in the lowest tier (§18) |
| **`torch.compile` unavailable** | **Environment limitation** | MSVC lacks the Windows SDK headers, and Inductor's Windows branch requires an MSVC-style CLI; bypassed with hand-written AVX2 kernels (§21/§22) |
| **20M loses on both**    | **Confirmed** | At T=512 ppl 48.1% higher, 38.7% slower, memory 52.7% higher (§4)      |
| **Multi-distribution imbalance**    | **Untreated** | The three domains are mixed at fixed proportions (0.4/0.35/0.25), with no domain-balanced sampling               |
| **CPU cache hit rate** | **Not measurable** | This machine has no PDH Cache counters; only the page-fault rate serves as a proxy, and there is no measured value               |
| **Incomplete corpus**     | **Confirmed**  | Missing the encyclopedia domain (Wikipedia unreachable), only 3 domains                             |
| **Measurement contamination**      | **Declared** | Non-project processes occupied the CPU during the period; absolute values should be treated as upper bounds                       |
| **0.5B extrapolation**   | **Limited**  | The scale05b preset was corrected to 498.4M parameters; the available memory on this machine is insufficient to instantiate it (§10)     |

**Summary of positive conclusions**:

| Advantage       | Condition                | Data                                                            |
| -------- | ----------------- | ------------------------------------------------------------- |
| Speed       | T=4096, batch=1 forward | 12.63× faster                                                      |
| Speed+quality+memory | T=1024, real training       | 2.08× faster, ppl 21.2% better, memory 36.8% lower                                 |
| Bounded state    | Any T              | The convex combination guarantees `\|s_t\| ≤ max(\|s_{t-1}\|, \|u_t\|)`, provable (§ARCHITECTURE) |
| Training stability    | 20M / 10 layers        | Gradient norm 0.84–1.93, neither exploding nor vanishing (§4)                                       |

---

## 10. Comparison with same-complexity O(N) architectures

> **Why this comparison is necessary**: comparing only against Transformer is not enough to support "surpasses or matches existing public methods" —
> Transformer is O(N²) and is not in the same complexity class as SunnyNet. **The real comparison group is public architectures that are also
> O(N)**: linear attention, RWKV, Mamba, Hyena, H3.

### 10.1 Comparison implementations

Hand-written implementations in `sunnynet/linear_baselines.py` (no third-party libraries, so the parameter count stays controllable):

| Architecture | Source paradigm | Core mechanism |
|---|---|---|
| `linear_attn` | Linear attention (Katharopoulos 2020 form) | Feature map `φ(x)=elu(x)+1`, causal accumulation `KᵀV`, no softmax |
| `rwkv` | RWKV-4 | Time-mixing wkv (implemented with a closed-form scan) + channel mixing `relu²` |
| `mamba` | Mamba / S6 | Input-dependent selective gating + causal depthwise convolution + SiLU gate |
| `hyena` | Hyena | FFT long convolution ×3 layers + multiplicative gating + depthwise short convolution |
| `h3` | H3-2 | Diagonal SSM (closed-form scan) + short convolution + multiplicative gating |

### 10.2 Fairness guarantees

1. **Parameter-count alignment**: the target is SunnyNet's compute-stack parameter count (embedding excluded, since SunnyNet has no positional embedding).
   Alignment uses meta device instantiation counts, not hand-derived formulas —
   measured hand-derived formulas miss terms badly (RWKV's deviation reaches **34%**).
2. **Layer-count constraint**: each architecture's layer count differs from the baseline by no more than ±1 (±2 when necessary),
   avoiding padding the parameter count with "extremely deep layers" or "an extremely wide single layer".
3. **Same protocol**: same corpus mixture (0.4/0.35/0.25), same token budget, same step count, same thread count, same validation set.

Alignment results:

| Architecture | `small` stack parameters | Deviation | `proto20m` stack parameters | Deviation |
|---|---|---|---|---|
| `linear_attn` | 374,112 | 2.40% | 16,862,528 | 0.60% |
| `rwkv` | 371,712 | 1.75% | 22,644,224 | 1.38% |
| `mamba` | 367,136 | 0.49% | 13,742,528 | 0.36% |
| `hyena` | 369,488 | 1.14% | 16,851,304 | 0.54% |
| `h3` | 354,640 | 2.93% | 16,862,256 | 0.60% |
| (SunnyNet target) | 365,844 | — | 16,761,394 | — |

> Note: SunnyNet's stack parameter count changed from 365,332 to 365,844 (+512) because of the two §12 corrections —
> the decay parameter changed from a frozen buffer to a learnable `theta` (+128 parameters per layer), and the residual connection introduces no parameters.
> The peers' aligned values were not recomputed accordingly; the deviations stay within ±3% and do not affect the conclusions.

### 10.3 Experimental results

> The values in the table below have been updated by §12. What is shown here are the raw readings **before** the corrections, to show the magnitude of the correction.
> After the corrections SunnyNet's ppl drops from 93.82 to **57.83** and its rank rises from second-to-last to second.
> The full post-correction table is in §12.5.

Protocol: `small` preset, T=256, batch=8, 8 threads, 600 steps, warmup=60,
cosine decay to 0.1×, the same corpus mixture and validation set, the same seed (1234).
Unified lr=3e-3 (Transformer keeps its best 6e-3, see the learning-rate calibration in §3).

| Architecture | Compute-stack parameters | Deviation | Best val | ppl | ms/token | Peak memory | wall |
|---|---|---|---|---|---|---|---|
| `sunnynet` | 365,332 | — | 4.5414 | 93.82 | 0.3277 | 876 MB | 428 s |
| `transformer` | 361,952 | 0.93% | 4.5018 | 90.18 | 0.3394 | 925 MB | 444 s |
| `linear_attn` | 374,112 | 2.40% | 4.5208 | 91.91 | 0.3360 | 800 MB | 437 s |
| `rwkv` | 371,712 | 1.75% | 4.1605 | 64.10 | 0.3046 | 766 MB | 398 s |
| `mamba` | 367,136 | 0.49% | 4.1485 | 63.34 | 0.4481 | 867 MB | 577 s |
| `hyena` | 369,488 | 1.14% | 4.4127 | 82.50 | 0.2520 | 838 MB | 333 s |
| `h3` | 354,640 | 2.93% | 3.9877 | 53.93 | 0.2793 | 781 MB | 370 s |

Speed normalization (relative to SunnyNet; >1 means slower than SunnyNet):

| Architecture | Relative ms/token | Relative ppl |
|---|---|---|
| `sunnynet` | 1.00× | 1.00× |
| `transformer` | 1.04× | 0.96× |
| `linear_attn` | 1.03× | 0.98× |
| `rwkv` | 0.93× | **0.68×** |
| `mamba` | 1.37× | **0.68×** |
| `hyena` | 0.77× | 0.88× |
| `h3` | 0.85× | **0.57×** |

### 10.4 Reading (negative conclusion)

1. **At T=256 SunnyNet is not better than same-complexity peers**. RWKV / Mamba / H3 all have
   lower ppl than SunnyNet, with H3 best (ppl 53.93, **0.57×** SunnyNet's).
   This is consistent with the short-sequence disadvantage in §2 — T=256 is far below the O(N) crossover (T≈834),
   exactly the interval where SunnyNet's constant factor is most penalized.
2. **The speed advantage also fails to materialize**: SunnyNet's ms/token of 0.3277 is higher than Hyena's (0.2520),
   H3's (0.2793), and RWKV's (0.3046). At T=256, "O(N) is faster" does not hold for SunnyNet,
   because the scan's operator-dispatch overhead is a large share at small T (§6 measured: backward accounts for 74%,
   `aten::mul` 338 times/step).
3. It is roughly on par with Transformer: ppl 93.82 vs 90.18 (4.0% apart),
   ms/token 0.3277 vs 0.3394 (SunnyNet 3.4% faster). At T=256 the O(N²)
   attention is still small in scale, so the complexity-class difference does not show.

### 10.5 Three limitations

1. **Unconverged, so it cannot serve as a final criterion**. 600 steps ≈ 1.23M tokens, cosine has already annealed to 0.1×,
   while both train loss and val loss are higher than the pre-annealing plateau, and **none of the seven architectures converged**.
   By this project's consistent standard (§5 declares the ablation invalid because it is unconverged), §10.3 can only be read as
   a cross-sectional ranking of **early sample efficiency**, not as a final quality ranking.
2. **The learning rate was not tuned per architecture**. Except for Transformer, lr was a unified 3e-3, with no hyperparameter search
   for architectures sensitive to it such as Mamba / H3. **Mamba usually needs a lower lr**, so it may be underestimated here;
   conversely, it cannot be ruled out that the other architectures have better settings that were not searched out.
3. **Single seed**. §7 measured a ppl range of 6.32 across seeds, larger than part of the inter-architecture gaps here
   (e.g. SunnyNet 93.82 vs Transformer 90.18, a gap of 3.64). **Small inter-architecture gaps are not trustworthy**;
   only H3's 40-point ppl gap over SunnyNet exceeds the noise range.

**Direction of the conclusion**: this comparison **falsifies** the claim that "SunnyNet surpasses public same-complexity methods on short sequences";
its only demonstrable advantage interval remains long sequences (the constant amortization at T≥834 and the 12.63× at T=4096).
Whether these five peers benefit equally at long sequences is **not measured**, and is listed as follow-up work (§10.6).

### 10.6 Speed scaling: all seven architectures measured at long sequences

§10.3 measured only T=256. Here the scaling of each architecture's ms/token with T is added,
answering "does SunnyNet's long-sequence advantage also apply to the peers".
Protocol: batch=1 forward, 8 threads, an independent process per cell (§2.4 discipline), 9 repetitions, median.
Script `experiments/on_scaling.py`, data `experiments/results/on_scaling_t8.json`.

![Seven-architecture scaling](../figures/on_scaling.png)

ms/token (lower is better):

| Architecture | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| `sunnynet` | 0.0707 | 0.0508 | 0.0406 | 0.0389 | 0.0352 |
| `transformer` | 0.0565 | 0.0742 | 0.1515 | 0.2922 | 0.8206 |
| `linear_attn` | 0.0720 | 0.0563 | 0.0553 | 0.0616 | 0.0644 |
| `rwkv` | 0.0506 | 0.0468 | 0.0363 | 0.0434 | 0.0475 |
| `mamba` | 0.0575 | 0.0348 | 0.0320 | 0.0313 | 0.0423 |
| `hyena` | 0.0326 | 0.0215 | 0.0164 | 0.0179 | 0.0184 |
| `h3` | 0.0439 | 0.0388 | 0.0342 | 0.0269 | 0.0343 |

Ratios relative to SunnyNet (>1 = slower than SunnyNet):

| Architecture | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| `transformer` | 0.80 | 1.46 | 3.73 | 7.51 | **23.34** |
| `linear_attn` | 1.02 | 1.11 | 1.36 | 1.58 | 1.83 |
| `rwkv` | 0.72 | 0.92 | 0.89 | 1.12 | 1.35 |
| `mamba` | 0.81 | 0.69 | 0.79 | 0.81 | 1.20 |
| `hyena` | 0.46 | 0.42 | 0.40 | 0.46 | **0.52** |
| `h3` | 0.62 | 0.76 | 0.84 | 0.69 | 0.98 |

**Reading**:

1. Transformer's O(T²) holds and is cleanly reproduced: ms/token rises from 0.0565 to 0.8206
   (T grows 16×, per-token cost grows 14.5×), and at T=4096 it is **23.3×** slower than SunnyNet.
2. **SunnyNet's advantage interval against same-complexity peers is not 512<T<2048**
   (the draft once wrote "leads all peers for 512<T<2048", which contradicts this table's data; corrected here).
   Ranking per cell by the raw ms/token in `experiments/results/on_scaling_t8.json`, SunnyNet
   places among the seven architectures: **6th at T=256, 5th at T=512, 5th at T=1024, 4th at T=2048, 3rd at T=4096**.
   The rank improves monotonically with T, but **it never takes first place at any length** (hyena is always first).
   At T=1024 it is slower than four peers — Hyena/Mamba/H3/RWKV; at T=2048 it is slower than Hyena/H3/Mamba.
3. **At T=4096 it is still not closed**: Hyena is still clearly faster (0.52×) and H3 roughly ties (0.98×).
   That is, "the longer the sequence, the bigger SunnyNet's advantage" is not absolute — at the 4096 tier,
   there are still 2 peers that are not slower than it.
4. **Hyena is faster at all lengths** (0.40–0.52×). Its FFT long convolution has an extremely low constant factor,
   and SunnyNet's element-wise scan never catches up with it on this curve.

**Conclusion**: only one clear, provable claim holds for SunnyNet's forward latency —
**it is better than Transformer for T≥256** (O(N) vs O(N²), more pronounced at larger T).
Against peers that are also O(N), it is **a chaser that improves with T, not a consistent winner**:
6th at T=256, 3rd at T=4096, never reaching the top. The available data does not support "leads in full at long sequences",
nor "leads all peers for 512<T<2048" — the latter directly contradicts this table's data and is void.

> **Convention difference**: this table is **batch=1 pure forward**. The ranking in training mode (batch=8, with backward + optimizer)
> differs, see §17.

### 10.7 Follow-up work

- Per-architecture lr search (at least two extra groups of 1e-3/2e-3 for Mamba).
- Multiple seeds (≥3) to confirm whether H3's lead is stable.
- Fill in intermediate lengths outside 512 < T < 2048 to pin down each peer's crossover point relative to SunnyNet.
- Verify whether the conclusions hold with multi-threading turned off in `torch.set_num_threads` (operator dispatch on this machine is sensitive to thread count).

---

## 12. Architecture corrections

§10's negative conclusion drove a round of **structural corrections**. Three measurable defects were A/B tested one by one; two were confirmed
and one was falsified. All under 600 steps / T=256 / batch=8 / lr=3e-3 / warmup=60, with an
independent subprocess per cell and multiple seeds.

### 12.1 Correction one: residual bypass (the largest single gain)

The original architecture excluded the residual, and the block output $o_t$ completely replaced the input $x_t$. Measurement showed this was **a misjudgment**:

| Configuration | seed 1234 | seed 2024 | seed 3407 |
|---|---|---|---|
| No residual (old default) | 93.82 | — | — |
| **With residual (current default)** | **67.40** | **70.72** | — |
| H3-2 reference | 53.93 | | |

The residual alone gives −28.2% ppl. Mechanism: the residual provides an identity path that does not pass through the state relay, so deep layers do not have to carry information solely
on the state bus, and the under-scaling collapse no longer appears.

### 12.2 Correction two: CenteredScaleNorm floor ($s_{\min}$)

The original $s_{\min}=0.25$. Measurements showed the norm input std of each layer is as low as $0.06$ and the embedding std is only $0.02$,
**all below $0.25$** — so all inputs took the "near-constant branch", the denominator was constantly $0.25$,
and the output was just a $4\times$ constant scaling of the input, meaning **normalization never actually happened**.

Sweep (with residual, mean of 2 seeds):

| $s_{\min}$ | 0.250(old) | 0.100 | 0.050 | 0.025 | 0.015 | 0.010 | 0.005 |
|---|---|---|---|---|---|---|---|
| Mean ppl | 69.06 | 61.17 | 58.35 | 57.51 | 57.42 | 57.01 | **56.79** |

The decrease is monotonic, and a plateau appears over `[0.01, 0.025]` (differences enter seed noise). **Finalized $s_{\min}=0.02$**
(inside the plateau; old value→new value is about −16%, while retaining constant protection several times better than LayerNorm, see §12.4).
`experiments/results/smin_sweep.json`, `experiments/_cell_ab.py`.

### 12.3 Correction three (falsified): learnable decay

The peers (Mamba/RWKV/H3) all have learnable decay, while SunnyNet's was a frozen buffer, which was assumed to be a disadvantage.
After converting it to a learnable `theta`, the measurement gives **99.83 / 98.65**, **worse than the frozen 93.82**, so the hypothesis is falsified.
The learnable parameter is kept (needed for fairness alignment with the peers), but the gain is negative.

### 12.4 Head-to-head: post-correction CSN vs LayerNorm

After the CSN correction, a head-to-head re-measurement against LayerNorm (both with residual, 3 seeds):

| Normalization | seed 1234 | seed 2024 | seed 3407 | Mean |
|---|---|---|---|---|
| CenteredScaleNorm ($s_{\min}=0.02$) | ~57.1 | ~56.6 | — | ~56.8 |
| **LayerNorm (comparison)** | **54.90** | **54.58** | **56.58** | **55.35** |
| H3-2 reference | | | | 53.93 |

**LN is still slightly better on quality (2.6% lower)**, but the gap has narrowed from ~12 ppl before the correction to ~1.5 ppl,
and falls within the seed-noise level (range 6.32). CSN's independent value is not in quality but in **constant-input stability**:

| Input | CSN($s_{\min}=0.02$) | LayerNorm |
|---|---|---|
| Strictly constant $3.14159$ | max$\vert y\vert = 0$ | max$\vert y\vert = 0$ |
| Near-constant ($5.0 + 10^{-7}$ noise) | std $=1.0\text{e-}6$ | std $=1.6\text{e-}5$ |

CSN suppresses near-constant noise amplification **15×** more strongly than LN — this is the direct countermeasure to the architectural property that
"the rolling state easily tends toward a constant along some dimensions", and it is the core reason CSN is kept.

### 12.5 Post-correction ranking (seven architectures re-measured under the same protocol)

After the corrections the seven architectures were rerun under a **completely identical protocol** (`experiments/battery5.sh`,
data `experiments/results/oncompare_*.json`); rank changes:

| Architecture | Compute-stack parameters | Pre-correction ppl | **Post-correction ppl** | Current rank |
|---|---|---|---|---|
| `h3` | 354,640 | 53.93 | 53.93 | 1 |
| **`sunnynet` (post-correction)** | **365,844** | 93.82 | **57.83** | **2** |
| `mamba` | 367,136 | 63.34 | 63.34 | 3 |
| `rwkv` | 371,712 | 64.10 | 64.10 | 4 |
| `hyena` | 369,488 | 82.50 | 82.50 | 5 |
| `transformer` | 369,200 | 90.18 | 90.05 | 6 |
| `linear_attn` | 374,112 | 91.91 | 91.91 | 7 |

The values of the other six of the seven architectures are word-for-word identical to the pre-correction values (they do not use CSN / residual),
proving that this change **affects only SunnyNet** and that the comparison's fairness is intact; `transformer`'s 90.18→90.05
is normal run-to-run jitter.

**Conclusion**: from "second-to-last" to **second**; the relative ppl drops from 93.82 to 57.83 (**−38.4%**),
surpassing Mamba / RWKV / Hyena / Transformer / linear_attn,
but **still not exceeding H3** (57.83 vs 53.93, about 7% apart).

As in §10.5, this comparison is **600 steps, unconverged, single seed, with no per-architecture lr tuning**,
so it can only be read as an early sample-efficiency ranking, not as a final quality ranking.

On the speed side: after the corrections SunnyNet's ms/token rises from 0.3277 to **0.3960** (the residual brings extra computation),
and it is still faster than most peers other than Transformer (0.3619 is the value from this re-measurement, with jitter between batches) —
**this timing column's cross-architecture comparison is affected by batch load and is unreliable on its own; see the process-isolated measurement method in §10.6**.

### 12.6 Measurement discipline for the correction round

- Every training cell uses an **independent subprocess** (`_cell_ab.py`), avoiding the timing/memory distortion of accumulated construction within one process.
- The gradient matrix (`_grad_matrix.py`) is a pure forward-backward, deterministic measurement, not subject to load contamination.
- The `s_min250` cell reproduced the previously recorded 67.3976 / 70.7208, confirming the experiment is reproducible.

---

## 13. Training-speed focus

**Question**: §2's "crossover point" and "22.69×" are both **batch=1 forward**, whereas CPU training cares about **training speed**.
The two orderings are completely different and cannot be cited interchangeably. Here the **training step (forward+backward+update)** is measured.

Protocol: `small` preset, batch=8, 8 threads, **an independent subprocess per cell** (§2.4 discipline),
5 repetitions, median, `experiments/on_scaling.py --backward`.
Data `experiments/results/on_train_speed.json`, `train_longseq.json`.

### 13.1 Results (ms/token, lower is better)

| Architecture | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| `sunnynet` | 0.1858 | 0.1699 | 0.1935 | 0.2027 | 0.2327 |
| `transformer` | 0.2041 | 0.2540 | 0.5396 | 1.0792 | — |
| `mamba` | **0.1192** | **0.1078** | **0.1157** | **0.1723** | **0.1866** |
| `h3` | **0.1273** | **0.1122** | **0.1291** | **0.1496** | **0.1658** |
| `hyena` | **0.1459** | **0.0942** | **0.1162** | **0.1205** | **0.1123** |
| `rwkv` | 0.1704 | 0.1048 | 0.1187 | 0.1990 | — |
| `linear_attn` | 0.2566 | 0.1149 | 0.1320 | 0.1867 | — |

### 13.2 Conclusion: training speed lags all O(N) peers

- **It leads Transformer 1.10× only at T=256**; everywhere else it lags.
- Against same-complexity peers: **mamba 20–56% faster, h3 24–29% faster, hyena 27–52% faster**,
  and this disadvantage **persists from T=256 through T=4096 with no crossover point**.
- **The previously reported "at T=256 SunnyNet is 8.4% faster in training" (§3) is void**: that was a pre-correction-stack measurement without process isolation.

> This is a clearly unfavorable result. The confirmed quality advantage (ppl second) and the
> speed disadvantage (peers 20–50% faster) hold at the same time — the current architecture trades quality for speed, it is not "fast and good at once".
>
> **Follow-up (§16, same round)**: §13's conclusion **holds only for the `small` (dim=128) configuration**.
> After the two equivalent changes of a **narrow residual stream (dim 128→80) + removing `p.clamp` at construction time**,
> the training step ties rwkv at T=256 (0.0953 vs 0.0982 ms/tok, nominally 3.0% faster) and is 3.7% slower at T=512,
> with a steady 16–41% lead over h3/mamba/hyena/linear. **"Lags all peers" still holds under the old configuration,
> but is no longer the current architecture's best reading**; at the same time the peers' advantage at T≥1024 is **not eliminated**, see §16.5.

### 13.3 Root cause (each item measured and ruled out; none is a viable improvement direction)

| Hypothesis | Measurement | Verdict |
|---|---|---|
| More computation | MAC/token: SunnyNet **8064** ≈ mamba 7029, far below Transformer 55224 | **No** |
| Too many chunks in the chunked scan | `chunk=64` is already optimal; enlarging the chunk length is instead slower (the original cumsum reduces along the stride=N dimension) | **No** |
| Thread count not tuned | Crossover measurement at 4/6/8 threads: 1483 / 1510 / 1518 ms, differences within noise | **No** |
| Operator rewriting could save time | `_compete` slicing, `_feedback` inlining, the CSN two-step var difference all < IQR | **No** |

**Actual root causes**:
1. **Operator count** (not FLOPs). Each SunnyNet forward makes 63 batched `aten::mm` calls,
   plus **small-tensor element-wise operators** at mul 762 / add_ 376 / fill_ 341 times/step;
   the short-sequence bottleneck is operator dispatch and memory access, not computation.
2. **The intra-block width is minimal**. SunnyNet uses `dim=128 / inner=192`, while the peers at equal parameters
   have `d_model` of 88–112 but **a wider inner layer** (e.g. hyena d_ff=224), giving higher matmul efficiency.
3. **Backward accounts for 64.6% of the training step** (forward 480 ms vs training step 1357 ms, T=1024 B=8),
   while the scan accounts for only 12–13% of the forward pass — **optimizing the scan is nearly useless for training speed**.

### 13.4 Acceleration means available on this machine

| Means | Status |
|---|---|
| `torch.compile` | **Unavailable** — MSVC actually exists (BuildTools 14.39.33519), but `Windows Kits\10` lacks `Include`/`Lib`, and `cl /nologo t.c` (including `stdio.h`) reports `cannot open 'stdio.h'`; also `vcvars64.bat` depends on `reg.exe`, which is blocked by the sandbox. Fusion must find another path (§21 uses w64devkit with hand-written AVX2 kernels) |
| `torch.jit.script` | **Segmentation fault** (segfault), unavailable |
| triton / numba / cpp_extension | None installed |
| `torch.associative_scan` | Not present in this torch build |

### 13.5 Optimizations rejected by measurement (do not repeat)

**Intra-block transposed scan** (transpose the cumsum to the last dimension, reducing along stride 1):
- In an isolated test at dimension `N=512` the scan is 3.4× faster, which looks like a large gain;
- **but at the real block dimension `dim=128` it is instead 26.2% slower** (12.11 → 15.29 ms);
- The end-to-end training step is **net flat** (the −7% forward is offset by +4% backward).
- It has been reverted, and `sunnynet/ops.py` keeps the original version. **Scan benchmarks must use the real block dimension (dim=128),
  not the large dimension of an isolated test (N=512) — the two give opposite conclusions.**

**Competition gating `roll` → `index_select` (re-measured on both GPU and CPU, §15.6/15.7)**:
- Pure gating microbenchmark: index_select is faster (cuda 0.717×, cpu 0.891×), numerically exactly equivalent (maxdiff=0);
- End-to-end training step: no stable difference on either platform (cuda 5.8%, within the round-to-round range; cpu 0.2%);
- It has been reverted, and `_compete` in `sunnynet/layers.py` keeps `roll`.
- The same type of lesson recurs: an operator-level microbenchmark advantage gets diluted into noise end-to-end and cannot alone justify adoption.

### 13.6 Optimization directions

1. **Raise `h_min` (2.0 → 8/16)**: relax the `_safe_chunk` chunk-length cap from 64 → 332/687,
   reducing the number of Python loop iterations. **But the measured scan time does not drop** (chunk count is not the bottleneck),
   and the change shifts the decay spectrum → a quality regression must be run in step. **Gain uncertain.**
2. **Reduce the operator count within a block**: requires operator fusion. `torch.compile` is unavailable, but w64devkit is available,
   so hand-written AVX2 kernels are used instead (§21/§22).
3. **Widen the intra-block dimension** (`dim` 128 → 160/192) to raise matmul efficiency:
   it would break the current ±1% parameter alignment and require redoing all comparisons → costly, but the **most direct direction**.
4. **Reduce the backward cost**: the 64.6% backward share comes from the autograd graph, and only operator-level fusion can improve it.

> **§14 implements two of these**: direction 2 is landed with the ready-made oneDNN kernels, and direction 3's "widening dim"
> is implemented without breaking parameter alignment (see §14.3, layer reduction with widening).

---

## 14. Operator fusion and layer reduction with widening

### 14.1 The only viable fusion path: off-the-shelf oneDNN kernels

`torch.compile` / Inductor are unavailable (§13.4 was corrected to "MSVC is present but the SDK headers are missing").
The fusion that is genuinely available on CPU is **oneDNN's native fused operators** — `F.layer_norm` fuses
mean/var/sub/mul/add into a single kernel.

- `CenteredScaleNorm` at `c = 0` is **strictly equivalent** to gamma-scaled LayerNorm
  (`gamma/center` and LN's `weight/bias` play the same role). After switching to `F.layer_norm`:
  - Forward 4.535 ms → **0.232 ms (19.5×)**; backward 11.075 → 2.485 ms (4.46×)
- **Two-level parallel scan** `gated_scan`: changes "a serial per-chunk Python loop of ceil(T/chunk) iterations" into
  "a single in-chunk cumsum covering all chunks + a serial nblk-step inter-chunk pass", **bit-identical** in both fp32/fp64.

End-to-end training step (`experiments/_speed_ab.py`, an independent subprocess per cell, median over multiple rounds):

| T | Speedup |
|---|---|
| 256 | −19.3% |
| 512 | −24.4% |
| 1024 | −18.9% |
| 2048 | −29.4% |
| **4096** | **−41.2%** |

**Quality improves at the same time** (600 steps, three seeds, `_quality_ab.py`):

| Implementation | seed1234 | seed2024 | seed3407 | Mean ppl |
|---|---|---|---|---|
| Old (hand-written CSN + serial scan) | 92.96 | 90.46 | 92.71 | 92.04 |
| **New (fused LN + parallel scan)** | 84.36 | 83.81 | 84.95 | **84.37** |

⇒ **Quality and speed improve together**, with all 43→47 tests green.

### 14.2 Key finding: under parameter alignment SunnyNet's layer count far exceeds the peers

meta device measurements (real counts, not hand-derived formulas):

| Scale | SunnyNet | mamba | h3 | hyena | rwkv | linear_attn |
|---|---|---|---|---|---|---|
| proto20m (stack 16.77M) | **L=10** | L=5 | L=5 | L=7 | L=6 | L=6 |
| p100m (stack 93.74M) | **L=14** | L=7 | L=8 | L=9 | L=7 | L=7 |

- This **deviates from the project's own fairness discipline** ("layer count within ±1 of the baseline, ±2 when necessary"), exceeding it by +3~+8 layers.
- Cause: low rank (`rank/dim ≈ 0.6`) + narrow `dim` → few parameters per layer → forced to stack deeper under parameter alignment.
- Consequence: more layers = **the per-layer operator set × more repetitions** ⇒ this points directly at root cause ① "operator count" in §13.

### 14.3 Layer reduction with widening (under stack-parameter alignment)

With **stack-parameter deviation of ±0.1%**, the layer count is brought back to the peers' tier while widening `dim`:

| Preset | Configuration | Stack parameters | T=256 ms/tok | T=512 ms/tok | Peak RSS |
|---|---|---|---|---|---|
| proto20m | d512 i896 r320 **L10** | 16,766,514 | 1.262(median of 5 rounds) | 1.330 | 2253 / 3817 MB |
| **proto20mw** | d640 i1280 r384 **L6** | 16,755,230(+0.07%) | **1.110(−12.1%)** | **1.108(−16.7%)** | **1935 / 3240 MB** |
| p100m | d1024 i1792 r640 **L14** | 93,738,054 | 4.957 | OOM | 5802 MB |
| **p100mw** | d1408 i1920 r1024 **L8** | 93,828,904(+0.10%) | **4.158(−16.1%)** | 3.969 | **4897 MB** |

- Tool `experiments/_rebalance_probe.py` (an independent subprocess per cell); presets `proto20mw` / `p100mw` have been added to `train/train.py`.
- A reverse-order re-measurement rules out an order effect: **proto20mw is faster in 5/5 rounds**.
- The 12–17% speedup matches the "40% fewer layers" magnitude ⇒ the root cause is operator count (layer count), not FLOPs;
  it also brings the layer count back to the peers' tier, **fixing the fairness defect**.

### 14.4 Quality verification: layer reduction with widening is also a win on both axes (20M / T=256 / 400 steps / real corpus)

| Preset | L | dim | Stack parameters | 100-step ppl | 200 | 300 | 400(best) | Wall clock |
|---|---|---|---|---|---|---|---|---|
| proto20m | 10 | 512 | 16,766,514 | 121.56 | 88.47 | 69.13 | 61.55 | 1133.7 s |
| **proto20mw** | 6 | 640 | 16,755,230 | **112.75** | **78.79** | **60.94** | **54.82** | 1144.3 s |

- **ppl leads throughout, −10.9% at the last checkpoint** (54.82 vs 61.55).
- **400 steps is far from convergence** (both configurations' ppl is still falling monotonically), so this difference should be read only as
  **a same-budget sample-efficiency ranking**, not as a final quality conclusion.
- The end-to-end wall clock is nearly identical (1133.7 vs 1144.3 s), which contradicts the isolated single-step probe
  (−12.1%). The per-checkpoint data shows that both configurations' wall-clock increment per 100 steps is **increasing**,
  a **contamination feature of external load growing over time** (this machine permanently runs WorkBuddy with 58142 s of accumulated CPU,
  SunloginClient, HipsTray, 3×DownloadManager). **Speed conclusions should follow the isolated-subprocess measurements.**

### 14.5 Hard scaling constraints

| Preset | Stack parameters | Peak instantiation | T=512 training step |
|---|---|---|---|
| small | 0.37M | 737 MB | Passes |
| proto20m | 16.77M | 2241 MB | Passes |
| p100m | 93.74M | 5799 MB | **Killed by OOM** |
| p100mw | 93.83M | 4897 MB | Passes |
| **scale05b** | **481.65M** | **Instantiation OOM** | **Cannot run** |

- **0.5B is impossible on this machine**: training-mode weights + AdamW state need >7 GB of available memory,
  while commit memory is often occupied up to 20 GB+ by other processes (limit 27.87 GB). **No 0.5B results.**
- p100m OOMs at T=512, whereas the rebalanced p100mw (more memory-efficient) passes — an incidental benefit of layer reduction with widening.

### 14.6 Same-protocol comparison at 20M scale (quality rises to first)

Protocol: driven by the `proto20mw` preset, stack parameters aligned at 16.76–16.86M, T=256, batch=8,
200 steps, real corpus, `experiments/scale_compare_short.sh`.

**Quality (ppl, lower is better)**

| Architecture | Stack parameters | **ppl** | Rank |
|---|---|---|---|
| **sunnynet** | 16,755,230 | **77.90** | **1** |
| rwkv | 16,844,768 | 79.12 | 2 |
| mamba | 16,763,712 | 81.92 | 3 |
| h3 | 16,862,256 | 86.99 | 4 |
| hyena | 16,801,776 | 109.07 | 5 |

Speed (ms/token, each cell re-measured in an independent subprocess, interleaved, `_scale_speed_probe.py`)

| Architecture | ms/token | Relative to SunnyNet |
|---|---|---|
| rwkv | **0.729** | 0.63× |
| mamba | 0.922 | 0.80× |
| h3 | 0.945 | 0.82× |
| hyena | 1.066 | 0.93× |
| sunnynet | 1.152 | 1.00× |

Conclusion (opposite in direction to the small scale):

- **On quality SunnyNet rises to first at 20M scale** — at small scale (1.4M) it is "second among seven architectures",
  while at 20M it is first among five. **The speed disadvantage narrows**: at small scale the peers are 20–52% faster,
  while at 20M scale it narrows to **8–37%** (hyena only 7% faster). The direction is right, but **it still does not overtake**.
- **Attribution**: this change **cannot be attributed entirely to "scale"**. The 20M run uses the layer-reduced, widened
  `proto20mw` (§14.3), while the small run uses the un-rebalanced `small` preset.
  The rise in quality ranking is the joint effect of **layer reduction with widening + scale**;
  also, the two comparisons' architecture sets (seven vs five) and step counts (600 vs 200) differ, so they **cannot be subtracted directly**.
- **200 steps is far from convergence**; the `mamba` cell's in-loop reading (2.057 ms/tok)
  is clearly contaminated by external load (wall clock 872 s vs 370–510 s for the others, with ms/token degrading checkpoint by checkpoint),
  so **speed should always follow the isolated re-measurement**.

### 14.7 Upper bound on the gain from writing the core scan / competition gating as fused kernels

**Question**: would writing the core scan and competition gating as fused kernels be significantly faster?

**Method**: `experiments/_fuse_ceiling.py` replaces each of these two segments with a minimal implementation (not numerically equivalent),
and measures the end-to-end training step. This gives the **gain ceiling** from "zeroing both segments' cost", not a landable rewrite.
small preset, T=1024, B=8, an independent subprocess per cell, two interleaved rounds, median:

| Variant | ms/token | Relative to baseline |
|---|---|---|
| base (current) | 0.1575 | — |
| Competition gating cost zeroed | 0.1465 | −7.0% |
| Scan cost zeroed | 0.1476 | −6.3% |
| **Both zeroed at once (upper bound)** | **0.1436** | **−8.8%** |

**Conclusion**:

1. **The replacement experiment gives a gain upper bound of 8.8%** (zeroing both segments' cost directly; not numerically equivalent, measuring only the ceiling).
2. **The landed measurements in §21/§22 exceed this upper bound**: after w64devkit got the AVX2 kernels working, the fused kernels
   measure 1.10–1.17× on the training step (§22.3/§22.6). The upper-bound estimate is low because that experiment was run under
   the `small/T=1024` convention, whereas the landed measurement is `smalln80/T=256–512` (a higher scan share).
3. **Single-block profiling explains where the shares come from**: `_compete` accounts for 14.4% of a single block and `_scan` 17.2% (forward),
   while **backward is about 65% of the training step** (§13.3), so however high the forward share is, it cannot be spread into the large denominator.
4. **Practical judgment**: the measured gain from fused kernels (1.10–1.17×) is lower than that of layer reduction with widening (12–17%),
   but the two can be stacked.

---

### 14.8 Same-protocol comparison at 100M scale (some architectures diverge at high lr)

`p100mw` preset (dim=1408/inner=1920/rank=1024/L=8, stack 93,828,904),
T=256, B=8, 100 steps, unified lr=3e-3, same corpus, peers aligned by stack parameters (`n_layers=8`):

| Architecture | Stack parameters | Deviation | ppl | ms/token | Status |
|---|---|---|---|---|---|
| rwkv | 94,115,520 | +0.31% | 105.66 | 3.351 | Normal |
| **sunnynet** | 93,828,904 | — | **109.47** | 4.657 | Normal |
| mamba (lr reduced to 1e-3) | 93,410,400 | −0.45% | 126.47 | 3.826 | Re-run passes |
| h3 | 93,588,592 | −0.26% | 128.13 | 3.782 | Normal |
| mamba (lr=3e-3) | 93,410,400 | −0.45% | — | — | **Diverges at step 37 (NaN)** |
| hyena (lr=3e-3) | 93,674,760 | −0.16% | — | — | **gn≈41.5, gradient explosion** |

**Two negative facts**:

1. **The unified lr=3e-3 is too large for mamba/hyena at 100M**. This is not an advantage of SunnyNet but
   a defect of the comparison protocol: lr was not tuned per architecture (§13 already declared this limitation).
   **The reduced-lr re-run is verified (lr=1e-3)**: mamba no longer diverges (100 steps, ppl 126.47);
   confirming that the earlier divergence was **an optimizer step-size problem, not an architecture defect**.
2. **SunnyNet's 100M ppl reaches only second** (109.47), about 3.6% behind rwkv (105.66) (relative
   to rwkv) and about 14.6% ahead of h3. **The "quality first" at 20M does not carry over to 100M** —
   the scaling trend is not monotonically improving, and the available data does not support "the larger the scale, the larger the advantage".

**Corroborating evidence: the missing defenses in the trainer**. When mamba diverged, `_safe_chunk` called
`int(NaN)` on `p_min=NaN` and raised a `ValueError`, crashing the entire comparison process with no results written to disk.
Fixed: `_safe_chunk` first checks `isfinite`; the trainer prints `[diverged]` and aborts gracefully when loss/grad_norm are non-finite,
writing `diverged` into the summary (previously it died silently).

**hyena's segfault**: hyena@100M aborted several times on this machine with SIGSEGV (exit=139),
and **with no definite relation to the step count** (steps=5/20 crash, steps=50 passes; B=2/4 pass, B=8 sometimes passes).
Investigation confirmed **it is not an architecture defect**: the same configuration passes on its own and at low thread counts; the crash occurs when the process's peak RSS reaches
5977–6131MB, an intermittent event under this machine's memory/page-file pressure (§14.5's 15.9GB physical-memory constraint,
plus only 7GB left on the C drive and a page file that can hardly grow).
**Therefore the comparable 100M comparison covers 4 architectures** (sunnynet / rwkv / h3 / mamba-lr1e3),
and hyena is recorded as "cannot be completed stably on this machine" — **no values for it**.

---
## 15. GPU Validation and 0.5B Scaling

### 15.1 GPU Environment

The earlier record "this machine has no graphics card" **does not hold**. Measured: **NVIDIA GTX 1060 / 6144 MiB** (sm_61), driver 572.16 (CUDA 12.8). The project's main venv torch is a **CPU-only build** (`2.14.1+cpu`), so `cuda.is_available()=False`—this is a **software issue, not a hardware issue**.

A separate GPU venv (`E:/sunnyclaw-data/venvs/gpu`, Python 3.12) was created to install **torch 2.7.1+cu118** (cu118 is compatible with sm_61). The download used 8-thread Range chunking (single thread only 262–273 KB/s, chunked reaches 9 MB/s). Validation: `cuda_available=True`, CUDA 11.8, GPU matmul passes.

### 15.2 SunnyNet Runs on GPU, with No CUDA-Specific Failure in Scan/Gating

`experiments/_gpu_probe.py` (independent subprocess per cell), B=8:

| Preset | T | CPU forward | GPU forward | Forward speedup | CPU train step | GPU train step | Train step speedup | GPU peak |
|---|---|---|---|---|---|---|---|---|
| small | 512 | 138.8 ms | 17.4 ms | **7.97×** | 569.9 ms | 57.5 ms | **9.92×** | 804 MB |
| small | 1024 | 439.3 ms | 28.9 ms | **15.20×** | 1186.3 ms | 99.0 ms | **11.99×** | 1562 MB |
| proto20mw | 512 | 1697.7 ms | 121.5 ms | **13.98×** | 4620.8 ms | 384.6 ms | **12.02×** | 2827 MB |
| proto20mw | 1024 | 3308.5 ms | 244.4 ms | **13.54×** | 9408.9 ms | 752.0 ms | **12.51×** | 5373 MB |

(Raw data in `experiments/results/gpu_compare.json`.)

**Conclusions**:

1. **The architecture runs directly on GPU**—the core two-level parallel scan (`cumsum`/`exp`) and the competitive gating (`sigmoid` in pairs) are all standard CUDA operators, and pass without any CUDA-specific rewriting. This shows that the barrier for a GPU kernel is not whether it runs, but **whether performance requires specialization**.
2. **Train step speedup on the order of 12×**, and the speedup grows with T (small T=1024 forward 15.2×).
3. **But this is a consumer-grade old GPU (GTX 1060, 6GB)**, and **no CUDA-side optimization was done** (no fused kernel, no AMP, no `channels_last`)—**the speedup ratio is a lower bound**.
4. **§15.2 did not test peer comparisons on GPU**, so it does not support the claim "faster than peers on GPU" (official implementations of Mamba/H3 etc. under GPU were not tested).

### 15.3 0.5B (scale05b) Ladder Test—Runs on GPU, but Training Is Hard-Constrained by VRAM

`experiments/_05b_gpu_ladder.py`, parameters **498,464,858 total / 481,687,642 stack**:

| Level | Content | Result | Peak VRAM |
|---|---|---|---|
| L1 | Build to cuda (fp32) | **pass** (80.7 s) | 1902 MB |
| L2 | Forward B=1 T=512 | **pass** | 1978 MB |
| L3 | Forward B=1 T=1024 | **pass** | 2044 MB |
| L4 | **SGD train step** B=1 T=512 | **pass** (loss 9.48) | 4426 MB |
| L5 | **AdamW train step** | **OOM** (T=64/128/256 all fail) | >6144 MB |

**Conclusions**:

1. **0.5B forward and SGD training run through on a 6GB GPU**—§14.5's "0.5B cannot run on this machine" should be qualified as: **not instantiable on CPU fp32; forward/SGD runnable on GPU**.
2. **AdamW is not viable**: weights 1.9GB + gradients 1.9GB + AdamW's two momentum states (≈3.8GB) = 7.6GB exceeds 6GB, so **full optimizer training is impossible** at the current VRAM. The feasible paths are gradient accumulation + SGD, or a card with more VRAM, or a sharded optimizer—**none of these were validated here**.
3. **There is no "0.5B training runs through" conclusion**: only a single SGD step ran; no multi-step convergence was run, and there is no 0.5B quality data of any kind.

**Measurement flaw in the ladder script (located)**: `_05b_gpu_ladder.py` runs the T=512/T=1024 forwards and then the train step inside the same process, so **memory fragmentation** makes L4/L5 report OOM. After switching to a separate subprocess running L4 alone: **single SGD step passes** (peak 5004 MB, 563 MB free, loss 9.4274). ⇒ The ladder script's OOM reading is a false negative caused by fragmentation; **each level must use an independent subprocess** (same discipline as the CPU side in §2.4).

**Optimizer state offload attempt (`experiments/_05b_offload.py`, unsuccessful)**: keep AdamW's m/v resident on CPU and transfer per tensor; the theoretical extra VRAM is only one parameter tensor. Measured failure—this machine has only 4.5GB of physical memory left, and allocating the 3.71GB optimizer state triggers `Memory allocation failure`; the page-file pressure in turn makes the CUDA context report `unknown error`. Switching to storing the optimizer state in bf16 (1.86GB) still OOMs, stuck on the activation side (the `exp(P)` allocation in `gated_scan` fails at 20MB). ⇒ **This machine does not have the memory conditions for 0.5B AdamW training**; this is unrelated to VRAM and is insufficient host physical memory.

### 15.4 0.5B Multi-Step Training Probe (`experiments/_05b_train.py`, real corpus)

**Motivation**: §15.3 only answered "how far up the ladder it can run". This adds multiple steps to see whether loss actually moves.

Configuration: `scale05b` (498,464,858 parameters), **momentum-free SGD**, T=128, B=1, lr=1e-3, 30 steps, data taken from the real token streams of the three domains in `data/proc` (not random tokens).

| Step | loss | grad_norm |
|---|---|---|
| 1 | 9.3152 | 36.4 |
| 2 | 9.2628 | 34.0 |
| 3 | 9.4613 | 155.1 |
| 10 | 9.1438 | 40.6 |
| 20 | 8.9834 | 52.0 |
| 30 | 9.3103 | 25.9 |

Peak VRAM 4425 MB (6GB card), build 87.8 s.

**Conclusions**:

1. **Loss oscillates in the noise; 30 steps show no downward trend** (first 9.3152 → last 9.3103, Δ=−0.005, while the step-to-step range reaches 0.72). This is **not convergence data**, only "the machine can run and gradients have values".
2. **grad_norm is large and unstable** (24–155). 0.5B + lr=1e-3 + no warmup is under-calibrated; the architecture cannot be judged from this, and 9.31 **cannot be cited as 0.5B ppl/token quality**.
3. **No momentum is a VRAM compromise**: adding `momentum=0.9` immediately OOMs (weights 1.9 + gradients 1.9 + momentum 1.9 ≈ 5.7GB, approaching 6GB). That is, 0.5B on a 6GB card can only run the plainest SGD.
4. **Gradient accumulation is unvalidated**—genuinely training 0.5B requires gradient accumulation or a sharded optimizer; neither was tested on this machine.

**Addendum: T=64 re-test and stability boundary** (with warmup 5 steps, lr=1e-3, 30 steps):

| Step | loss | grad_norm |
|---|---|---|
| 1 | 9.3587 | 41.8 |
| 20 | 7.7206 | 61.9 |
| 30 | 9.2953 | 34.4 |

First 9.3587 → last 9.2953 (Δ=−0.063), **still within the noise** (step-to-step range 1.87; the 7.72 at step20 is a single fluctuation, not a decline). Peak VRAM 4118 MB.

**Stability conclusion**: **0.5B on a 6GB card is a borderline state, and results are not reproducible**—same configuration (lr=1e-3, T=128) run three times: one run passed all 30 steps, two runs OOMed at the first backward step (failing on only a 3.74GB allocation, while `nvidia-smi` showed 6031MB free). The root cause is the **VRAM allocation cliff of the WDDM platform** (torch explicitly warns `expandable_segments not supported on this platform`), not a model defect. Therefore: **0.5B can only be used for "forward/single-step correctness validation", not as a reproducible training baseline**. Raising lr to 3e-3 makes the first backward step always OOM (activations inflate with the weight update).

### 15.5 GPU-Side Operator Profiling: the Headroom for Fusion Is Smaller Than on CPU

**Motivation**: §14.7's 8.8% upper bound was measured on CPU. The operator cost structure on GPU is completely different, and a re-measurement is needed to answer "is fusion worth it on GPU".

Method: `experiments/_gpu_profile.py`, small preset, T=512, B=8, `torch.profiler` aggregating self time by CUDA kernel, averaged over 12 runs.

| Category | Forward (eval) | Train step |
|---|---|---|
| `cumsum`/scan class | **1.85%** | **1.22%** |
| `sigmoid`/competitive gating class | **2.55%** | **2.02%** |
| `aten::mm` + cuBLAS sgemm | 32.8% | 28.6% |
| `aten::roll` | 3.13% | 1.91% |
| Top 8 operators combined | 51.1% | 53.6% |

**Conclusions**:

1. On GPU the share of these two segments is an order of magnitude lower than on CPU. On CPU the forward `_compete` + `_scan` is about 31.6% (§14.7 single-block profiling); on GPU the two together are under 5% (diluted by GEMM). Therefore the headroom for a "fused kernel" on GPU is smaller than on CPU, not larger.
2. **There is no single dominant operator on GPU** (the largest, `aten::mm`, is 16.4%), consistent with the CPU-side conclusion: improvement can only come from reducing the **number of operators**, and this machine has no nvcc / no CUDA toolkit / no triton (`torch.utils.cpp_extension` even lacks `setuptools`), so no custom kernel can be compiled.
3. The local GPU is used only for "correctness validation + speedup ratios", not for kernel-level optimization; this is the capability boundary of this machine.

### 15.6 Appendix: Cross-Platform Measurement of Competitive Gating `roll` → `index_select`

**Background**: GPU profiling found `aten::roll` takes 3.13% (forward). On CPU, "roll→slicing" was previously rejected (recorded in the rejection list). The two platforms have different cost structures, so a **cross-platform re-measurement** was done.

Micro-benchmark (`experiments/_compete_bench.py`, pure gating, numerical equivalence verified by the same script):

| Platform | roll | index_select | Ratio | Forward maxdiff | Gradient maxdiff |
|---|---|---|---|---|---|
| cuda | 2349.7 µs | 1683.8 µs | **0.717** | 0.0 | 0.0 |
| cpu | 15518.8 µs | 13819.9 µs | **0.891** | 0.0 | 0.0 |

The micro-benchmark on both platforms points to index_select being faster, with exact numerical equivalence—opposite to the old CPU conclusion that "roll is better". The micro-benchmark alone is not sufficient to decide; an end-to-end determination is needed (§14.4 has repeatedly shown micro-benchmark and end-to-end disagreement).

The end-to-end A/B (`experiments/_compete_e2e.py`, independent subprocess per cell, two interleaved rounds) is reported in §15.7; **the end-to-end result governs**.

### 15.7 End-to-End Determination of the Competitive Gating Rewrite

`experiments/_compete_e2e.py`, small, T=512, B=8, train step, independent subprocess per cell, two interleaved rounds. Numerical consistency: the two implementations have exactly the same loss (cuda 9.034 / cpu 9.035); the rewrite **does not degrade quality**.

| Platform | roll, two rounds | index, two rounds | Median difference | Conclusion |
|---|---|---|---|---|
| cuda | 59.57 / 56.87 ms | 58.62 / 64.58 ms | roll faster by 5.8% (between-round range 13.5%) | within noise |
| cpu | 548.1 / 520.6 ms | 519.9 / 547.0 ms | level (0.2%) | no difference |

Conclusion: end to end there is no stable difference on either platform; the rewrite is not adopted, and `roll` is kept.

The same class of lesson appears again: the pure-operator micro-benchmark says index_select is faster (cuda 0.717×, cpu 0.891×), **yet end to end no difference is measurable**. The reason is that the gating operators account for only a small end-to-end share (GPU forward 2.55%, CPU forward about 14%), so the micro-benchmark's gain is diluted into the noise. **A micro-benchmark cannot by itself be the basis for adoption**—consistent with the repeated conclusions of §14.4. This entry joins the rejection list.

---

## 16. Training Speed Matches Peers

**Background**: §13 recorded "training speed falls behind all O(N) peers by 20–52%", the project's only long-standing unclaimed metric. §16 closes it with two **equivalent** changes. Goal: training speed on par with or ahead of peers.

### 16.1 Equivalent Optimization A: Deciding `p.clamp` Away at Construction Time

In `_core`, `p = 1 - delta*(1 - sigmoid(gate))` carries `p.clamp(p_floor, 1.0)`. Analytical bounds: `delta = 1/H ≤ 1/h_min`, hence `p ≥ 1 - 1/h_min = p_min_bound`; and `sigmoid(gate) ∈ (0,1) ⇒ p < 1`. The upper bound always holds, and the lower bound never triggers when `p_min_bound > p_floor`.

| Quantity | Value (`small`, h_min=2.0 / h_max=2048.0) |
|---|---|
| `p_min_bound` = 1−1/h_min | **0.5** |
| `p_floor` = 1/h_max | 4.883e-4 |
| Measured p range (20 steps × 4 layers sampled) | **[0.5902, 0.99994]** |

The true lower bound 0.5902 is **1209×** above `p_floor`, so the clamp is unreachable. It is replaced by a construction-time decision `_need_clamp`:

```python
self._need_clamp = self.p_min_bound < self.p_floor
```

- **Equivalence**: forward output bit-for-bit equal after removing the clamp (`maxdiff = 0.000e+00`, `experiments/_clamp_check.py`).
- **Gain**: eliminates the `ClampBackward1` node from the backward graph (previously `aten::clamp_backward` 59.5 ms / 12 steps, 1.7%) and the tensor round trip of its forward `clamp`. When `h_min=1.0` the decision automatically falls back to `True`, and behavior is unchanged.
- Added 3 property tests (`_need_clamp` true/false sides + measured p falling within the analytical bounds), tests 51 → **54**, all green.

### 16.2 Equivalent Optimization B: Narrow Residual Stream Configurations (`smalln80` / `smalln88`)

§13.3 attributed the cost to "element-wise operators paid by **activation width** (width determined by `dim`)". Lowering `dim` at equal stack parameters (~365k):

| Preset | Configuration | Stack parameters | Deviation |
|---|---|---|---|
| `small` | d128 i192 r64 L4 | 365,844 | baseline |
| `smalln96` | d96 i192 r152 L4 | 363,860 | −0.54% |
| `smalln88` | d88 i176 r196 L4 | 365,924 | +0.02% |
| **`smalln80`** | **d80 i160 r244 L4** | **365,940** | **+0.03%** |

`experiments/_train_bench.py`: independent subprocess per cell + interleaved multiple rounds + median (§2.4 discipline). T=256, B=8, 8 threads, 5 rounds, ms token = 2048/step.

### 16.3 Results: Train Step Speed (ms/token, lower is better, median of 5 independent subprocesses)

| Architecture | Stack parameters | **T=256** | **T=512** |
|---|---|---|---|
| **sunny80** | 365,940 | **0.0953** | 0.0986 |
| rwkv | 361,920 | 0.0982 | **0.0951** |
| sunny96 | 363,860 | 0.1011 | 0.1090 |
| sunny88 | 365,924 | 0.1013 | 0.1089 |
| h3 | 369,760 | 0.1138 | 0.1239 |
| sunny128 (old) | 365,844 | 0.1158 | 0.1243 |
| mamba | 364,960 | 0.1292 | 0.1271 |
| hyena | 392,760 | 0.1312 | 0.1468 |
| linear_attn | 364,720 | 0.1569 | 0.1676 |

**Relative to the old configuration (`small` / dim128)**: T=256 **−17.7%**, T=512 **−20.7%**.

### 16.4 head-to-head: sunny80 vs rwkv (round-by-round pairing, 9 independent subprocesses)

A single median can be misled by "between-round drift", so **round-by-round pairing** was added (each round runs sunny80 and rwkv **once each at adjacent times**):

| Metric | T=256 | T=512 |
|---|---|---|
| sunny80 median | 204.7 ms (0.0999 ms/tok) | 466.5 ms (0.1139 ms/tok) |
| rwkv median | 215.5 ms (0.1052 ms/tok) | 475.8 ms (0.1162 ms/tok) |
| **Paired ratio median (s/r)** | **1.042** | **1.006** |
| sunny80 **rounds faster** | **3 / 9** | **4 / 9** |

The three metrics contradict each other ⇒ the conclusion is "statistically level", not "ahead": the ratio-of-medians favors sunny80, but both the paired median ratio and the sign test point to rwkv, and the between-round range of the two architectures (rwkv single round 171–253 ms) far exceeds the difference between their medians. **This reading is kept as is.**

### 16.5 Four Limitations

1. **Speed is level only at `L=4`**. When layers are aligned to rwkv's `L=6`, the equal-budget narrow configurations (d56/d64/d72) are **30–40% slower instead** (260–342 ms vs rwkv 225 ms)—at equal budget, pushing `dim` down to 56–72 makes the layer-count burden outweigh the width gain, so **the bottleneck is operator count, not FLOPs**. The project's existing alignment protocol uses "stack parameter count" and does not constrain layer count, so the level result in §16.3 rests on SunnyNet using fewer layers.
2. **T=256 is the best point**: here `smalln80` beats rwkv; at T=512 it has turned into rwkv being marginally faster. **This cannot be extrapolated to long sequences**—§13 already showed the peer advantage widens at T ≥ 1024.
3. **Quality cost**: `smalln80`'s 600-step multi-seed ppl is **62.37** (d80) vs 54.89 for the old configuration (d128). What the cost buys is level speed. This is not "fast and good", it is the speed-quality tradeoff point moving forward.
4. **lr was not re-tuned per architecture**, single machine single thread count (8), single implementation (hand-written baseline), same as other sections.

### 16.6 Speed-Quality Pareto Frontier (T=256, stack parameters all ~365k)

| Preset | dim | ppl (600 steps, mean of 3 seeds) | ms/tok | Position |
|---|---|---|---|---|
| `small` | 128 | **54.89** | 0.1158 | best quality |
| `smalln96` | 96 | 58.84 | 0.1011 | compromise |
| `smalln88` | 88 | 61.90 | 0.1013 | **dominated by 96** (same speed, worse quality) |
| `smalln80` | 80 | 62.37 | **0.0953** | best speed |

Peer reference (600 steps, single seed 1234): h3 **53.93** / mamba 63.34 / rwkv 63.86 / linear 91.91.

- **`smalln88` is dominated by `smalln96`**: same speed but worse quality → already removed from the preset candidate list.
- Speed first + quality not worse than same-speed peers: take `smalln80`—its speed 0.0953 is level with rwkv, and its 600-step ppl 62.37 is **better than rwkv 63.86 and mamba 63.34**. ⇒ This is a point that **loses nothing on either side**.
- **Quality first**: still use `small` (0.1158); the cost is being about 15% slower than rwkv.
- **The design intent of `smalln88`** (keep dim lower but rank wider) **is ineffective in measurement**.

### 16.7 Conclusion

- **Train step speed reaches "level"**: `smalln80` at T=256 is **nominally 3.0% faster than rwkv** and 3.7% slower at T=512; the paired test is not significant (3–4/9 rounds ahead). By project discipline, the conclusion is "statistically level with rwkv", not read as "ahead".
- **It is a stable lead over h3 / mamba / hyena / linear_attn**: 16–39% faster at T=256, 20–41% faster at T=512 (the median difference for each architecture exceeds the between-round range).
- It has long been ahead of Transformer (§13's 1.10× now widens, since its T dependence is ∝T while SunnyNet is ~O(N)).
- **§16 does not change**: the peer advantage at long sequences (T≥1024) remains; in quality ranking `small` is still second among the seven architectures; §13's "falling behind across the board" from T=256 to T=4096 still holds **under the old dim128 configuration**.

**Target**: training speed on par with or ahead of peers—**parity achieved (rwkv / h3 / mamba / hyena / linear_attn / transformer), ahead not achieved (rwkv is only nominally behind)**.

---

## 17. T=1024 Training-Mode Speed and Peak Memory Audit

**Trigger**: the claim that "the T=1024 range has the slowest speed and the highest peak memory".
**Gap in existing data**: §10.6 only measured **batch=1 pure forward**, and **memory was never recorded**. §17's `experiments/_T1024_audit.py` fills in these two—training mode (forward + backward + AdamW step) and **activation memory peak**. Protocol: batch=8, T=1024, 8 threads, independent subprocess per cell, build only one model.

Speed (ms/token, smaller is faster):

| Architecture | ms/token | Seven-architecture rank |
|---|---|---|
| **sunnynet** | **0.13186** | **1** |
| h3 | 0.13190 | 2 |
| mamba | 0.14721 | 3 |
| hyena | 0.16754 | 4 |
| linear_attn | 0.17646 | 5 |
| rwkv | 0.19677 | 6 |
| transformer | 0.55693 | 7 |

Peak memory (current_rss after building the model is the baseline, peak_rss is the high-water mark; `act` = the difference of the two):

| Architecture | peak RSS | Rank (large→small) | Activation peak | rss_alloc (fixed) |
|---|---|---|---|---|
| transformer | 3192 MB | 1 | 2769 MB | 423 MB |
| **sunnynet** | **1812 MB** | **2** | 1521 MB | 291 MB |
| hyena | 1809 MB | 3 | 1513 MB | 296 MB |
| mamba | 1780 MB | 4 | 1485 MB | 294 MB |
| linear_attn | 1777 MB | 5 | 1482 MB | 295 MB |
| h3 | 1689 MB | 6 | 1394 MB | 295 MB |
| rwkv | 1611 MB | 7 | 1317 MB | 295 MB |

**Reading (the difference in protocol from §10.6 is key)**:

1. **The ranking in training mode (batch=8, including backward) is completely different from batch=1 pure forward**. In pure forward at T=1024 SunnyNet ranks 5th (§10.6); in training mode it **ranks 1st** (0.13186, effectively tied with h3's 0.13190). The reason is that backward accounts for 56.9% of the train step (§13.3), and the backward of O(N) peers contains large element-wise chains, which suffer more than SunnyNet at batch=8; Transformer's O(T²) attention backward is the most expensive (0.557, 4.2× slower).
2. "SunnyNet has the highest memory" does not hold: its peak RSS (1812 MB) ranks **2nd** among the seven architectures, only 0–12% above 5 O(N) peers, and **43% below Transformer** (1812 vs 3192). The highest memory is Transformer (activation 2769 MB), because its attention must store O(B·H·T²) intermediates.
3. **Activation memory is the main term**: the seven architectures' `rss_alloc` (parameters + optimizer + inputs) all lie in 291–423 MB, and the differences come entirely from activations. SunnyNet's activation of 1521 MB is **higher than four peers** (h3 1394 / rwkv 1317 / linear_attn 1482 / mamba 1485), because it stores the entire scan states `(B,T,d)` (§4.2 already recorded this mechanism), but the Henry class (hyena) is in the same bracket and Transformer is far above it. That is: SunnyNet's memory is on the high side among O(N) peers and significantly lower than Transformer, but is not the highest.
4. **Point-by-point against the claim**: "slowest speed + highest memory at T=1024"—**the premise is batch=1 pure forward**—under that protocol §10.6 shows it ranks 5th (not slowest; transformer is last), though it is indeed slower than 4 peers; memory is **not the highest under any protocol**. Under the training-mode protocol neither charge holds.

**Limitations**: §17 measures the single-step cost of a randomly initialized model (speed and memory are **unrelated to model quality**, so random tokens do not harm the conclusion); the AdamW state makes `rss_alloc` higher than with SGD, but the seven architectures are comparable under the same protocol.

---

## 18. Cross-Distribution (OOD) Generalization: Measurement on an External New Dataset

**Trigger**: feedback that "generalization is poor; switching to an independent dataset degrades a lot".
**Gap in existing data**: all quality numbers in §3/§10/§12 are ppl on **in-distribution** validation sets (validation files and training files come from the same corpus, split only at the file level). **A distribution never seen during training was never tested**—so this claim was previously **neither confirmed nor refuted**.

**New datasets** (`prepare_extra.py`, sources reachable in actual network tests on this machine; **not splits of the original corpus**):

| New domain | Source | Token count | Out-of-vocabulary character rate |
|---|---|---|---|
| `arxiv_en` | arxiv API (cs.CL/LG/AI/stat.ML) 500 English scientific abstracts | 236,595 | **0.05%** |
| `zh_tech_news` | Xinhua/GMW/People.cn science and education body text, 97 articles | 164,711 | **23.02%** |

**Fairness issue**: the 8192 BPE vocabulary was trained on a **traditional-Chinese** ancient-text corpus (the raw text of `zh_ancient` is traditional characters such as "女誡/婦德/賜芳筵"). Common characters in simplified-Chinese news (电/戏/级/实/这/验/时/动/帧…) largely fall outside the vocabulary ⇒ `zh_tech_news` has an out-of-vocabulary rate of **23.02%**, while `arxiv_en` is only **0.05%**. **The high ppl in the new Chinese domain has a tokenizer-mismatch component, not purely distribution shift**—so the results below are **stated per domain, not merged into a single ratio**.

**Protocol**: train 600 steps on the original three-domain mixture (0.4/0.35/0.25), batch=8, T=256, lr=3e-3, seed=1234, save ckpt; then compute ppl on "three-domain val + two new-domain val" (T=1024, 24 windows, fixed seed 2024). Script `experiments/_ood.py`, data `experiments/results/_ood.json`. The table below is the result after 600 training steps.

### 18.1 Results (ppl, lower is better; 600 steps, single seed, T=1024)

| Architecture | zh_ancient | en_literature | code_python | **arxiv_en** (new) | **zh_tech_news** (new) | orig mean | ext mean | **ood_gap** |
|---|---|---|---|---|---|---|---|---|
| **sunnynet** | **326.12** | 20.85 | 14.49 | **35.58** | **321.08** | 120.49 | 178.33 | **1.480** |
| h3 | 358.14 | **18.69** | **12.99** | **31.52** | 446.77 | 129.94 | 239.15 | 1.840 |
| rwkv | 375.41 | 23.59 | 16.41 | 41.03 | 427.22 | 138.47 | 234.12 | 1.691 |
| mamba | 373.40 | 23.31 | 16.24 | 40.87 | 499.74 | 137.65 | 270.30 | 1.964 |
| hyena | 404.47 | 32.06 | 24.06 | 54.22 | 400.39 | 153.53 | 227.31 | 1.481 |
| linear_attn | 422.43 | 36.94 | 28.43 | 61.55 | 489.23 | 162.60 | 275.39 | 1.694 |
| transformer | 423.77 | 37.66 | 28.12 | 61.34 | 531.77 | 163.18 | 296.55 | 1.817 |

**Per-cell ranking in the new domains**:

| New domain | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| `arxiv_en` | h3 31.52 | **sunny 35.58** | mamba 40.87 | rwkv 41.03 | hyena 54.22 | transformer 61.34 | linear 61.55 |
| `zh_tech_news` | **sunny 321.08** | hyena 400.39 | rwkv 427.22 | h3 446.77 | linear 489.23 | mamba 499.74 | transformer 531.77 |

### 18.2 Reading

1. The "poor cross-dataset generalization" claim is refuted by measurement. On domains **never seen during training**:
   - **Chinese tech news (`zh_tech_news`): SunnyNet ppl 321.08, 1st among the seven architectures**—and it wins **under the most unfavorable input conditions**: that domain's out-of-vocabulary rate is 23.02% (traditional vocabulary against simplified text), while other architectures use the same vocabulary but have higher ppl (hyena 400, h3 447, transformer 532).
   - **English scientific abstracts (`arxiv_en`): 35.58, 2nd among the seven**, second only to H3 (31.52), and better than mamba 40.87 / rwkv 41.03 / transformer 61.34.
2. **The smallest OOD drop is SunnyNet and Hyena (both 1.48)**, the largest is Mamba (1.964). SunnyNet's performance decline from the training distribution to new distributions is **in the lowest bracket among the seven architectures**, the opposite of "poor generalization".
3. **A high absolute ppl in the new Chinese domain (321) does not equal poor generalization**: all architectures are high in that domain (400–532). **But there is a protocol trap**: `zh_ancient`'s "in-domain" ppl is the highest for all architectures (sunnynet 326, transformer 424), far above the same architecture's `en_literature` (20.9 / 37.7). The reason is that **600 steps is far from convergence + that domain (traditional ancient text) is itself the hardest, so using it as an "in-domain baseline" to compute a ratio is distorted**—SunnyNet's `news/zh_anc` = 0.985 looks like "no drop" but is actually caused by an **inflated denominator**, and **cannot be read as "cross-domain better than in-domain"**. The only reliable generalization metric is the same-language `arxiv_en / en_literature` whose denominator is not contaminated by a hard domain: SunnyNet **1.706**, in the same bracket as h3 (1.686) / hyena (1.691), better than mamba (1.753) / rwkv (1.739).
4. **Consistency with the in-distribution conclusion**: in distribution SunnyNet ranks 2nd at T=256 (§12); under OOD, `arxiv_en` is also 2nd. **The ranking is stable across the two distributions**—the phenomenon of "good in distribution, collapsing across distributions" does not occur.

### 18.3 Limitations

1. **Single seed, 600 steps, not converged**: same limitation as §10.5, can only be read as an early sample-efficiency ranking.
2. **The 23% unk rate in the new Chinese domain is a tokenizer-level confounder**: the absolute ppl of `zh_tech_news` is inflated by out-of-vocabulary characters and **should not be cited alone**; the value of that domain is that **the seven architectures are comparable under the same protocol** (all subject to the same vocabulary), so the ranking is valid and the absolute values are not.
3. **The new domains are limited in size** (arxiv 236K / news 165K tokens), and `arxiv_en` is **abstracts** rather than full text, with a length distribution different from the original books.
4. **No domain balancing was done**: the new-domain test sets all come from a single source and do not cover the intermediate case of "new domain but same genre".

**Conclusion**: on an **external independent dataset**, SunnyNet's absolute ppl falls in the top bracket of the seven architectures—`zh_tech_news` **1st** (321.08, under the unfavorable condition of 23% unk) and `arxiv_en` **2nd** (35.58); the combined OOD drop of 1.480 is the **lowest bracket** among the seven (tied with hyena). That is, "SunnyNet collapses after switching to an independent dataset" **has no support**, and on the contrary it ranks **1st on the most unfavorable new domain (simplified-Chinese news)**. The reverse claim of "strong generalization" also does not hold—absolute ppl is still high (English new domain 35.58, Chinese new domain 321), 600 steps is not converged, single seed, and §18.2-3 has explained the ratio protocol trap in the Chinese domain. The only reliable conclusion of §18 is: **cross-distribution performance is not worse than peers of the same complexity**.

---

## 19. Short-Sequence Inversion: Training-Mode Verification and Mitigation Magnitude

**Trigger**: feedback that "at T<384 it is slower than Transformer, limiting its use in routine tasks such as short-sentence classification and dialogue response", with a request to "fix and improve". **That statement is based on the batch=1 pure-forward protocol (crossover at T≈384 in §2.4)**, but those tasks are in practice **training/fine-tuning** scenarios (with backward, batch>1), so a re-measurement in training mode was done.

Script `experiments/_shortseq.py`: batch=8, train step (forward + backward + AdamW), independent subprocess per cell, 3 interleaved rounds taking the median.

**Training-mode ms/token median (3 rounds)**:

| Configuration | T=128 | T=256 | T=384 |
|---|---|---|---|
| `small` (d128 i192 r64 L4) | 0.13009 | 0.14094 | 0.15241 |
| `smalln80` (d80 i160 r244 L4, §16) | **0.12315** | **0.10675** | **0.11148** |
| `transformer` (stack-aligned 6 layers) | 0.11785 | 0.20975 | 0.26238 |

**Ratio relative to Transformer (>1 = slower than TF)**:

| Configuration | T=128 | T=256 | T=384 |
|---|---|---|---|
| `small` | 1.104 | 0.672 | 0.581 |
| `smalln80` | **1.045** | **0.509** | **0.425** |

**Reading**:

1. **The training-mode crossover moves down considerably from the forward one**: the forward (batch=1) crossover is T≈384 (§2.4), while in training mode (batch=8) **the crossing is already complete at T=256** (`small` 0.672, `smalln80` 0.509). This is consistent with §2.3's "the crossover moves to smaller T as batch grows", and it quantifies it.
2. **T=128 is still inverted, but by a small margin**: `small` is 10.4% slower, `smalln80` is **4.5%** slower. Note that this bracket falls within the noise band of a **high-memory-pressure measurement** (in this run only 2.54 GB of physical memory was available / 84% load), so a 4.5% difference **is not enough to judge as a real disadvantage**, but it also should not be claimed to be eliminated.
3. **The narrow residual stream (§16) is also a net improvement for short sequences**: `smalln80` is faster at all three T, speeding up **5.3% (T=128) / 24.3% (T=256) / 26.9% (T=384)** relative to `small`. ⇒ The mitigation direction for the short-sequence disadvantage **is verified effective**: pushing dim from 128 down to 80 and rank up to 244 (equal stack parameters) narrows the short-sequence disadvantage from 10.4% to 4.5%, and longer sequences are ahead across the board.
4. Widening or adjusting batch cannot "make it up": §16.5 already showed that when layers are aligned to L=6, equal-budget narrow configurations are instead 30–40% slower, and §13.3 already ruled out the three options of "compute volume / threads / operator rewriting"—**the root cause is operator count**, so the only effective measures are **reducing the operator count** (fusion, fewer layers plus wider) and **compressing the activation width** (narrow residual stream).

**Conclusion**:

> The short-sequence inversion **holds only under the batch=1 pure-forward protocol** (crossover T≈384).
> In **training mode (batch=8)**, the crossover moves down to **T<256**: at T=256/384 SunnyNet is already
> 1.5–2.4× faster than Transformer. At T=128 it is still 4.5–10.4% slower (of which 4.5% falls within measurement noise).
> **The narrow residual stream narrows the T=128 disadvantage by about 6 percentage points**, a verified mitigation.

**Limitations**: 3 rounds taking the median, memory load 84% (absolute values should be treated as an upper bound), single batch=8, no measurement of very short sequences such as T=64/96; the real forms of "short-sentence classification / dialogue response" (classification head / multi-turn concatenation) were not constructed for measurement, and §19 covers only **the train step cost of language modeling**.

---

## 20. Extremely Short Sequences and Decoding Mode: the Real Boundary of the Inversion

**Trigger**: criticism that the short-sequence inversion "limits its use in short-sentence classification and dialogue response". §19 measured only down to T=128, and all its data is **training mode or whole-segment forward**. §20 adds two protocols never covered before: **extremely short sequences (T=32–96, the real length of routine short sentences)** and decoding mode (token-by-token autoregression).

### 20.1 Extremely Short Sequences (training mode, `experiments/_shortseq2.py`)

A new narrow configuration `smalln64` was added (d64 i128 r368 L4 = 367,508, +0.45%). Protocol: batch=8, train step, independent subprocess per cell, 3 interleaved rounds taking the median.

Training-mode ms/token median and ratio relative to Transformer (>1 = slower than TF):

| Configuration | T=32 | T=64 | T=128 | T=192 | T=256 |
|---|---|---|---|---|---|
| `small` (d128) | 1.183 | 1.218 | 0.944 | 0.839 | 0.898 |
| `smalln80` | **0.907** | **0.940** | **0.730** | **0.549** | **0.557** |
| `smalln64` | **0.867** | **0.889** | **0.772** | **0.688** | **0.455** |

**Reading**:

1. In training mode the narrow configurations are not slower than Transformer at any T (ratios ≤0.94), being 6–13% faster at T=32/64 and 1.4–2.2× faster at T≥192. §19's "T=128 still 4.5% slower" applies to the old `d128` configuration—`smalln64` brings T=128 to 0.772 (23% faster).
2. **T=48/96 is non-monotonic** (T=96 slower than T=128), contradicting the trend at T=32/64/128. This is a **single-round noise** signature (this machine's memory usage was 83%, absolute values are an upper bound), so **this table does not cite these two points**; more rounds are needed to settle it; the single-round data does not support "all T are already ahead".
3. The old `d128` configuration is still 18% slower at T=32 (1.183); **narrowing is an effective measure** (same conclusion as §16/§19).

### 20.2 Decoding Mode (token-by-token autoregression, `experiments/_decode_bench.py`)

**This is a protocol that was completely missing before.** The real deployment form of "dialogue response" is token-by-token decoding, and this repository's Transformer `forward_step` was originally `NotImplementedError` ⇒ **the two sides had never been compared here**. For this, a standard KV cache inference path was implemented (single-step/continuation vs whole-segment forward maxdiff **6.7e-16** in fp64, 2 new invariant tests added, total tests **52 → 54**).

Mechanism differences (which determine that the decoding-mode conclusion may be opposite to the training-mode one):

- Transformer decoding: the KV cache saves recomputing K/V, but each step's `q·k` is still **O(t)**, and the per-step memory access grows with history.
- SunnyNet decoding: the state `(B,L,D)` is **fixed-length**, the per-step cost is **independent of t**, always `O(1)`.

Decoding-mode ms/step median (batch=1, 3 rounds):

| Configuration | T=32 | T=64 | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|---|---|
| `small` | 5.218 | 5.072 | 6.010 | 6.365 | 7.178 | 6.508 |
| `smalln80` | 7.861 | 4.469 | 5.664 | 7.509 | 6.406 | 5.247 |
| `transformer` | 4.164 | 3.533 | 4.180 | 3.512 | 4.091 | 4.872 |

**Ratio relative to Transformer (>1 = slower than TF)**:

| Configuration | T=32 | T=64 | T=128 | T=256 | T=512 | T=1024 |
|---|---|---|---|---|---|---|
| `small` | 1.253 | 1.436 | 1.438 | 1.812 | 1.755 | 1.336 |
| `smalln80` | 1.888 | 1.265 | 1.355 | 2.138 | 1.566 | 1.077 |

**Reading (the most important negative conclusion of §20.2)**:

1. **In decoding mode SunnyNet is slower than Transformer across the board (1.08–2.14×)**—**opposite** to the training-mode conclusion (in training mode the narrow configurations are 1.3–2.2× faster at T≥128). **The same criticism has different answers under the two protocols**, depending on whether "routine NLP tasks" means training/fine-tuning or autoregressive decoding.
2. **The root cause is still the operator count per layer** (same origin as §13.3/§16.5): a SunnyNet single step must run the full operator chain of L layers (30+ operators per layer), while Transformer has 6 layers per step but each layer has only a few large operators qkv/attn/proj/mlp. Although the fixed-length state saves the O(t) term, **the constant term is swallowed by operator dispatch overhead**.
3. **The theoretical advantage of an `O(1)` single step is not realized within T≤1024**: the ratio falls from 1.25 to about 1.08–1.34 as T grows, with a converging trend (Transformer's O(t) term is growing), but no crossing occurs within **T≤1024**. The curve for longer T (≥2048) is in §20.3.
4. **SunnyNet is the slower side under the T≤1024 decoding protocol**, which does not support "SunnyNet is suitable for dialogue response".

### 20.3 Limitations

1. **batch=1, single sequence**: batched decoding was not tested (real serving is batched inference, and the ratio may change).
2. No KV cache quantization/paging was done: the comparison is "naive KV cache vs fixed-length state" and contains no engineering optimization.
3. Both sides use PyTorch interpreted execution, with no hand-written SIMD kernel; hand-written SIMD implementations such as llama.cpp are outside the scope of §20 (§21/§22 already did a separate AVX2 kernel comparison).
4. 3 rounds taking the median, this machine's memory usage 83%, absolute values should be treated as an upper bound.

**Conclusion**:

> The short-sequence inversion **depends on the protocol**. **Training mode (batch=8)**: the narrow configurations are
> not slower than Transformer across T=32–256 (6%–2.2× faster), and §19's "T=128 still slower" holds only for the old `d128`.
> **Decoding mode (batch=1, token by token)**: SunnyNet is **slower across the board by 1.08–2.14×**, opposite to the training-mode conclusion;
> its `O(1)` single-step advantage is **not yet realized** within T≤1024, and shows a converging trend as T grows but no crossing.
> ⇒ If "routine NLP tasks" means **training/fine-tuning**, the inversion is essentially eliminated; if it means **autoregressive decoding**, the inversion genuinely exists.

---

## 21. Hand-Written AVX2 Kernel: Build Chain Connected and Speedup Measured

**Goal**: do a hand-written SIMD kernel on CPU, escaping the limitation that "fast inference depends on GPU".

### 21.1 Compiler Re-Check

§13.4 previously ruled out hand-written kernels on the basis that "this machine has no compiler". Re-check results:

| Item | Previous record | Measured |
|---|---|---|
| MSVC `cl.exe` | "present but missing SDK, unusable" | indeed present (14.39.33519), still missing Windows SDK headers |
| gcc / clang / zig | "all absent" | indeed all absent from the system PATH |
| **w64devkit** | not probed | **installable**: GCC 16.2.0 + its own mingw-w64 headers |

`w64devkit` is a single-file self-extracting package with its own libc and all headers; it needs no Windows SDK and no `reg.exe` (avoiding the sandbox blacklist). Installed at `F:\devtools\w64devkit` (727MB).

### 21.2 Validation Chain (level by level, all measured passing)

1. `gcc -O3 -mavx2 -mfma` compiles an AVX2 + FMA intrinsics program → runs correctly.
2. `-shared` compiles a DLL → loaded via `ctypes` → first-order recurrence scan values vs numpy **1.19e-07**.
3. The hand-written kernel aligns with PyTorch autograd: forward `2.38e-07`, `dp` `9.54e-07`, `dnu` **`0.000e+00` (bit-for-bit identical)**. Already frozen as an invariant test (total tests **54 → 55**).

### 21.3 Speedup (`experiments/_kernel_bench.py`)

Kernel: `sunnynet/kernels/scan_avx2.c` (**original**; channel dimension 8-way AVX2 parallelism, FMA fusion, scalar tail loop; backward includes the `ds *= p_t` time-passing term). Reference: torch `logspace_scan`. **Single thread against single thread** (the kernel is single-threaded, hence `torch.set_num_threads(1)`):

| B | T | N | AVX2 | torch | Speedup |
|---|---|---|---|---|---|
| 8 | 256 | 160 | 0.154 ms | 3.397 ms | **22.03×** |
| 8 | 512 | 160 | 0.480 ms | 6.968 ms | **14.50×** |
| 8 | 1024 | 160 | 1.903 ms | 11.514 ms | 6.05× |
| 8 | 1024 | 512 | 9.010 ms | 88.610 ms | 9.83× |
| 8 | 2048 | 512 | 31.139 ms | 136.811 ms | 4.39× |
| 4 | 4096 | 512 | 35.130 ms | 151.688 ms | 4.32× |

**Reading**:

1. **Operator-level scan speedup 4.3–22×**. The torch version materializes intermediate tensors via `cumsum/exp`; this kernel streams through in a single pass with no intermediate allocation, and FMA fuses `p*s + g` into one instruction.
2. **The speedup narrows as T grows** (22.0× → 4.3×): the kernel time is linear in T, while the torch version amortizes its dispatch overhead at long T. But even at T=4096 (beyond the lengths commonly used for training in this project) it is still **4.3×**.
3. **Protocol boundaries**: ① this is a **operator-level, single-threaded** protocol; real training torch uses 8 threads, which would narrow the speedup; ② scan is only part of the train step (§CPU profiling: forward 24.8% / backward 56.9%).

### 21.3b End-to-End A/B (`experiments/_kernel_e2e.py`)

The kernel was wired into the scan call site of `SunnyBlock` (`sunnynet/kernels/scan_auto.py`, an `autograd.Function` wrapping ctypes), with everything else unchanged. T=256, batch=8, 8 threads, smalln80, 4 interleaved independent subprocesses:

| Round | torch scan | AVX2 kernel |
|---|---|---|
| r1 | 268.66 ms | 279.55 ms |
| r2 | 281.18 ms | 280.14 ms |
| r3 | 394.52 ms | 353.98 ms |
| r4 | 314.24 ms | 289.48 ms |
| **Median** | **297.71 ms** | **284.81 ms** |

**Reading**:

1. **Correctness confirmed**: across 4 rounds **loss is 7.589502 every time, bit-for-bit identical** ⇒ wiring in the kernel did not change the training semantics.
2. **The end-to-end speedup is only 1.045×, and falls within the noise**: the between-round range reaches 47% (268–394 ms), and the kernel wins 2/4 rounds. **By measurement discipline (§5/§16.4), this difference is not enough to judge as a real improvement**.
3. **The gap from the operator-level 6.05× (same T) is as expected**: scan is a hotspot inside `logspace_scan`, but its share of the whole train step is limited, and the torch version already parallelizes the `cumsum/exp` part with 8 threads. ⇒ "N× faster at operator level" cannot be linearly extrapolated to the whole model; the two must be stated separately.

### 21.4 Relationship to `torch.compile` (still unusable, root cause identified)

This time it was found that `torch.compile` can now **reach** Inductor's C++ compilation step (previously it could not enter at all), but it is stuck in two places:
① Inductor's Windows branch hard-codes a requirement for an MSVC-style CLI (`subprocess.check_output(['where','cl'])` + `/I /O2 /MD /link`), and gcc argument syntax is not accepted by `_is_msvc_cl`/`_is_clang`;
② the MSVC link path depends on `reg.exe`, which is blocked by the sandbox blacklist.

**Conclusion**: the automatic fusion route (Inductor) on this machine requires clang-cl, and is **not yet connected**; but the **hand-written kernel route is already usable**—it does not depend on Inductor and needs only gcc, as verified.

### 21.5 Answer to "Can It Beat llama.cpp"

- **Not yet benchmarked**: llama.cpp / ggml are not installed on this machine, and there is **no direct measurement**, so no data supports "already ahead" and no data supports "cannot do it".
- **Necessary conditions already in place**: a hand-written AVX2/FMA kernel, numerical alignment with autograd, compilable and loadable. This is the **equivalent tooling system** for doing hand-written SIMD kernels (llama.cpp's core approach).
- **Structural difference (favorable)**: SunnyNet's decoding single step is a **fixed-length state O(1)** (§20.2), whereas llama.cpp's Transformer decoding must do O(t) attention over the history at each step. On **long-context decoding** this is an architecture-level advantage; but the §20.2 measurement shows that within T≤1024 this advantage is not yet realized (SunnyNet is instead 1.08–2.14× slower), and **only overtakes at T=4096 (0.788, 21% faster)**.
- ⇒ The toolchain is connected and the **prerequisites for a top-tier kernel are met**; the benchmark against llama.cpp **awaits later measurement**, and §21.5 draws no conclusion.

### 21.6 Reproduction

```bash
# Compiler installed at F:\devtools\w64devkit (GCC 16.2.0)
export SUNNY_W64=F:/devtools/w64devkit
python -X utf8 experiments/_kernel_bench.py
```

**Limitations**: ① only the single scan operator is implemented; the remaining operators (competitive gating, CSN, readout) are not yet kernelized; ② the end-to-end A/B has been done (§21.3b), **speedup 1.045×, falling within the noise, no real gain observed**; ③ the backward kernel's `s0` non-zero branch was not tested separately (the test uses `s0=0`); ④ AVX-512 was not tested (this machine's i7-8750H has only AVX2, §13.4); ⑤ no direct benchmark against llama.cpp (not installed on this machine, no measurement).

---

## 22. Fused Kernel: Competitive Gating + Gating Probability + Scan in a Single Pass

**Trigger**: §21 only kernelized the single scan operator, and end to end it was only 1.045×, falling within the noise. This time the entire segment **before** scan in `SunnyBlock` (in-group pairwise competition → gating probability → scan) was fused into one kernel.

### 22.1 Why It Can Be Fused

`SunnyBlock._pick_group` returns **8** under **all presets** of this project (measured dim 64/72/80/96/160/320/512/640/1024/1408), exactly the width of one AVX2 register. Thus `torch.roll(gg, -1/-2, dims=-1)` in `_compete` degenerates into a **`_mm256_permutevar8x32_ps` lane permutation**, and in-group competition stays entirely within registers.

On the torch side, each layer originally had to materialize (B,T,D)-level intermediate tensors: 2 `roll` + 2 `sub` + 2 `sigmoid` + `mul` (competition), `sigmoid` + `sub`/`mul` (probability), then the whole `log`/`cumsum`/`exp`/`cumsum` (scan). The kernel computes it all in one pass, and the intermediates never enter memory.

**Second-round optimization**: `sigmoid(g_i − g_j) = 1/(1 + e_j/e_i)` (`e = exp(−g)`), **one `expf`** per lane replaces the original three.

### 22.2 Validation Chain (numerical, all measured)

| Level | Reference quantity | maxdiff |
|---|---|---|
| Kernel forward `u` vs torch reference | fp64 reference | **7.97e-08** |
| Kernel forward `p` | same as above | 7.51e-08 |
| Kernel forward `states` | time-by-time recurrence | 7.40e-08 |
| Kernel backward `dq` vs autograd | — | 1.94e-07 |
| Kernel backward `ddelta` | — | 1.84e-05 |
| Kernel backward `ds0` | — | 7.17e-06 |
| **Loss after wiring into the model** | torch path | **0.000e+00 (bit-for-bit identical)** |
| **59 parameter gradients after wiring into the model** | torch path | maximum absolute difference **4.60e-09** |

Total tests **55 → 59** (new: kernel forward, kernel backward, end-to-end wiring, inference-mode equivalence, 1 item each).

### 22.3 Train Step A/B (`experiments/_core_e2e.py`, 8 threads, interleaved independent subprocesses)

smalln80 (d80/i160/r244/L4), batch=8:

| T | rounds | torch median | kernel median | Median ratio | Paired ratio median | kernel wins |
|---|---|---|---|---|---|---|
| 256 | 4 | 198.45 ms | 177.77 ms | **0.8958 (1.116×)** | 0.9043 | **4/4** |
| 256 | 6 | 213.24 ms | 186.49 ms | **0.8746 (1.143×)** | 0.8981 | **6/6** |
| 512 | 6 | 462.44 ms | 420.10 ms | **0.9084 (1.101×)** | 0.9434 | **5/6** |

**Reading**:
1. **All three metrics agree in direction** (median ratio, paired ratio, sign test), unlike §21.3b's 1.045× (2/4 wins, metrics contradicting each other) ⇒ **this can be judged a real improvement**.
2. The improvement is 1.10–1.14×. The gain at T=512 is slightly smaller (1.101×, 5/6), consistent with scan's share falling as T grows.
3. **Loss is bit-for-bit identical throughout** (T=256: 9.0264→9.0262; T=512: 9.0242→9.0240) ⇒ the speedup is not bought with quality.
4. **Protocol**: 8-thread train step. §21.3b's single-threaded kernel against 8-thread torch is an **unfair protocol**, and this is one of the main reasons that 1.045× was low.

### 22.4 Train Step Profiling: Where the Kernel's Ceiling Is (`experiments/_scan_share.py`)

T=256, batch=8, 8 threads, breakdown of the 186 ms train step (profiler self CPU):

| Item | Share |
|---|---|
| Forward | 28.2% |
| CE | 4.6% |
| **Backward** | **88.7%** |

Operator-level top: `aten::mm` **33.0%** (63 calls, goes through MKL; §13.4 measured it at 80% of peak, incompressible), `_log_softmax_backward_data` 8.6%, `mul` 7.3%, `fill_` 5.1%, `cat` 4.2%. **The scan family is only 6.4%**, and the whole train step has **4524 aten calls**.

⇒ **Conclusion**: the kernelization ceiling is bounded by two things—① matrix multiplication already uses MKL and is near peak, leaving no room for the kernel; ② although the share of element-level operators is high, backward dominates absolutely. Fusing the whole segment before scan yields 1.10–1.14×, already close to the reasonable upper bound of this route.

### 22.5 Inference Mode: the Gain Depends on the Thread Protocol and Model Width

The single-step fused kernel (`sunnynet/kernels/step_avx2.c`: deep fusion + depthwise separable convolution single step) was wired into `SunnyBlock.step`, with numerical equivalence confirmed (single step vs whole segment **1.13e-06**).

**First round of measurement (8 threads): all results unfavorable**:

| Protocol | kernel/torch | Rounds won |
|---|---|---|
| T=128 B=1 8 threads | 0.736 | 0/4 |
| T=256 B=1 8 threads | 0.870 | 0/4 |
| T=512 B=1 8 threads | 0.810 | 0/4 |
| B=4/8/16/32 8 threads | 1.06–1.25 | 0/3 each |

**Root causes (located by measurement, three)**:
1. **ctypes FFI overhead 13.6 µs/call** (28.5 µs including the `data_as` pointer fetch), whereas one torch 3-operator chain on (1,80) costs only **7.2 µs**. In B=1 decoding, 4 layers × 2 kernel calls per step = 8 calls, so **pure FFI alone is 0.1–0.2 ms**.
2. OpenMP is a negative gain on small parallel domains: when the parallel domain is `(B, D/8)`, thread wake-up is about **100 µs** (measured 131 µs at B=1 vs 28 µs single-threaded). An `if(work >= 256)` clause was used to fall back to serial as a fix.
3. **A more important protocol finding**: in B=1 decoding, **torch with 1 thread is 1.42× faster than with 8 threads** (2.71 vs 3.84 ms/step). **All previous decoding measurements used 8 threads, so the protocol itself was suboptimal** ⇒ part of the first-round A/B's "all negative" comes from **comparing a single-threaded kernel against torch's suboptimal 8-thread configuration**.

**Second round: fair single-thread protocol + width sweep (`experiments/_decode_dim.py`, T=256 B=1)**:

| Configuration | dim | 3-round kernel/torch | 3-round wins | 5-round kernel/torch | 5-round wins |
|---|---|---|---|---|---|
| n80 | 80 | 0.907 | 2/3 | 0.938 | 4/5 |
| d160 | 160 | 0.929 | 3/3 | 0.986 | 2/5 |
| d320 | 320 | 0.939 | 3/3 | 0.952 | 4/5 |
| d512 | 512 | 1.035 | 0/3 | 0.983 | 3/5 |

**Third round: re-test after adding delta caching (`experiments/_decode_final.py`, 5 rounds, three metrics side by side)**:

| Configuration | dim | Median ratio | Paired ratio median | Wins |
|---|---|---|---|---|
| n80 | 80 | 0.9421 | **0.9421** (agree) | 3/5 |
| d160 | 160 | 0.8342 | 0.9384 (**contradict**) | 3/5 |
| d320 | 320 | 0.9699 | 0.8789 (**contradict**) | 4/5 |

**Judgment after convergence**:
1. **n80 (the dim=80 actually used) is the only configuration where the three metrics agree**: both the median ratio and the paired ratio are **0.9421**, i.e. **a tendency of about 5.8% speedup**. But **the sign test wins only 3/5** ⇒ **the significance threshold is not reached, and no improvement can be judged**.
2. **d160/d320 have the three metrics contradicting each other** (the median says 16.6%/3% faster, the pairing says 6%/12% faster, the sign test 3/5 and 4/5) ⇒ **falls within the noise**.
3. The cause of the first round's "all negative" is an unfair thread protocol (1-thread kernel against torch's suboptimal 8-thread configuration); this methodological correction holds and is important; but **after switching to the fair protocol the gain is still unstable**.
4. The root cause is unchanged: at B=1 the tensor (1,D) is too small, and the operator dispatch saved by fusion is eaten by ctypes cross-boundary calls and tensor allocation. Measured, a pure ctypes call costs 0.28 µs (zero arguments) / 2.02 µs (8 arguments), and a single torch operator 2.42 µs—**the same order**, so fusion must rely on "calling fewer times" rather than "being faster each time".

**Boundaries**:
- The table above is **B=1 single-threaded**; multi-threaded production decoding needs a separate measurement (not done here).
- **The inference-mode kernel gain is not significant**, and it is off by default (off when `SUNNY_KERNEL` is unset); `set_kernel_active()` can switch it at runtime, and **enabling it only in training mode is recommended**.
- **The real way out for inference mode is not the kernel but the architecture**: SunnyNet's single step is O(1) and Transformer's is O(t), with the crossover at T≈2048–4096 (§20.2; 21% faster at T=4096). **Long-context decoding is the point where it pays off**, and short-T decoding of small tensors is simply not an advantageous scenario.

**Two inference-mode costs fixed along the way** (numerically equivalent, verified):
- `delta()` costs **57.2 µs** per call, 229 µs per step for 4 layers (8.3% of the decoding step)—it is purely a function of theta, and a version-number cache was added (`delta_cached()`), automatically invalidated when theta changes (measured maxdiff 0.0).
- The `contiguous().float()` in `step_core/dw_step` is decided away up front, and no copy is made on a hit.

### 22.6 Joint Comparison of ppl and Training Speed (`experiments/_kernel_ppl.py`)

Same protocol (smalln80, 200 steps, T=256, batch=8, 8 threads, same seed and same data mixture):

| Path | ppl | ms/step | tokens/s | loss start→end |
|---|---|---|---|---|
| torch | 87.6326 | 206.58 | 9914 | 9.0258 → 3.9335 |
| kernel | 87.2331 | 176.76 | **11586** | 9.0258 → 3.9315 |

**Speedup 1.169×** (tok/s 9914 → 11586).

**Qualitative vs quantitative distinction of the 0.40 ppl difference**:
- Single-step loss on the same batch of data and the same forward: torch and kernel are **bit-for-bit identical (0.000e+00)**, with a maximum absolute difference of **2.98e-08** across 59 parameter gradients.
- After 20 training steps the loss difference is **2e-06** (6.748762 vs 6.748760).
- ⇒ **The 0.40 ppl difference is chaotic amplification of floating-point accumulation over the 200-step training trajectory, not a quality improvement**. The kernel and the torch reference are mathematically equivalent (§22.2 bit-for-bit identical), and **the kernel does not improve model quality**.
- **Statement**: the kernel **does not change quality** (numerically equivalent) and **only improves speed** (train step 1.10–1.17×).

### 22.7 Decoding Step Operator Count Reduction (`experiments/_decode_profile.py`)

| Protocol | aten calls/step | Time per step |
|---|---|---|
| torch path | 1124 | 3.747 ms |
| kernel path | **704 (−37.4%)** | **2.757 ms (26.4% faster)** |

**Note**: this comes from **sequential measurements within the same process**, and it **contradicts** the independent-subprocess A/B of §22.5 (ratios 0.94–0.99). By measurement discipline, **the independent-subprocess result governs** (§22.5), and the numbers here serve only to show that "the kernel does reduce the operator count", not as a speedup conclusion. (Measuring multiple configurations in the same process distorts the readings, see §2.4.)

### 22.8 Where the Gap with Mainstream Frameworks/Peers Lies (after kernelization)

Kernelization does not change the cause of this gap, only the project's own absolute speed. Three points:

① Matrix multiplication: already at the ceiling; the gap is not in implementation but in hardware.

- The train step's `aten::mm` accounts for **33.0%** (§22.4), going through MKL + oneDNN v3.12.
- §13.4 measured actual-size GEMM at **237.9 GFLOPS = about 80% of this machine's peak** ⇒ **no room at the implementation level**.
- **Source of the gap**: the i7-8750H has only AVX2 (no AVX-512/VNNI/AMX), whereas the modern server CPUs llama.cpp targets have AVX-512 + VNNI and GPUs have Tensor Cores. **This is a hardware gap, not a code gap.**

② Element-level operators: kernelization has already cut what can be cut, close to the ceiling.

The fused kernel merges each layer's competitive gating + gating probability + scan into a single pass, reducing the operator count from 1124 → 704/step (§22.7). The train step improves 1.10–1.17× (§22.3/§22.6). **The remaining room is limited**:

- Backward accounts for **88.7%** of the train step (§22.4), and within it `mm` takes the largest share; among element-level backward ops, `_log_softmax_backward_data` 8.6% and `mul` 7.3% already fall within MKL/oneDNN coverage.
- Continued fusion has diminishing marginal returns, and §22.5 already showed that **inference mode has no gain because the tensors are too small**.

③ Relative position versus peers: cross-architecture re-measurement was not done after kernelization, so no change is claimed.

The comparisons in §16.3/§16.4 (sunny80 vs rwkv/h3/mamba/hyena/linear_attn) were **all measured before kernelization**, and **the cross-architecture comparison was not re-run** this time. Therefore:

- No data supports "beat rwkv after kernelization".
- **Reasonable inference** (labeled as inference, not measurement): kernelization gives sunny80 a 1.10–1.17× train step speedup; if peers did not do equivalent optimization, the relative position would improve. **The report draws no conclusion from the inference.**
- Known definite gaps (which kernelization cannot eliminate):
  - **Short sequences**: forward is still inverted at T<384 (§19/§20), with the root cause being operator count.
  - **Long-sequence speed**: ahead for 512<T<2048, and at T=4096 Hyena is 1.9× faster (§16).
  - **Convergence quality**: at 600 steps the early sample efficiency lags H3 (53.93 vs 56.8); without convergence no conclusion is warranted.

④ Source of the gap with llama.cpp / mainstream frameworks: hardware instruction set gap + missing measurement against peers who have not done equally deep optimization, not "the kernel cannot be written well". The kernel chain has been connected on this machine and verified to numerical equivalence; **the next bottleneck is hardware and measurement cost**.

### 22.9 Reproduction

```bash
export SUNNY_W64=F:/devtools/w64devkit
python -X utf8 experiments/_core_e2e.py 256 6 8     # train step A/B
python -X utf8 experiments/_scan_share.py 256       # train step composition profile
python -X utf8 experiments/_step_e2e.py 128,256 1 4 # inference-mode A/B
python -X utf8 experiments/_decode_batch.py 256 1,4,8,16,32 3
python -X utf8 experiments/_decode_profile.py 256 0 # decoding step operator composition
python -X utf8 experiments/_decode_longpath.py 256,512,1024,2048,4096 2  # §22.10 single step vs context
```

**Limitations**: ① the kernel requires `group==8` and `_need_clamp` false (all presets on this machine satisfy this; changing `h_min` may fall back to the torch path); ② it fuses only up to the end of scan, and the `W_c`/`W_s` on the `_emit` side still go through MKL; ③ the training-mode gain was measured only on smalln80, T≤512, single machine 8 threads; ④ the inference-mode gain is negative.

### 22.10 Where the Inference Way Out Really Lies: Single-Step Decoding Cost Growth with Context

§22.5 shows that doing kernel fusion on small tensors (B=1, short T) has its gain eaten by cross-language call overhead, and the inference-mode kernel is off by default. The inference advantage is not in the kernel but in the architecture: SunnyNet single-step decoding is a **fixed-length state O(1)**, while Transformer single-step decoding does attention over the entire history at each step, which is **O(t)**.

Measurement method: an independent subprocess per T; first walk T steps token by token to build a real decoding cache (the walk is not timed, to avoid Transformer being accelerated by parallel prefill), then measure the single-step decoding time at that context length; 1 thread (§22.5 already showed single-threaded beats 8-threaded in decoding mode); 2 rounds taking the median. Script `experiments/_decode_longpath.py`, data `experiments/results/_decode_longpath.json`.

| T | SunnyNet single step ms | Transformer single step ms | sunny/tf |
|---|---|---|---|
| 256 | 2.8716 | 2.4902 | 1.153 |
| 512 | 2.8818 | 3.0298 | **0.951** |
| 1024 | 2.8426 | 3.3265 | 0.855 |
| 2048 | 2.7269 | 4.7904 | 0.569 |
| 4096 | 2.9014 | 6.8544 | **0.423** |

**Reading**:

1. **SunnyNet's single-step cost is flat**: 2.87 → 2.90 ms, with drift within ±5% relative to T=256 (0.950–1.010), consistent with "O(1)".
2. **Transformer's single-step cost rises monotonically with context**: 2.49 → 6.85 ms (2.75×), consistent with "O(t)".
3. **Crossover at T≈512**: at T=256 SunnyNet is 15.3% slower; it overtakes from T=512, and at T=4096 it is **2.36×** faster.
4. The difference from §20.2's early decoding measurement comes from the protocol: §20.2 used 8 threads and included final logits overhead, whereas this uses 1 thread + fixed-length single step, which is what brings out SunnyNet's flatness. The two must not be mixed.

**Conclusion (positioning of the inference way out)**:

- Short-context (T<512) decoding **is not an advantageous scenario**, and kernel fusion has no gain there (§22.5)—both routes are ineffective for short contexts, which is jointly determined by the architecture and execution overhead, not an implementation issue.
- **Long-context decoding is the architecture-level payoff point**: crossover at T≈512, 2.36× faster at T=4096, and the gap continues to widen with T (Transformer's single step ∝t has no upper bound, SunnyNet's is constant). This is an advantage obtainable without any kernel.
- Boundaries of this conclusion: **single sequence, B=1, randomly initialized model, median of 2 rounds**. Real serving is batched decoding, and the ratio will change (as batch grows, Transformer's per-step attention can be amortized in parallel, narrowing the advantage); the batched protocol was not measured. It also excludes engineering optimizations such as KV cache quantization/paging (§20.3).
- **The only comparison object in this section is Transformer**. If decoding mode is compared head-to-head against O(N) peers that are also O(1), SunnyNet is not ahead on the same torch implementation (§23.3 measured: rwkv is 2nd throughout, 1.73–2.16× faster than `sunny` (torch)); with the whole-model C backend it is 1st at every length. The scope in which the "inference advantage" holds is "relative to Transformer", and the torch path excludes peers.

---

## 23. Dimension-by-Dimension Comparison with Six Mainstream Architectures

The six comparison architectures: `h3`, `rwkv`, `mamba`, `hyena`, `linear_attn`, `transformer`. All numbers come from measurements on this machine, with an independent subprocess per cell; where the same cell has different protocols in different sections (batch, threads, whether backward is included), they are listed separately and not compared across protocols.

### 23.1 Scorecard

| Dimension | SunnyNet position | Best | Gap | Protocol |
|---|---|---|---|---|
| **Quality ppl** (1.4M, 600 steps) | 2nd (57.83) | h3 53.93 | behind by 7.2% | §12.5 |
| **Quality ppl** (20M, 200 steps) | **1st** (77.90) | sunnynet | ahead of rwkv by 1.5% | §14.6 |
| **Quality ppl** (100M, 100 steps) | 2nd (109.47) | rwkv 105.66 | behind by 3.6% | §14.8 |
| **Training speed** (L=4 equal stack params) | statistically level with rwkv | rwkv (nominal) | T=256 ratio 1.042, pairing not significant | §16.3–16.4 |
| **Training speed** (L=6 layer-aligned) | **behind** | rwkv | 30–40% slower | §16.5 |
| **Training-mode memory** (T=1024) | 2nd lowest | rwkv 1611MB | 12% higher, 43% lower than TF | §17 |
| **Forward ms/token** (batch=1) | improves with T, 6th→3rd | hyena | never tops the list, still 1.92× slower than hyena at T=4096 | §10.6 |
| **Decoding ms/step** (B=1, torch path) | 7th of 8 | **rwkv** | 1.73–2.16× slower | §23.3 |
| **Decoding ms/step** (B=1, whole-model C runtime) | **1st of 8** | sunnynet | 1.02–1.15× faster than rwkv | §23.3 |
| **Long-distance retrieval** (code domain) | level with TF | tie | far bucket ngram=4 is 0.1208 for both | §8.2 |
| **Long-distance retrieval** (natural language) | near 0 | — | no advantage | §8.3 |
| **Cross-distribution generalization** | 1st on Chinese new domain / 2nd on English | sunnynet | OOD drop 1.480, lowest bracket | §18.1 |

**In one sentence**: SunnyNet is currently competitive in quality (20M) and training memory, statistically level with rwkv in training speed (only at L=4), and **behind hyena in forward speed** and behind rwkv on the same-torch decode path; the only table it tops in decoding comes from the whole-model C backend, which is not at the same implementation level as the torch path—"comprehensive superiority" has no support in any single column of data.

### 23.2 Five Protocol Flaws in the Earlier Comparisons

| # | Flaw | Impact | Fix |
|---|---|---|---|
| 1 | The five O(N) peers **have no single-step decoding interface** | Decoding mode was never measured against peers, only a head-to-head with Transformer in §22.10 | This round implements `init_state`/`step` for the five blocks, numerical consistency maxdiff ≤ 4e-05 (`_step_equiv.py`) |
| 2 | Speed comparisons were mostly **batch=1 pure forward** | Completely different ranking from training mode | §17 already added training mode; §23.3 adds decoding mode |
| 3 | Training-mode speed did **not enforce layer alignment** | §16's "level" rests on SunnyNet using fewer layers | Declared in §16.5; 30–40% behind after layer alignment |
| 4 | Quality comparisons had **few steps, no convergence, single seed** | Can only be read as early sample efficiency | Noted in each section, not used as a final quality ranking |
| 5 | Some speed data **included external load** | Single cells inflated by up to 2× | Independent subprocess per cell already implemented |

### 23.3 Eight-Path Comparison in Decoding Mode

Protocol: an independent subprocess per (path, T); first walk T steps token by token to build a real decoding state (not timed), then measure the single-step decoding ms at that context length; single-threaded (§22.5 already showed a tendency for single-threaded to beat multi-threaded in decoding mode); 2 rounds taking the median. Stack parameters aligned to 365,844. The eight paths include a whole-model C backend of the same specification (`sunnyC`, `d80` blob, one `sn_step` per token), listed separately from the torch path. Script `experiments/_decode_seven.py`, data `experiments/results/_decode_seven.json`.

![Decode latency vs sequence length](../figures/perf_decode.png)

ms/step (lower is better):

| Path | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| **`sunnyC` (whole-model C)** | **1.560** | **1.630** | **1.547** | **1.518** | **1.647** |
| `rwkv` | 1.779 | 1.879 | 1.673 | 1.675 | 1.674 |
| `transformer` | 2.473 | 3.405 | 5.236 | 4.056 | 6.962 |
| `h3` | 2.643 | 2.944 | 2.568 | 2.753 | 2.874 |
| `mamba` | 3.160 | 3.063 | 2.845 | 3.171 | 3.897 |
| `linear_attn` | 3.467 | 4.207 | 4.019 | 3.101 | 3.636 |
| `sunny` (torch) | 3.778 | 3.973 | 3.166 | 3.624 | 2.889 |
| `hyena` | 4.338 | 3.310 | 3.281 | 3.693 | 3.046 |

Per-cell ranking (1=fastest, of 8):

| Path | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| `sunnyC` | **1/8** | **1/8** | **1/8** | **1/8** | **1/8** |
| `rwkv` | 2/8 | 2/8 | 2/8 | 2/8 | 2/8 |
| `h3` | 4/8 | 3/8 | 3/8 | 3/8 | 3/8 |
| `transformer` | 3/8 | 6/8 | 8/8 | 8/8 | 8/8 |
| `mamba` | 5/8 | 4/8 | 4/8 | 5/8 | 7/8 |
| `linear_attn` | 6/8 | 8/8 | 7/8 | 4/8 | 6/8 |
| `sunny` (torch) | 7/8 | 7/8 | 5/8 | 6/8 | 4/8 |
| `hyena` | 8/8 | 5/8 | 6/8 | 7/8 | 5/8 |

Ratio relative to `sunnyC` (>1 = slower than the whole-model C backend):

| Path | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| `rwkv` | 1.140 | 1.153 | 1.081 | 1.104 | 1.016 |
| `transformer` | 1.585 | 2.089 | 3.384 | 2.672 | **4.226** |
| `h3` | 1.694 | 1.806 | 1.660 | 1.813 | 1.744 |
| `mamba` | 2.026 | 1.879 | 1.839 | 2.089 | 2.366 |
| `linear_attn` | 2.222 | 2.580 | 2.597 | 2.043 | 2.207 |
| `sunny` (torch) | 2.422 | 2.437 | 2.046 | 2.387 | 1.754 |
| `hyena` | 2.781 | 2.030 | 2.120 | 2.432 | 1.849 |

**Reading**:

1. **Transformer's O(t) is reproduced**: 2.473 → 6.962 ms (T grows 16×, single-step cost grows 2.82×); against the whole-model C backend it is **4.23×** slower at T=4096. Same direction as §22.10 (2.36×); the ratio is larger here because the C backend's baseline is lower than §22.10's torch single step.
2. **The whole-model C backend's single-step cost is flat**: 1.560 → 1.647 ms, drift of −3%~+6% relative to T=256, and 1st at every length. Consistent with §22.10's ±5%.
3. **On the same torch implementation SunnyNet does not match rwkv**: `sunny` (torch) ranks 7/7/5/6/4, and rwkv is 2nd at all five lengths, **1.73–2.16×** faster than `sunny` (torch). Its wkv recurrence is pure element-wise `p*s+k*v`, with no scan and no log/exp chain.
4. **The whole-model C backend reverses the ordering**: `sunnyC` is 1st at all five lengths, **1.02–1.15×** faster than rwkv. The reversal comes from eliminating roughly 1100 aten calls per step (§24.7), **not from a change in architectural operator count**—the two paths must be quoted separately.
5. **hyena's FFT long convolution is not worthwhile in decoding mode**: its whole-segment forward is the fastest of the seven architectures (§10.6), but in decoding mode it is 8th/6th; the constant factor of rebuilding the FFT per token is far higher than that of its scan-class peers.

**Conclusion**:

- **The decoding advantage comes from two distinct sources that must not be conflated**. On the same torch implementation, SunnyNet is faster than Transformer (and the longer the more so) but 1.73–2.16× slower than rwkv; switching to the whole-model C backend drops the single step to 1.52–1.65 ms, 1st at every length, 1.02–1.15× faster than rwkv. §22.10's "inference advantage" holds only in the comparison "against Transformer" and does not cover the rwkv column.
- **rwkv remains the opponent to catch on the torch path**. Its advantage comes from **a low operator count**—its single step has only one element-wise chain `p*s+k*v`, whereas SunnyNet's single step contains competitive gating (3 sigmoids), gating probability, scan, and `W_c`/`W_s` feedback, so its element-wise operator count is several times larger. This is the same root cause as the training-mode "the bottleneck is operator count, not FLOPs" (§13.3/§16.5); the C backend overtakes precisely because it folds that operator chain into one `sn_step` call.
- Boundaries: B=1, random initialization, median of 2 rounds, no batched decoding and no KV cache quantization/paging. `sunnyC` and the other seven are not at the same implementation level (C kernel vs PyTorch); comparisons across that column must carry the caveat.

### 23.4 Merging the Root Cause of the Gaps

Among the ten dimensions, those where SunnyNet is behind or merely level are: decoding speed (behind rwkv on the same torch implementation), forward speed (behind hyena), L=6 training speed (behind by 30–40%), 100M quality (behind rwkv by 3.6%), 1.4M quality (behind h3 by 7.2%), natural-language long-distance retrieval (no advantage). These **converge to the same root cause**: **element-wise operator count**.

| Evidence | Number |
|---|---|
| Single-block forward hotspots are dominated by `mm`/`mul`/`add`/`var`/`cat`, with no single dominant operator | §13.3 |
| Train step 4524 aten calls/step | §22.4 |
| Fewer layers + wider (equal stack parameters) trains 12–17% faster, ppl −10.9% | §14.3 |
| After layer alignment to L=6, equal-budget narrow configurations are instead 30–40% slower | §16.5 |
| In decoding mode rwkv wins with the single chain `p*s+k*v` | §23.3 |

So there is only one solution direction: **reduce the operator count**, not widen, adjust batch, or swap the matrix library.

### 23.5 Dimension-by-Dimension Attack Route (Reachability and Hard Constraints)

| Dimension | Goal | Actionable step | Expectation and constraints |
|---|---|---|---|
| **Decoding speed** | catch rwkv | The fused step kernel is implemented (§22.5), but FFI eats the gain at B=1; **eliminating FFI** (C extension / custom torch operator) could cut 10+ aten calls per step | reachable, but must bypass ctypes; the ceiling is catching rwkv, not passing it |
| **Forward speed** | catch hyena | hyena's FFT constant factor is extremely low, and SunnyNet's element-wise scan is hard to catch up with in batch=1 forward | **limited by architecture**: short forward is a structural disadvantage of scan-class architectures |
| **Training speed** | still level after layer alignment | Fewer layers + wider already done (§14.3); the next step is **operator-level fusion** (bringing `W_c`/`W_s`/`_emit` into the kernel too) | reachable, but with diminishing returns (fusion upper bound only 8.8% in §14.7) |
| **Quality 1.4M** | beat h3 | h3's gap (7.2%) is within 600-step single-seed noise; multiple seeds + more steps are needed to decide | insufficient data, not an architectural disadvantage |
| **Quality 100M** | beat rwkv | Behind by 3.6%, and lr was not tuned per architecture | requires an lr search then re-measurement; direction undetermined |
| **Natural-language long distance** | beat TF | The state `H_max=8192` only guarantees "not forgotten", not "retrievable"; the state design needs changing | **requires an architecture change**, beyond parameter tuning |
| **Batched decoding** | untested | As batch grows, Transformer's per-step attention can be amortized in parallel, narrowing SunnyNet's advantage | must be tested, otherwise §23.3 is incomplete |
| **Direct benchmark against llama.cpp** | not done | llama.cpp has hand-written SIMD kernels, and this machine already has w64devkit to compile a comparison | reachable, but requires building a GGUF-side implementation |

**Hard constraints (insurmountable on this machine)**:

- Hardware reaches only AVX2 (i7-8750H, no AVX-512/VNNI/AMX); quantization is a negative optimization on this machine (bf16 520× slower), and low precision has no native support either (§CPU headroom).
- On the GPU side there is no nvcc/CUDA toolkit/triton, so no CUDA kernel can be compiled (§15.5).
- 0.5B is not instantiable on CPU fp32 (§15.3).

⇒ Among the routes above, **the one that can land is "reduce the operator count"**; the rest are limited by hardware or structural architecture constraints. The magnitude obtainable on that one route is **train step 1.10–1.17×** (already in hand) + the extra decoding-mode gain if FFI is eliminated (not measured)—**not enough for "comprehensive superiority" over the six**.

### 23.6 Reproduction

```bash
python -X utf8 experiments/_step_equiv.py              # single-step interface consistency of the five peers
python -X utf8 experiments/_decode_seven.py 256,512,1024,2048,4096 2  # §23.3 decoding comparison
```

---
## 24. Generic Recurrent C Runtime

A C inference kernel independent of PyTorch, directory `sunnynet/runtime/`, depending only on ctypes and a C compiler.

> Scope note: the Mamba-2 / DeltaNet / GLA / HGRN2 measurements from §24.12 onward are a
> historical record. As of 2026-10-26 the supported architecture list is RWKV-7 and Mamba-2
> only; the code and CLI dispatch for GLA / DeltaNet / HGRN2 have been removed (all three
> carry weights ≥1.37 GB and their C speedup is already at the bandwidth ceiling). The data
> here is retained as the rationale for that decision and as reproducible evidence.

### 24.1 Unified recurrence form

```
s_t = a_t * s_{t-1} + b_t            per-channel independent   (SN_FMA)
s_t = a_t * s_{t-1} + (1 - a_t) * u_t gated form               (SN_GATED)
```

Any first-order linear state recurrence is one of these two forms. Architectures solving this form differ in parameterization; the recurrence operator itself is the same. The kernel vectorizes along channels (AVX2/FMA), serial in the T dimension; parallel in the `B` dimension via OpenMP.

### 24.2 Cross-architecture mapping verification

Each architecture's recurrence is manually mapped to kernel parameters and compared element-wise against its own PyTorch recurrence.
`experiments/_unified_ir_map.py`, B=2, T=128, N=128, fp32, random weights.

| Architecture recurrence form | Mapping | max\|diff\| |
|---|---|---|
| RWKV-4 wkv | `a=p`, `b=k*v` (FMA) | 9.5e-07 |
| Mamba / S6 SSM | `a=decay`, `u=bu` (GATED) | 2.4e-07 |
| H3 diagonal SSM | `a=decay`, `u=v` (GATED) | 1.2e-07 |
| Linear attention accumulated KV | `a=1`, `b=k v^T` (FMA) | 0.0 |
| SunnyNet rolling state | `a=p`, `u=u` (GATED) | 1.9e-06 |
| S4 diagonal SSM | `a=A_bar`, `b=B_bar x` (FMA) | 7.2e-07 |

Hyena is a long-convolution (FIR) recurrence, not a first-order state recurrence, and requires a separate convolution backend; it is not included here.

### 24.3 Kernel self-check

`experiments/_recur_equiv.py`, compared against an fp64 step-by-step recurrence reference:

| Case | max\|diff\| |
|---|---|
| FMA arbitrary coefficients | 2.4e-07 |
| GATED (1-a)u | 1.2e-07 |
| matches `ops.gated_scan` | 5.7e-06 |
| backward da / db / ds0 | 0.0 |
| GATED backward du | 0.0 |
| single step vs full segment | 4.8e-07 |

### 24.4 Speed: against the PyTorch vectorized baseline

`experiments/_runtime_bench.py`, each cell in an independent subprocess, single thread, B=1, median of 10.
Baseline B is `ops.gated_scan` (torch cumsum closed-form chunked scan), not a Python loop.

| T | N | Mode | Vectorized ms | Kernel ms | vs vectorized |
|---|---|---|---|---|---|
| 256 | 128 | FMA | 0.391 | 0.043 | 9.08× |
| 256 | 128 | GATED | 0.831 | 0.043 | 19.29× |
| 1024 | 128 | FMA | 1.256 | 0.132 | 9.54× |
| 4096 | 128 | GATED | 8.500 | 0.806 | 10.55× |
| 1024 | 320 | GATED | 3.677 | 0.378 | 9.73× |
| 4096 | 640 | FMA | 26.944 | 7.574 | 3.56× |
| 4096 | 640 | GATED | 32.822 | 8.286 | 3.96× |

Median 9.90×, range 3.56–19.29×; maximum numerical difference across the three branches 7.63e-06.
At N=640 it drops to 3.6–4.0× (under a single thread memory bandwidth becomes the limit).

The multiple against a Python per-token loop (10.9–342.8×) is not stated as a result—that is not a real deployment baseline.

### 24.5 Dynamic range of recurrent state quantization

`experiments/_state_quant.py`, the p/u sequence from the layer-0 forward of a real ckpt, B=1, T=2048,
symmetric INT8. Compared against "unbounded accumulation" (`s += u`, simulating unnormalized KV accumulation).

| Mode | \|s\| at 1/8 | \|s\| at end | Drift | Single-quantization relative error |
|---|---|---|---|---|
| Bounded recurrence (real p) | 1.773 | 1.883 | 1.06× | 3.9e-03 |
| Unbounded accumulation | 466.8 | 3708.7 | 7.94× | 3.9e-03 |

Fixed scale (scaled by the sequence midpoint absmax, quantizing the final state):

| Mode | \|s\| at end | Overflow channels | Overflow ratio | Relative error |
|---|---|---|---|---|
| Bounded recurrence | 1.88 | 1 / 128 | 0.78% | 4.0e-02 |
| Unbounded accumulation | 3708.7 | 7 / 128 | 5.47% | 5.0e-01 |

**Conclusion**: The convex-combination state magnitude goes from 1.773 to 1.883 over the final 1/8 segment, drift 1.06×; unbounded accumulation drifts 7.94× over the same interval.
When the scale is not recomputed, the bounded state relative error is 4.0e-02, unbounded 5.0e-01. The two differ by 12.5×.
Per-channel single block (block=0) and chunking of 32 give the same error magnitude (3.9e-03 vs 3.5e-03).

**Boundary**: Only layer 0, a single ckpt (1.4M), a single random token sequence were measured; no ppl comparison of end-to-end quantized inference;
"round-trip quantization at every step" (relative error 4.9e-01) is not a real usage and is included only as a reference, not as a conclusion.

### 24.6 End-to-end decoding: no speedup after kernel integration

`experiments/_runtime_e2e.py`, same ckpt / prompt / thread count, switching only the recurrence operator implementation
(`SUNNY_KERNEL=0` uses torch, `=1` uses the AVX2 kernel); prefill fills T to build state, then the median of 30 single steps is measured.

| Threads | T | torch ms/step | Kernel ms/step | Ratio |
|---|---|---|---|---|
| 1 | 256 | 3.1303 | 3.0587 | 1.023× |
| 1 | 1024 | 3.0464 | 2.8416 | 1.072× |
| 8 | 256 | 4.3067 | 4.8008 | 0.897× |
| 8 | 1024 | 4.1083 | 3.8295 | 1.073× |

The four cells are 1.023 / 1.072 / 0.897 / 1.073, **inconsistent in direction** (8 threads at T=256 is 10.3% slower).
⇒ End-to-end decoding speedup must not be claimed; consistent with the inference-state conclusion in §22.5.

Cause: the kernel vectorizes along N and is serial in T, while on the model side the T and B dimensions are tiny (T=1, B=1),
and the cost of cross-language calls and state movement is the same order as a single operator (§22.5 measures a ctypes call with no
arguments at 0.28µs, 8 arguments at 2.02µs, against a torch single operator at 2.42µs). The kernel's 9.90× gain comes from eliminating the
inter-chunk serial dependency and the `exp(-P)` intermediate tensor, which holds in **long-sequence scans across segments**, but not in B=1 single-step decoding.

A B=1 single sequence is also not the Runtime's target scenario (§23.3: on the same torch implementation the decoding-state best is rwkv).

### 24.7 Whole-model C decoding: moving the entire forward into C

The negative result in §24.6 is localized to "cross-language calls + state movement", so the way out is to cross the language boundary
only once per step: put embedding lookup, per-layer norm / depthwise-convolution history / gated competition / state recurrence / readout,
final normalization, and the LM head all into C, one `sn_step` call per step. Weights are exported by `experiments/_export_runtime.py` into
a packed fp32 blob: matmul weights pre-transposed to `(in, out)` for GEMV (a single AVX2 load covers 8 output rows),
`circ ∘ W_s` folded into a single matrix, `p` clamping decided at construction time (consistent with §16).

Numerical consistency `experiments/_runtime_model_equiv.py` (1.4M ckpt, 48 tokens step by step):

| Metric | Result |
|---|---|
| Single-step logits max\|diff\| | 1.24e-05 |
| argmax agreement | 48 / 48 |
| prefill last token vs torch full-sequence forward | 3.81e-06 |

End-to-end latency `experiments/_runtime_decode.py` (B=1 single sequence, single thread, each cell in an independent subprocess,
fill prefill first then measure single steps, 3 interleaved rounds):

| prefill T | torch ms/step | C ms/step | Ratio | Per-round wins |
|---|---|---|---|---|
| 256 | 3.796 | 2.487 | 1.53× | 3/3 |
| 1024 | 3.802 | 1.851 | 2.05× | 3/3 |
| 2048 | 4.309 | 2.189 | 1.97× | 3/3 |

The median ratio agrees with the per-round sign test (all three cells 3/3), so a real improvement can be judged. The source of the speedup is
eliminating about 1100 aten calls per step (profile: `aten::mm` 16.4% / `aten::mul` 10.2% / `aten::select` 8.5% /
`aten::cat` 8.0%), not operator fusion itself.

**Boundary**: single-thread B=1 metric. With the CLI default of 8 threads, C is 4.40 vs torch 4.56 ms/tok (a single 24-token
reading, not multiple rounds), and the gap is flattened by scheduling—parallelism over small tensors at B=1 is a net loss (§20.2). The C
backend's advantage is at single thread and larger batches; multi-thread B=1 must not be linearly extrapolated from the table above.

**One real bug localized during development (fixed)**: mingw's `fopen` uses the ANSI code page, so a Windows path containing
Chinese characters cannot open the weight file; the C side was changed to `_wfopen` + `MultiByteToWideChar(CP_UTF8)`.

### 24.7b Memory-access rework of whole-model C decoding

`p100mw` (105.36M parameters, fp32 blob 467.6 MB) takes 80.2 ms per step on the C backend, not on the same order as 20M-class models
(the 1.4M ckpt in §24.7 is about 2 ms/step). The cause localizes not to the operators but to the **weight layout**.

**First measure the memory-access ceiling.** Single-step decoding must read all weights at least once, so the lower bound on time is
`blob bytes / achievable bandwidth`. `experiments/bench_bw.c` sequentially reads 419 MB with AVX2
(read only, no writes, best of 5 rounds):

| Threads | Sequential read bandwidth |
|---|---|
| 1 | 15.70 GB/s |
| 2 | 21.57 GB/s |
| 4 | 23.46 GB/s |
| 8 | 25.31 GB/s |

In the same period `experiments/_bench_runtime_bandwidth.py` measured a C runtime single step at 75.75 ms,
effective bandwidth only **6.17 GB/s** — 1.08× the single-thread lower bound, 1/4 of the multi-thread lower bound.
(Note: numpy's `a.sum()` measures only 6.69 GB/s and underestimates the machine's bandwidth, the source of an early misjudgment;
the bandwidth comparison in that script is therefore only an order-of-magnitude reference.)

**The cause is the access pattern.** The original GEMV implementation blocks along output channels, and the inner loop over `in` strides across with an `out` stride:

```c
for (; j + 8 <= out; j += 8) {
    const float* row = Wt + j;
    for (int i = 0; i < in; ++i)
        acc = _mm256_fmadd_ps(b, _mm256_loadu_ps(row + (size_t)i * out), acc);
}
```

`experiments/bench_gemv.c` micro-benchmarks three implementations (p100mw's head `D→vocab`, 46.1 MB):

| Implementation | 1 thread | 4 threads | 8 threads | vs original |
|---|---|---|---|---|
| Original stride layout | 4.11 GB/s | 8.66 | 8.39 | 1.00× |
| Original stride + OpenMP | — | — | — | 2.11× |
| **Row-major `(out,in)`** | 18.91 | 30.91 | 28.86 | **4.61×** (1 thread) / **7.53×** (4 threads) |

Row-major makes each output row's weights contiguous, so one load hits adjacent cache lines; parallelism splits along output
channels, with no dependency between rows. **Adding OpenMP alone without changing the layout gives only 2.1×; the layout change is the main term.**

**Change and compatibility.** The export format is unchanged (still `(in,out)`, published blobs need no re-export):
after `sn_load` reads in, it flips in place in memory to row-major; the element count is unchanged so subsequent section offsets are
unaffected. GEMV is changed to contiguous reads parallelized along output channels.

**Thread count uses physical cores, not logical cores.** Decoding is memory-access bound, and hyperthreading does not increase
bandwidth; measured 6 physical cores 26.69 ms vs 12 logical cores 36.05 ms (26% difference). A
`sn_recommended_threads()` was added on the C side: on Windows it uses `GetLogicalProcessorInformation` to count
`RelationProcessorCore`, on Linux it parses `/proc/cpuinfo` and deduplicates `(physical id, core id)`.

**The optimal thread count is set by weight size, not a fixed value.** Sweeping threads per blob tier on the same
machine (`experiments/_decode_cthreads_size.py`, an independent process per tier):

![Decode thread policy](../figures/perf_threads.png)

At ≤23 MB (d80–d256) **a single thread is optimal**: the weights fit in cache and parallelism only adds synchronization
overhead (d128 single thread 1.70 ms vs 4 threads 2.96 ms). From ≥39 MB (d384 up) the weights exceed cache capacity and the
step becomes memory-access bound, where **physical cores win** (d768 single thread 9.84 ms vs 6 threads 7.65 ms). The
boundary sits near 32 MB, so `threads_for_blob()` picks by blob byte count rather than always taking the core count.

**A/B comparison** (`experiments/_ab_runtime_opt.py`: the pre-optimization library is checked out of `git show` and compiled separately,
same blob, same 32-token sequence, 5 interleaved head-to-head rounds):

| Metric | Result |
|---|---|
| logits max\|diff\| | **2.766e-05** |
| argmax agreement | **32 / 32** |
| bitwise identical | False (parallel summation order changes rounding, as expected) |
| Before optimization | 76.60 ms/token, 6.10 GB/s |
| After optimization | **25.72 ms/token, 18.18 GB/s** |
| Speedup | **2.98×**, per-round **5/5** wins |

Numerical consistency against torch still meets the existing tolerance (`_verify_runtime_opt.py` using the
`forward_step` single-thread metric: max\|diff\| 7.15e-06, argmax 32/32).
Thread-count sweep (`experiments/_bench_runtime_threads.py`, median of 5 rounds):

| Threads | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| ms/token | 56.12 | 40.37 | 32.57 | 30.35 | **26.69** | 36.05 | 36.03 |

**Boundary**: 2.98× is the single-step decoding multiple on **this machine (i7-8750H dual-channel DDR4-2667)** and
only shows that this implementation's memory-access efficiency went from 6.1 to 18.2 GB/s. 18.18 GB/s is still below the
25.31 GB/s ceiling of `bench_bw.c`; the difference comes from the per-step reduce/normalization and other non-streaming accesses.
Training or prefill must not be extrapolated from this (prefill runs the same `sn_step` loop, with no batch parallelism).
The 75.75→76.60 difference for the original implementation comes from different measurement scripts (the former includes the first `_load` call) and is not stated.

### 24.7c Weights stored as fp16: halving the bytes read per step

After §24.7b the single step is still memory-access bound: the pure GEMV loop in `bench_layer.c` (41 calls, no serial section)
is 14.67 ms at 6 threads (31.87 GB/s), versus 21.6 ms in a session context — there is a state dependency between adjacent calls,
and each `gemv` builds its own parallel region, spreading the heat. Since the kernel has reached the pure sequential-read ceiling, only one lever remains:
**read fewer bytes per step**.

The 5 GEMV matrices and the LM head are truncated to fp16 at load time; small parameters (layernorm's g/c, conv taps,
delta, scalars, embed) stay fp32:

| | fp32 | fp16 |
|---|---|---|
| Large matrix weights | 421.0 MB | 210.5 MB |
| Actual bytes read per step | 467.6 MB | about 222.3 MB |

The kernel `gemv_h` reads 16 halves (32 B) per `_mm256_loadu_si256`, splits into two `__m128i` each
converted by `_mm256_cvtph_ps` into 8 floats for FMA — keeping the same 32 B load width as fp32.
`bench_f16b.c` compares load widths (same fp16 weights):

| Threads | fp32 | f16 128-bit load | **f16 256-bit load** | 256-bit/fp32 |
|---|---|---|---|---|
| 1 | 24.36 | 17.45 | **15.06** | 1.62× |
| 2 | 19.66 | 11.43 | **12.81** | 1.53× |
| 4 | 15.24 | 10.77 | **9.99** | 1.53× |
| 8 | 15.36 | 9.88 | **9.08** | 1.69× |

The 128-bit load loses load width and is actually slower than 256-bit at single thread; the two loads are numerically identical
(both have max\|diff\| vs fp32 of 1.805e-04), so 256-bit is used.

**The error is bounded, but not bitwise identical.** `experiments/verify_h.c` (direct C, same blob same sequence):

| Metric | Result |
|---|---|
| worst logits max\|diff\| | 4.181e-03 |
| Relative deviation | 7.42e-04 |
| **argmax agreement** | **64 / 64** |
| final-state per-layer state max\|diff\| | 1.226e-02 |

fp16 has only a 10-bit mantissa, weight relative error about 5e-4, so a logits relative deviation of 7e-4 is a necessary result within that
magnitude. The deviation accumulates with step count but stays below argmax (64/64 on this sequence).

**A/B comparison** (`experiments/_ab_fp16.py`: same process, same blob, 11 interleaved
head-to-head rounds, 6 physical cores):

| Metric | fp32 | fp16 |
|---|---|---|
| median ms/token | 24.67 | **18.60** |
| min ms/token | 22.20 | **17.26** |
| Effective bandwidth (in fp32-equivalent bytes) | 18.95 GB/s | 25.14 GB/s |
| **Per-round pairing** | — | **11/11 all wins** |

Speedup **1.33×** (1.29× on the min metric), per-round paired difference median 6.07 ms and all positive.

**fp16 is no longer memory-access bound.** fp16 actually reads 222.3 MB, and 18.6 ms corresponds to only **12.0 GB/s**,
far below fp32's 19 GB/s — the bottleneck moves to the `cvtph_ps` conversion and instruction issue. Thread scaling
(`experiments/_fp16_threads.py`) is therefore insensitive for both:

| Threads | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| fp32 median ms | 38.68 | 31.17 | 25.75 | 26.24 | 27.39 | 27.25 |
| fp16 median ms | 28.48 | 23.98 | 21.36 | **19.95** | 20.71 | 20.60 |
| Speedup | 1.36× | 1.30× | 1.21× | 1.32× | 1.32× | 1.32× |

**Usage**: `--backend c --half` (Python side `SnModel(..., half=True)`).
fp32 is still the default, bitwise unchanged. The CLI measured decode at 20.96 ms/tok under `--half`.

**Boundary**: ① 1.33× is a single-step fraction on this machine and must not be extrapolated to prefill or training;
② end-to-end is lower than operator-level (`bench_f16b.c` 4 threads 1.53×) — consistent with existing discipline,
operator-level microbenchmark advantages are diluted end-to-end; ③ the 7.4e-04 relative deviation will cause occasional divergence
in long-sequence greedy decoding, and 64/64 on this sequence is just a sample; use fp32 when bitwise reproduction is needed;
④ the weight file is still a fp32 blob, fp16 is converted only at load time, and disk usage is unchanged
(reducing disk usage would need a separate export, not done).

### 24.8 Command-line inference

`sunnynet/cli.py`, `python -m sunnynet.cli --ckpt <path> --prompt "..."`.
Loads ckpt → tokenizer → autoregressive decoding, `--stats` outputs prefill/decode timing.
`--backend c` uses whole-model C decoding (requires exporting a blob first with `experiments/_export_runtime.py`).
`--gguf <path>` uses external GGUF weights (see 24.9).

### 24.9 GGUF read-only and external weights (RWKV-7)

The `--gguf` path reads external GGUF weights and generates token by token, without going through PyTorch. The first one made to
work is the official RWKV-7 191M (BlinkDL/rwkv-7-world's `.pth` converted to GGUF via llama.cpp, bf16).
Tensor naming and recurrence form are taken from llama.cpp's `src/models/rwkv7*.cpp`; the reference math is taken from
BlinkDL/RWKV-LM's `RWKV-v7/rwkv_v7_numpy.py`.

**（1）Weight alignment** (`experiments/_gguf_align.py`)
The GGUF dequantized result is compared tensor by tensor with the official `.pth`, all `max|diff| = 0` (lossless bf16 restoration).
The LoRA decomposition matrices (w1/w2, a1/a2, g1/g2, v1/v2) are stored in GGUF as `(out, in)`, opposite to `.pth`,
while the rest of the matrices are in the same direction. The vocabulary stores strings as `\xNN` escaped literals, which must be unescaped to bytes.

**（2）Numerical equivalence** (`experiments/_rwkv7_gguf_equiv.py`)
The GGUF-driven implementation and an independent reference implementation of the official `.pth`, compared on per-token single-step logits:

| Metric | Result |
|---|---|
| Single-step logits max\|diff\| | 4.79e-06 |
| argmax agreement | 12 / 12 |

**（3）Generation** (`experiments/_rwkv7_gguf_bench.py`, temperature 0.7 / top-k 20 / seed 42)
English factual continuation, Chinese continuation, and Python code completion are all coherent:

```
The Eiffel Tower is located in the city of
-> Paris, France. The Eiffel Tower was designed by Gustave Eiffel in 1887 as a monument to
def fibonacci(n):
-> \n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)
```

**（4）Two boundaries that must be recorded**
- `rwkv7.py` is a **numpy reference path**, not a C kernel. It proves "the Runtime can read external GGUF and compute correctly",
  decoding about 388 ms/step — **this number is not stated as a speed result**.
- RWKV-7's state is a **per-head matrix** `S ∈ R^{64×64}`, updated as
  `S_t = S_{t-1}⊙w_t − (S_{t-1}k_t)k_tᵀ + v_t k_tᵀ`, which contains a rank-1 removal term,
  **not** the element-wise `s = a⊙s + b` (the two forms in 24.1). For the unified IR to cover it, a matrix-valued
  recurrence (outer product / rank-1 correction) level must be added beyond the element-wise one. This is a correction to the existing narrative of "two unified forms".

**（5）Bugs localized during development**
- The rank-1 removal term must use the **undecayed** `S`: in `S⊙w − (S k)kᵀ + v kᵀ` the two terms share the same `S`.
  Writing it as decay first then compute the removal term is bitwise correct at step0 (S=0) but deviates by 0.5+ from step1 on, very easy to miss.
- LoRA decomposition direction: `x @ W1 @ W2`, not `W1 @ W2 @ x`.

### 24.10 Whole-model C kernel for RWKV-7

The numpy path in §24.9 is only used to verify weight correctness, decoding about 388 ms/step. This section moves RWKV-7's
**entire single-step forward** into C (`sunnynet/runtime/rwkv7.c`, one call per step): embedding lookup,
per-layer norm, time-mix (`r/k/v/o` projections + four LoRA groups `w/a/v/g` + `tanh`/`sigmoid`),
per-head matrix state recurrence (with rank-1 correction), group-norm, the `r_k` term, channel-mix,
final normalization, LM head. Weights are exported from GGUF by `experiments/_export_rwkv7.py` into a packed fp32
blob (matrices stored as `(in, out)` for GEMV; the v-LoRA rank differs per layer, with a per-layer rank table recorded in the header).

**（1）Numerical consistency** (`experiments/_rwkv7_c_equiv.py`, compared step by step with the numpy reference in `rwkv7.py`)

| Metric | Result |
|---|---|
| Single-step logits max\|diff\| | 7.25e-05 |
| argmax agreement | 12 / 12 |
| 120-step long-range argmax agreement | 120 / 120 |

**（2）Decoding speed** (`experiments/_rwkv7_c_bench.py`, single sequence, single step after state is built,
each cell in an independent subprocess, 5 interleaved rounds, median of 60 steps)

| Implementation | ms/step | Relative to numpy | Per-round wins |
|---|---|---|---|
| numpy reference | 366.36 | 1.00× | — |
| C kernel 1 thread | 67.92 | 5.39× | 5/5 |
| C kernel 4 threads | 52.26 | 7.01× | 5/5 |
| C kernel 8 threads | 48.88 | 7.49× | 5/5 |

The C kernel at 8 threads is another 1.39× faster than 1 thread (5/5). **Boundary**: the absolute single-thread single-cell time has
large variance (50–84 ms), fluctuating with machine load, but the ratio and the per-round sign test agree in direction; only a
single-sequence single-thread metric is measured, batch is not. GEMV's column-block parallelism does not change the summation order within each column, so 8 threads and 1 thread are bitwise identical.

**（3）Bugs localized during development**
- Weight layout inconsistent with the C side: the exporter decided LoRA tensors with `name.split(".")[-1]`, getting `"weight"`
  rather than the module name, so 8 LoRA matrices were not transposed and the `w1@w2` dimension collapsed to `(64,64)`. Exposed when the per-tensor float
  counts did not match (difference 21,120,000).
- `w`'s LoRA has a `tanh` in the middle (`exp(-σ(tanh(xw W1) W2 + w0)/√e)`), and **cannot be folded into a single matrix**;
  `a`/`v` have no interposed layer but keep the decomposition for uniformity.

**（4）CLI**
`--gguf-backend c` uses the C kernel (`sunnynet/cli.py` `_main_gguf_c`), requiring running
`experiments/_export_rwkv7.py` first to export a blob.

### 24.11 INT4 weight quantization

`sunnynet/runtime/quant.c`. Grouped symmetric INT4: every 64 consecutive weights share one fp32 scale,
quantization range [-8, 7], two values per byte (low nibble of byte k = the k-th in the group, high nibble = the k+32-th).
The quantized GEMV also quantizes activations to INT8 with the same grouping, using `maddubs` for the dot product:

```
Σᵢ (qᵢ − 8) xᵢ = Σᵢ qᵢ xᵢ − 8 Σᵢ xᵢ
```

Weights stay unsigned [0,15] and go directly into the first operand, activations go signed into the second operand.
The whole-model C kernel (`rwkv7.c`) mixes fp32 / INT4 by a per-matrix flag; vectors and embedding tables stay fp32
(embedding is a lookup, not a multiply-add).

**（1）Primitive correctness** (`tests/test_quant_i4.py`)
The quantized GEMV's integer path is **bitwise identical** to an explicit integer sum; the dequantization error does not exceed the block's
`scale/2`; relative error against fp32 GEMV is ~0.11 (within the 4-bit expectation).

**（2）Matrix-level speed** (`experiments/_quant_i4_bench.py`, single thread, each cell in an independent subprocess, median of 3 rounds)

| Matrix (out×in) | fp32 ms | INT4 ms | Speedup |
|---|---|---|---|
| 768×768 | 0.152 | 0.147 | 1.03× |
| 3072×768 | 0.308 | 0.174 | 1.77× |
| 768×3072 | 0.305 | 0.180 | 1.70× |
| 65536×768 | 7.036 | 1.659 | 4.24× |

Small matrices show no gain (decoding is bandwidth-limited and the matrix itself is already in L1/L2); the 4.24× for the large matrix comes from the read volume dropping to 1/8.

**（3）End-to-end** (`experiments/_rwkv7_int4_e2e.py`, whole-model single step, single thread, each cell in an independent subprocess,
5 interleaved rounds, median of 50 steps)

| Metric | fp32 | INT4 |
|---|---|---|
| blob size | 729.1 MB | 269.0 MB |
| single-step median | 49.16 ms | 17.35 ms |

End-to-end **2.83×** (5/5 per-round wins), size compressed 2.7×.

**（4）Quality** (`experiments/_rwkv7_int4_quality.py`, greedy comparison on the same prompt)
English factual/common-sense continuation stays coherent ("The Eiffel Tower is located in the city of Paris, France…");
Chinese and code degrade (Chinese repeats, the code `fibonacci` recurrence changes). **Boundary**: RWKV-7 has no validation set in this
project, so no ppl statement; here only "runs and is fast" is published, **no claim of lossless quality** — compressing quality loss at
4-bit requires calibration or mixed precision.

**（5）Bugs localized during development**
- The exporter writes a 4-byte format tag for **every** tensor, but the C side's `take_f32` does not skip it, so the tag is treated as
  weight data, the cursor is displaced throughout, and reads go past the end of the file. The fix is for `take_f32` to skip 4 bytes first.
- INT4 requantization matrix direction: `gemv_i4` wants `(out, in)`, but the exporter writes `(in, out)`,
  so it must be transposed before quantization, otherwise reads go out of bounds and crash.

### 24.11b External comparison: llama.cpp CPU backend (same weights)

Every RWKV-7 speed number so far was "C kernel vs our own numpy" — not an external comparison. This section supplies the only
credible form of it: **take one GGUF and compare against llama.cpp's CPU backend**.

The weight source is `Mungert/rwkv7-191M-world-GGUF`'s `rwkv7-191M-world-bf16.gguf`; both sides start from it. llama.cpp reads the
gguf directly (plus F32 / Q4_0 produced by `llama-quantize`); SunnyNet goes through the blob exported by `_export_rwkv7.py`
(fp32 / INT4). llama.cpp is **b11889** (Clang 20.1.8 build, `llama-bench`, warmup built in, `-r 5` median); SunnyNet uses its own C
kernel (single-step with state already built, median of 60 steps). Each cell is an independent subprocess; rounds are interleaved.

This i7-8750H throttles under sustained load — adjacent rounds of the same config can differ by 2×. So **only the per-round paired
ratio** is reported; absolute values are for reference. Script `experiments/_llamacpp_vs_sunny_st.py`.

**（1）Single thread** (7 rounds, `THREADS=1`)

| Impl | ms/token | Weight bytes | Effective BW |
|---|---:|---:|---:|
| llama.cpp bf16 | 35.7 | 384 MB | 10.75 GB/s |
| llama.cpp F32 | 52.8 | 723 MB | 13.69 GB/s |
| SunnyNet fp32 | 61.0 | 764 MB | 12.52 GB/s |
| llama.cpp Q4_0 | 14.5 | 127 MB | 8.76 GB/s |
| SunnyNet INT4 | 16.8 | 282 MB | 16.83 GB/s |

| Per-round pair | Ratio | Rounds won |
|---|---:|---:|
| SunnyNet fp32 vs llama.cpp F32 (both fp32) | **0.90×** | 1/7 |
| SunnyNet INT4 vs llama.cpp Q4_0 (both 4-bit) | **0.89×** | 0/7 |
| SunnyNet INT4 vs llama.cpp bf16 (cross-precision) | 2.12× | 7/7 |

**（2）Six threads** (5 rounds, `THREADS=6`)

| Impl | ms/token | Effective BW |
|---|---:|---:|
| llama.cpp F32 | 37.1 | 19.49 GB/s |
| SunnyNet fp32 | 61.0 | 12.52 GB/s |
| llama.cpp Q4_0 | 9.1 | 13.93 GB/s |
| SunnyNet INT4 | 13.1 | 21.51 GB/s |

| Per-round pair | Ratio | Rounds won |
|---|---:|---:|
| SunnyNet fp32 vs llama.cpp F32 | **0.58×** | 0/5 |
| SunnyNet INT4 vs llama.cpp Q4_0 | **0.71×** | 1/5 |

**Conclusion (contrary to the existing narrative, recorded as measured)**: **on the same weights and precision, llama.cpp's CPU
backend is faster than SunnyNet's C kernel** — about 1.1× single-thread, 1.4–1.7× at six threads. The earlier "5.39× over numpy"
is an internal multiple and must not be presented as leading an external engine.

Where the gap is:
- **Single thread is already at the bandwidth wall.** SunnyNet fp32 reaches 12.5 vs llama.cpp 13.7 GB/s; both are near the sequential
  read ceiling (18.63 GB/s measured on this machine), so the single-thread gap is only ~10%. It comes from llama.cpp's leaner
  kernels: it uses SIMD vectorized loads and repack for F16/BF16 weights (`ggml-cpu-haswell` backend), while SunnyNet's GEMV uses a
  scalar `cvtph_ps` expansion.
- **The gap widens with threads.** llama.cpp reaches 19.5 GB/s at six threads, SunnyNet only 12.5 — the former partitions rows without
  cross-thread redundant reads; SunnyNet's row-major parallelism does not saturate bandwidth.
- **SunnyNet INT4's effective bandwidth reads high (16.8–21.5 GB/s)** because its 282 MB counts the fp32 embedding table, while the bulk
  is INT4 matrices; dividing effective bytes by step time exceeds the sequential ceiling, meaning not all reads hit DRAM (some smaller
  matrices stay in L2). This does not change the "~1.1–1.4× slower than llama.cpp Q4_0" conclusion.

**Boundaries**: single machine, single model (191M), single sequence, B=1; the llama.cpp side has no KV cache (RWKV has none) and no
prefill; llama.cpp uses the official prebuilt Haswell backend (F16C/AVX2 included, AVX-512 not enabled). The conclusion does not
extrapolate to other architectures (Mamba-2 untested) or to other machines.

### 24.12 External weight integration: Mamba-2 and DeltaNet

`runtime/mamba2.py`, `runtime/deltanet.py`. The same `step(token, state)` interface loads the official weights of two other
linear architectures, per-token single-step recurrence, forward not going through PyTorch. The weights are large and not committed;
place them in any directory and point to it with `--hf-dir`, or set the `SUNNY_MAMBA2` / `SUNNY_DELTANET`
environment variables for tests (unset skips the related cases).

**（1）Mamba-2** (`state-spaces/mamba2-130m`, d_model 768 / 24 layers / 24 head × 64)

This checkpoint's `config.json` has only `ssm_cfg: {"layer": "Mamba2"}` and **does not contain** the dimension hyperparameters,
which must be filled in from `mamba_ssm`'s `Mamba2` defaults (`ngroups=1`, `d_state=128`); this combination is self-consistent with
`conv_dim = 1536 + 2·1·128 = 1792`, `nheads = 1536/64 = 24`, and `A_log` length 24.
`vocab_size` is actually padded to 50288. Weights are 219 tensors.

Mathematically each head's state is a `(head_dim=64, state_size=128)` matrix, but the decay `dA = exp(dt·A)`
is identical across `(p, n)` and the outer product is also decomposable, so it degenerates to an element-wise first-order recurrence with no matrix-vector product:

```
dA  = exp(dt_h · A_h)
S_t = S_{t-1} ⊙ dA + (dt_h · B_g) ⊗ x_h
```

Compared against `transformers`' `Mamba2ForCausalLM` full-sequence forward (`experiments/_mamba2_check.py`,
24 tokens):

| Quantity | Value |
|---|---|
| max\|diff\| | 5.875e-04 |
| mean\|diff\| | 6.003e-05 |
| argmax agreement | 24/24 |
| single step vs segmented continuation | **0.0** (bitwise) |

The magnitude of 5.9e-04 requires ruling out a formula error first: the official **fp32** itself deviates **3.814e-04**
relative to **fp64** (the cumsum/segment\_sum path of `mamba2_chunk_scan`), whereas this implementation deviates only
**2.518e-04** relative to fp64, **closer to the true value than the official fp32**. ⇒ The difference is single-precision accumulation, not a math error.

Greedy continuation (GPT-NeoX tokenizer, obtained separately from `EleutherAI/gpt-neox-20b` — this checkpoint does not include a tokenizer file):

```
The capital of France is a city of the French people, and the capital of the French Republic.
def fibonacci_2(n):\n    return n %
```

**（2）DeltaNet** (`fla-org`'s DeltaNet 340M, hidden 1024 / 24 layers / 8 head × 128)

338 tensors. `use_beta=true` (headwise, `b_proj` outputs 8), `qk_norm=l2`, `qk_activation=silu`,
`use_short_conv=true` (q/k/v each have a kernel=4 depthwise convolution, no bias), `use_output_norm=true`
(non-gated RMSNorm, `o_norm` length 128), `use_gate=false`, `allow_neg_eigval=false`
⇒ **no decay gate**, it is a pure delta rule. The math basis is the `fused_recurrent` path of `fla/layers/delta_net.py`
and `fla/ops/delta_rule/fused_recurrent.py`:

```
δ_t = (v_t − S_{t-1}ᵀ k_t) · β_t
S_t = S_{t-1} + k_t ⊗ δ_t
o_t = S_tᵀ q_t
```

In this repo `S` is stored as `(k_dim, v_dim)`, the transpose of the standard notation; in code, use
`kv = einsum("hkv,hk->hv", S, k)`, `o = einsum("hkv,hk->hv", S, q)`.

Check (`experiments/_deltanet_check.py`):

| Check | Result |
|---|---|
| Rank-1 form vs explicit `(I − β k kᵀ) S + β k vᵀ` | max\|diff\| **4.578e-05** |
| full run vs segmented continuation | **0.0** (bitwise) |
| Repeated runs bit-stable | **0.0** |
| Greedy continuation | `<s>The capital of France is the city of Paris. It is the largest city in Europe and the second largest in the world.` |

**（3）Speed** (`experiments/_hf_runtime_bench.py`, single-thread numpy reference path,
each cell in an independent subprocess × 5 interleaved rounds, median of 30 steps)

| Architecture | median ms/token | min | max |
|---|---|---|---|
| mamba2 | 45.68 | 38.27 | 55.42 |
| deltanet | 91.46 | 83.93 | 99.19 |

**This is the absolute time of the reference path and is not compared as a multiple against any baseline.** The single-step complexity of
both is `O(heads × head_dim × state_size)`, independent of context length.

**（4）The matrix-valued recurrence level is already implemented in the C kernel** (`sn_mat_scan`, `runtime/recur.c`): the
state updates of both RWKV-7 and DeltaNet contain an outer-product accumulation like `S ← S + k ⊗ (·)`, which the per-channel
forms `SN_FMA` / `SN_GATED` cannot express; Mamba-2 does not need it because its decayed outer product is decomposable. The kernel's unified form is

```
p[i]   = Σ_j S[i][j] · u[j]                    read S before the update
S[i][j] = S[i][j] · w[j] + v[i]·k[j] − p[i]·b[j]
y[i]   = Σ_j S[i][j] · r[j]
```

Passing `NULL` for any parameter skips that term, covering four architectures: RWKV-7 (`w`=decay column, `u`=kk, `v`=v, `k`=k,
`b`=kk⊙a, `r`=r), DeltaNet (`w=NULL`, `u=k`, `v=βv`, `k=k`, `b=βk`, `r=q`).
Serial per head (no dependency between heads), 8-wide vectorization along the `C` dimension plus a scalar tail.

The numerical contract is in `tests/test_mat_scan.py` (11 cases: all terms / no decay / outer product only / decay only / no readout /
tail paths for `C ∈ {7,20,24,33}` / consecutive multiple steps / reject non-contiguous), relative error on the order of 1e-7 (fp32 accumulation).

**（5）Speed and share — a fast operator is not a fast end-to-end** (`experiments/_mat_scan_bench.py`, run exclusively)

| Shape (H,R,C) | Kernel ms | numpy ms | Multiple | Kernel IQR% |
|---|---|---|---|---|
| 8, 128, 128 | 0.0758 | 0.3221 | **4.25×** | 3.0 |
| 8, 64, 64 | 0.0500 | 0.1170 | 2.34× | 4.6 |
| 16, 32, 256 | 0.0988 | 0.3032 | 3.07× | 6.3 |

**This is only an operator-level multiple.** Measured, its share in a DeltaNet single step:

| Quantity | Value |
|---|---|
| DeltaNet single step (L=24, H=8, 128×128) | 75.405 ms |
| of which the 24 layers' rank-1 updates total | 1.639 ms |
| **Share** | **2.2%** |

⇒ **This kernel can compress a DeltaNet single step to ~73.8 ms (−2.2%), which does not constitute an effective speedup.** The remaining time is in
per-layer GEMV (`step()`'s own tottime is 81.5%); to really speed up, the whole-model forward must be moved into C (task #71),
not just localizing one operator into C. This conclusion is consistent with the lesson of §24.6: operator-level advantages are diluted end-to-end.

**（6）DeltaNet whole model into C** (`runtime/deltanet.c`, follow-up to task #70)

The conclusion of §24.12(5) is that operator-level C localization accounts for only 2.2% of a DeltaNet single step, so the **entire forward** is moved into C
(`sn_dn_load` / `sn_dn_step` / `sn_dn_prefill`), with weights exported by `experiments/_export_deltanet.py`
into a 1.37 GB fp32 blob (large, not committed; point to it with `--blob`, `--hf-arch deltanet`).

Compared against the numpy reference path (independent implementation, einsum path), 36 tokens step by step:

| Check | Result |
|---|---|
| logits max\|diff\| | **1.335e-05** |
| argmax agreement | **36/36** |
| Repeated runs bit-stable | **0.0** |
| rerun after reset vs first run | **0.0** |
| prefill vs step by step | **0.0** |

Speed (`experiments/_deltanet_c_bench.py`, each cell in an independent subprocess × 5 interleaved rounds):

| Round | numpy ms | C ms | C multiple |
|---|---|---|---|
| 0 | 79.42 | 71.12 | 1.12× |
| 1 | 78.69 | 72.60 | 1.08× |
| 2 | 84.69 | 70.10 | 1.21× |
| 3 | 78.61 | 71.91 | 1.09× |
| 4 | 76.71 | 71.24 | 1.08× |
| median | 78.69 | 71.24 | **1.10× (5/5 per-round wins)** |

**Why only 1.10× while RWKV-7 has 5.39×**: a DeltaNet single step must read all 1.37 GB of weights, and 71.24 ms corresponds to an
effective bandwidth of **19.18 GB/s**; numpy's 17.36 GB/s is also already close to the single-thread sequential-read ceiling
(`bench_bw.c` measures 18.63 GB/s) — **both paths hit the bandwidth wall**, so C localization only gets the residual.
RWKV-7's 5.39× comes from its numpy reference (partially materializing whole segments, 366 ms) being far from bandwidth-bound, and the two
are not comparable. ⇒ **For memory-access-bound whole-model decoding, the ceiling on the C-localization gain is set by bandwidth, not by the implementation language.**

Thread scaling (same process, median of 30 steps, weights 1.37 GB):

| Threads | ms | Effective bandwidth GB/s |
|---|---|---|
| 1 | 97.08 | 14.08 |
| 4 | 67.84 | 20.14 |
| **6 (physical cores)** | **64.33** | **21.24** |
| 8 | 74.14 | 18.43 |
| 12 | 68.55 | 19.93 |

Same as p100mw in §24.7b: **taking the physical core count** (6) is better than logical cores (12). The default thread count is taken
automatically as the physical cores by `SnDeltaNet(..., threads=None)`.

**（7）Pitfalls localized during development**
- Mamba-2's `config.json` lacks `state_size` / `n_groups`; `conv_dim` only determines their **product**
  (`(conv_dim − d_inner)/2 = 128`), which cannot be split by looking at the weights alone. The four splits `(8,16)` / `(1,128)` / `(2,64)` / `(4,32)`
  all agree with the official forward to the same order, and only going back to the `mamba_ssm` defaults settles it.
- DeltaNet's `head_k_dim == head_v_dim == 128`, so an explicit matrix form multiplying the wrong side **does not raise a dimension error**,
  it silently computes a different expression (the first check therefore got a spurious difference of 134.3).
- `state-spaces/mamba2-130m` does not include a tokenizer; feeding the model with a 50277 vocabulary using a Llama tokenizer (vocab 32000) produces
  garbage, so a GPT-NeoX tokenizer (vocab 50254) must be obtained separately.

**（8）Five real defects localized during DeltaNet C kernel development** (localized by per-layer table comparison, not static self-check)

1. **`l2norm` was written as per-8-wide block normalization**, while it should compute one norm over the entire head (128 dimensions). Symptom:
   `v` (which does not do l2norm) is bitwise identical, `q`/`k` differ by 0.09 / 0.99 — the difference appears only on the tensors that do normalization.
2. **`dconv_silu` called in place stored the post-silu value into the history**. The reference implementation stores the convolution **input**
   (`hist[-1] = x`, i.e. the projection value before conv); after an in-place writeback `x` is already the silu result and the `memcpy` stores the wrong
   thing. A `tmp` buffer must be kept to copy the original input first. **This is invisible in a single step, since with an all-zero history the two versions give the same result** —
   it only shows up from step 4 (conv=4) onward.
3. **`o_norm` must normalize per head**, not the whole 1024 dimensions at once; and `o_proj`'s input dimension is
   `NH·HV`, not `HV`.
4. **Weight transpose direction**. HF stores `(out,in)`; if the C-side `gemv` reads row-major `(out,in)` with each row contiguous,
   **no transpose is needed**. The first version wrote `gemv` to read `(in,out)`, so the exporter transposed the matrices —
   a square matrix (1024×1024) transposed raises no error and does not equal itself, it silently computes a different expression.
   Finally changed to the same `(out,in)` + parallelize along output channels scheme as `sunnynet.c`, and the exporter no longer transposes.
5. **When tied, both `head` and `embed` are `(vocab,D)`**, exactly the `(out,in)` layout that `gemv` wants;
   the first version additionally made a transpose, which instead computed wrong. The three tie scenarios (`tied_head=1`) must be
   aligned with HF's `lm_head` semantics before fixing the layout.

**（9）Mamba-2 whole model into C** (`runtime/mamba2.c`)

§24.12(5) already described that Mamba-2's decay `dA = exp(dt·A)` is identical across `(p,n)` and its outer product is decomposable, so each head's
state is an element-wise first-order recurrence, simpler than DeltaNet's rank-1 form. The whole-model forward is moved into
`runtime/mamba2.c` (`sn_m2_load` / `sn_m2_step` / `sn_m2_prefill` / `sn_m2_dbg`),
with weights exported by `experiments/_export_mamba2.py` into a 516 MB fp32 blob (point to it with `--blob`).

Per-token comparison against the numpy reference (`experiments/_mamba2_c_equiv.py`, 36 tokens):

| Check | Result |
|---|---|
| logits max\|diff\| | **1.678e-04** |
| Relative error | **6.307e-06** |
| argmax agreement | **36/36** |

The absolute error is an order larger than DeltaNet (1.335e-05), but relative it is only 6.3e-06. Per-layer table comparison
(`experiments/_mamba2_layer_probe.py`, comparing the main-stream `x` after each layer with the reference's independent implementation) confirms
**no systematic bias**: the per-layer relative error across all 24 layers falls in **4e-08 ~ 1.1e-06**, and does not grow with layer index
(for token=415 it is actually smaller further along).

| token | layer 0 rel | layer 12 rel | layer 23 rel | \|x\|max @L23 |
|---|---|---|---|---|
| 1 | 3.87e-07 | 2.15e-07 | 1.02e-06 | 730.3 |
| 415 | 2.71e-07 | 5.31e-08 | 1.11e-07 | 685.0 |
| 5565 | 2.11e-07 | 3.98e-08 | 1.26e-07 | 606.4 |

The largest absolute error at token=1 (7.477e-04) is only because that token's final-layer activation is amplified to `|x|max=730`;
relative it is still 1.0e-06. ⇒ The difference is single-precision noise from the reduction order over 3352 dimensions within a single step, not accumulation and not a formula error.

Speed (`experiments/_mamba2_c_bench.py`, each cell in an independent subprocess × 5 interleaved rounds, median of 40 steps):

| Round | numpy ms | C ms | C multiple |
|---|---|---|---|
| 0 | 63.20 | 39.09 | 1.62× |
| 1 | 57.80 | 33.96 | 1.70× |
| 2 | 47.36 | 34.59 | 1.37× |
| 3 | 46.96 | 33.21 | 1.41× |
| 4 | 44.87 | 35.38 | 1.27× |
| median | 47.36 | 34.59 | **1.37× (5/5 per-round wins)** |

Higher than DeltaNet's 1.10×, again for bandwidth reasons: Mamba-2's weights are only 0.52 GB (fp32), 38% of DeltaNet's
1.37 GB, and neither path is pressed against the 18.63 GB/s ceiling. ⇒ **Same conclusion**: the C-localization gain of whole-model decoding
is determined by "weight bytes / available bandwidth"; the smaller the weights, the further from the bandwidth wall, and the higher the fraction the language layer can get.

CLI end-to-end (`--hf-arch mamba2 --blob`, backend marked `hf/mamba2/c`):

```
The capital of France is a city of the French people, and the capital of the French Republic.
It is the capital of the Republic of France
[load 221.7 ms; prefill 5 tok 201.3 ms; decode 24 tok 37.38 ms/tok]
```

**（10）Mamba-2 C kernel self-check**: `sn_m2_dbg` advances the state, so during per-layer table comparison **`reset` must be called before
every call**, otherwise from the second call on it reads the accumulated state. The same pitfall exists on the DeltaNet side (`sn_dn_dbg`),
hit when first writing the probe.

**（11）GLA whole model into C** (`runtime/gla.c`)

GLA, like Mamba-2, is an element-wise recurrence (§24.12 already confirmed the decay acts only on the key dimension and decomposes into row-wise scaling),
but its layer structure is more complex: an extra low-rank decay projection (`gk_proj` 16 dimensions), an output gate (`g_proj` + per-head
RMSNorm), and a SwiGLU MLP. The whole-model forward is moved into `runtime/gla.c`, with weights exported by
`experiments/_export_gla.py` into a 1.37 GB fp32 blob (source is bf16).

Per-token comparison against the numpy reference (`experiments/_gla_c_equiv.py`, 36 tokens):

| Check | Result |
|---|---|
| logits max\|diff\| | **1.431e-05** |
| Relative error | **1.081e-06** |
| argmax agreement | **36/36** |
| Repeated runs / rerun after reset | bitwise **0.0** |
| prefill vs step by step | **0.0** |

Speed (`experiments/_gla_c_bench.py`, each cell in an independent subprocess × 5 interleaved rounds):

| Round | numpy ms | C ms | C multiple |
|---|---|---|---|
| 0 | 77.65 | 74.15 | 1.05× |
| 1 | 74.92 | 75.23 | 1.00× |
| 2 | 75.47 | 70.69 | 1.07× |
| 3 | 72.83 | 69.95 | 1.04× |
| 4 | 74.89 | 74.44 | 1.01× |
| median | 74.92 | 74.15 | **1.01× (4/5 per-round wins)** |

**1.01× is the extreme case of the bandwidth wall**: GLA's 1.37 GB of weights is the same order as DeltaNet, the theoretical lower bound is
`1.37 GB ÷ 18.63 GB/s ≈ 73.5 ms`, and the measured medians for numpy and C (74.92 / 74.15 ms) **are both pressed against
that line**, so C localization gets no margin at all. ⇒ Viewed together with DeltaNet's 1.10× and Mamba-2's 1.37×,
the three architectures jointly confirm the same rule: **the C-localization gain of whole-model decoding is determined by "weight bytes / available bandwidth"**.

HGRN2 (§24.12（12）, 5.46 GB) pushes this rule to the extreme end: C relative to numpy is only **1.27×**,
and single-thread 381 ms already falls under the single-thread sequential-read ceiling (18.63 GB/s), while 8 threads at 205 ms corresponds to
26.6 GB/s, also close to the multi-thread ceiling (30.45 GB/s). **At 5.46 GB of weights, any implementation can only wait for memory to finish reading.**
At 1.37 GB the gain approaches zero, and only at 0.52 GB is there 1.37×.

CLI end-to-end (`--hf-arch gla --blob`, backend marked `hf/gla/c`):

```
<s> The capital of France is the city of Paris.
The city of Paris is the capital of France.
[load 569.2 ms; prefill 6 tok 498.5 ms; decode 20 tok 94.91 ms/tok]
```

**（12）A real defect localized during GLA development**: the single-step readout differs by **96%** while all projections (`attn_norm` /
`q` / `k` / `v` / `exp(gk)` / `g_proj`) are correct to 1e-7. The root cause is that in `gla_scan` every head reads the **0-th segment** of `v`
(`v + j`) rather than that head's own value segment (`v + h·HV + j`), so all heads share the same value vector.

The localization means was the **in-layer stage probe** `sn_gla_probe` (exposing intermediates at 10 stages: `attn_norm` / `q` /
`k` / `v` / `exp(gk)` / `g_proj` / readout / after per-head RMSNorm / after gate / `o_proj`),
comparing stage by stage with the reference — **looking only at the layer output cannot localize it**, because the layer output is a sum of multiple correct terms and one wrong term.

When `head_k_dim ≠ head_v_dim` (128 vs 256), this indexing error **does not raise an out-of-bounds**: the total `v` is
`NH·HV = 1024`, and using `h·HV` or 0 are both within bounds, it only silently computes a different expression.

Localization means: adding `sn_dn_dbg` on the C side to dump per-layer intermediates and compare **term by term** with the numpy reference.
Four of the five defects are pure numerical misalignment (no crash, no error) and can only be caught by comparing intermediates.

**（12）HGRN2 whole model into C** (`runtime/hgrn2.c`, `fla-hub/hgrn2-1.3B-100B`)

dim 2048 / 24 layers / 16 head / head_f_dim = head_i_dim = 128 / vocab 32000 /
244 tensors / 1.36B parameters. This checkpoint is **not in `transformers`' native support list**
(5.20.2 has only mamba / mamba2 / rwkv / zamba / falcon_mamba), so the comparison baseline is a numpy reference written
independently in this repo (`runtime/hgrn2.py`, explicit matrix-vector loops, without einsum).

Isomorphic in recurrence to GLA (the paper calls it "identical recurrent structure", HGRN2 reuses the GLA kernel),
with four parameterization differences:

| Term | GLA | HGRN2 |
|---|---|---|
| key | `k_proj` separate projection | **no k_proj**, `k = 1 − exp(g)` derived from the decay |
| decay | `gk_proj` low-rank (16 dims) + `logsigmoid` / normalizer | `f_proj` full rank 2048 + `logaddexp(lb, log1p(−lb) + logsigmoid(f))` |
| lower bound | `clamp_min` scalar (unused by this checkpoint) | **per-layer learned `lower_bounds[24, 2048]`** |
| q / output gate | q read out directly; output gate `RMSNorm(o)⊙silu(g_proj)` per head | **q goes through silu**; **no output gate**, `o` is RMSNorm'd over the whole value dimension |

`lb` is precomputed in the layer loop: `softmax(lower_bounds, dim=0).cumsum(0) − softmax(...)[0]`,
where the softmax is taken over the **layer dimension** (24), layer 0 is always 0; within a layer only `i > 0` participates in the mixing.

Comparison results:

| Check | Result |
|---|---|
| single step vs segmented continuation | **0.000e+00** (bitwise) |
| single step vs independent numpy reference (12 tokens) | **1.192e-05** |
| C kernel vs numpy reference (per token, 36 steps) | max\|diff\| **3.552e-05**, relative **2.384e-06** |
| argmax agreement | **36/36** |
| Greedy continuation | `The capital of France is Paris.` |
| Full test suite | **183 passed / 8 skipped / exit 0** (no regression, previously 175 / 2) |

Speed (`experiments/_hgrn2_c_bench.py`, B=1 single sequence, median of 8 steps after a 2-step warmup on the numpy side):

| Threads | ms/tok | vs numpy |
|---|---|---|
| numpy reference | 262.44 | 1.00× |
| 1 | 382.00 | 0.69× |
| 4 | 221.42 | 1.19× |
| 8 | 203.33 | **1.29×** |

Weights 5.46 GB fp32. The single-thread sequential-read lower bound is `5.46 GB ÷ 18.63 GB/s ≈ 292.9 ms`; the measured single-thread
382 ms follows this line (the single-thread sequential-read ceiling for pure GEMV is 18.63 GB/s); 8 threads at 203 ms corresponds to
26.8 GB/s, already close to the 30.45 GB/s multi-thread ceiling measured at 4 threads. **This is the extreme end of "the larger the weights,
the smaller the C-localization gain"**: at 5.46 GB of weights any implementation can only wait for memory to finish reading, and 1.29× is the entire obtainable margin.

Two real defects localized during development:

- **Probing the length in blob loading with `fseek(f, 0, SEEK_END)` fails silently**. mingw's `long` is 32-bit,
  so when 5.46 GB > 2³¹, `ftell` overflows, `fseek` returns nonzero, and the load function returns NULL without reporting a reason.
  That code is already redundant after switching to file mapping and was deleted outright.
- **A 5.46 GB fp32 blob cannot be `malloc` + `fread`'d**. This machine has 15.87 GB physical memory and 5.79 GB free;
  copying into the heap takes another 5.46 GB, and stacked on the peak during export it segfaults. Switched to `MapViewOfFile` / `mmap`
  read-only mapping, shared by the page cache, so only pages actually read occupy memory — **this is a necessary condition for edge devices, not an optimization**.

The numpy reference likewise switched to `np.memmap` over the blob (reading the same layout of the same data as C, making the comparison stricter),
while the HF side converts tensor by tensor lazily; neither path needs 5.46 GB resident anymore.

**（13）The INT4 weight path for HGRN2** (`--int4` export + kernel dispatch by tag to `sn_gemv_i4`)

The 1.29× in the table above shows fp32 HGRN2 has no margin left to mine — **at 5.46 GB of weights any implementation can only wait for memory to finish reading**.
The only lever left is **reading fewer bytes**, so the grouped INT4 GEMV already validated on RWKV-7 (`runtime/quant.c`)
is connected to HGRN2: GEMV matrices are symmetrically quantized with `block=64`, each matrix prefixed with an `int32` format tag
(1=INT4, 0=fp32), and the blob magic changed to `SHGQ` (`0x51474853`). Vectors and embedding tables are not quantized
(embedding is a lookup, not a multiply-add). **The fp32 blob and its verification chain are completely untouched**, and the two blob file names and magics differ.

| Item | fp32 | INT4 | Ratio |
|---|---|---|---|
| blob size | 5457.6 MB | **993.4 MB** | **5.49×** |
| single step 1 thread | 382.00 ms | 158.82 ms | 2.41× |
| single step 4 threads | 221.42 ms | 88.42 ms | 2.50× |
| single step 8 threads | 203.33 ms | **72.19 ms** | **2.82×** |
| Theoretical lower bound (sequential read 18.63 GB/s) | 293.2 ms | 53.3 ms | — |

CLI end-to-end (`--blob .../hgrn2.q4.bin`, backend still marked `hf/hgrn2/c`):
`The capital of France is Paris.`, **79.86 ms/tok** (fp32 on the same prompt is 222.05 ms/tok, **2.78×**).

**Precision**: grouped INT4 uses `scale = amax / 7` (covering `[0,15]`, 7 positive levels, with `q=0` corresponding to −8),
theoretical relative error `1/(2·7) ≈ 0.071`, measured per-matrix **0.097–0.168** (cos 0.993) —
the same level as RWKV-7's ~0.11. The whole-model per-token relative error median is **0.242** (synthetic tokens, 36 steps),
and the per-layer probe shows **layer 0 is already 0.189 in a single step and does not grow between layers**, so the entire difference comes from quantization itself rather than state accumulation.

Quality assessed by **greedy continuation on real text** (`_hgrn2_int4_quality.py`, same metric as RWKV-7):

| prompt | fp32 | INT4 |
|---|---|---|
| The capital of France is | `Paris.` | `Paris.` |
| Water is made of hydrogen and | `oxygen.` | `oxygen.` |
| The theory of relativity was proposed by | `Albert Einstein in 1905.` | `Albert Einstein in 1915.` |
| The sun rises in the east and sets in the | `west.` | `west.` |

The first word and semantics are all preserved, with a per-token agreement rate of 4–9/20; the degradation shows as a **repetition tendency** (`Hydrogen is a very
light element.` → the second half starts looping) and **factual detail errors** (1905→1915). **No claim of lossless quality** —
compressing quality loss at 4 bit requires calibration, and an attempt to keep sensitive matrices in fp32 (`fproj+head` kept fp32) showed no measured improvement
(argmax still 20/36), indicating the error is distributed across multiple matrices rather than concentrated in one category.

Tests: `tests/test_hgrn2_c.py` adds 3 items (Q4 magic loading, finite / different / argmax over half vs fp32,
repeated runs bitwise stable), full suite **186 passed / 8 skipped** (previously 183 / 8, no regression).

**（14）Extending the same INT4 mechanism to GLA and DeltaNet**

In §24.12's table, GLA's 1.01× and DeltaNet's 1.10× are more worth rescuing than HGRN2: both have
1.37 GB fp32 weights, **their C-localization gain has been eaten by the bandwidth wall**, while INT4 pushes the theoretical lower bound from 73.5 ms down to 14.4 ms.

| Architecture | blob fp32→INT4 | 8 threads fp32→INT4 | Speedup | Relative error median | argmax |
|---|---|---|---|---|---|
| HGRN2-1.3B | 5457.6 → 993.4 MB (5.49×) | 203.33 → 72.19 ms | **2.82×** | 0.242 | 20/36 |
| GLA-340M | 1366.8 → 305.8 MB (4.47×) | 66.54 → 33.83 ms | **1.97×** | 0.257 | 19/36 |
| DeltaNet-340M | 1366.4 → 306.7 MB (4.46×) | 70.91 → 35.30 ms | **2.01×** | 0.243 | 33/36 |

Single-step (state=0) relative error: HGRN2 0.264 / GLA 0.264 / DeltaNet **0.047** — DeltaNet is clearly lower
(its `q/k/v/o` are pure linear projections, and the error is not amplified through `exp(g)`), so its argmax hit rate of 33/36 is also the highest.

CLI end-to-end (`--blob .../xxx.q4.bin`):

```
GLA      : The capital of France is the city of Paris.    [decode 12 tok 42.23 ms/tok]
DeltaNet : The capital of France is Paris. It is the largest city in the world...  [40.06 ms/tok]
HGRN2    : The capital of France is Paris.                [16 tok 79.86 ms/tok]
```

Items not quantized: GLA's `gk_proj.1.weight` (`in = GR = 16` does not satisfy block 64, the exporter falls back to
fp32 automatically); DeltaNet's short-convolution kernels `q/k/v_conv1d` (they are convolutions, not GEMV) and `b_proj` (`out = NH = 8`,
size negligible). Embedding tables are not quantized in either.

The blob protocol copies HGRN2: new magic (GLA `QLAG` 0x51414C47 / DeltaNet `QNDD` 0x51444E44),
each GEMV matrix prefixed with an `int32` tag. The fp32 blob and all existing verification chains are untouched.

Tests: `tests/test_gla_c.py` and `tests/test_deltanet_c.py` each add 2 items (Q4 loading and error upper bound,
repeated runs bitwise stable), full suite **190 passed / 8 skipped** (previously 186 / 8, no regression).

### 24.13 Prebuilt native libraries

The six shared libraries from the preceding twelve sections (`runtime` / `model` / `rwkv7` / `quant` / `abi` / `kernels`,
which become **10** after §24.12 adds the four whole-model libraries DeltaNet / Mamba-2 / GLA / HGRN2 in turn)
were all originally **compiled on the fly with the local gcc on first call**, so after `pip install` a separate C compiler is still required.
This section moves compilation to **wheel build time**: compiled on the build machine, the artifacts distributed with the package, and loaded directly after installation.

**Artifact layout**: `sunnynet/_prebuilt/<platform tag>/`, tags such as `win-x86_64`, `linux-x86_64`,
`macos-arm64` (`sunnynet/_native.py`'s `platform_tag()`). The loader looks up by the current platform tag;
if `locate()` hits it goes straight to `CDLL`, and on a miss it falls back to the original source-compilation path — **behavior is unchanged for platforms without prebuilt artifacts**.

**Instruction-set threshold**: the libraries are built with `-mavx2 -mfma -march=x86-64-v3` and only execute on x86-64
machines supporting AVX2+FMA, so the check is by **instruction set** rather than platform alone (`isa_ok()`). Machines that fall
short go to source compilation even on an artifact hit, avoiding an illegal-instruction crash.

**Loader rework for the six libraries**: each extracts a `_bind(lib)` to bind argtypes, a prebuilt short-circuit is inserted at the start of
`load_*`, and `*_available()` is correspondingly relaxed to "has a compiler **or** has a usable prebuilt library".

Measured (Windows / i7-8750H, all on real hardware):

| Verification item | Result |
|---|---|
| Prebuilt library vs source-compiled library (`recur_scan` same batch of inputs) | out / send **max\|diff\| = 0.0, bitwise identical** |
| No gcc (trimmed PATH, no `SUNNY_W64`) loading all six libraries | all succeed, the `CDLL` path points at `_prebuilt/win-x86_64/` |
| No gcc + source tree full test suite | **115 passed / 21 skipped / exit 0** |
| Same, before the rework | 89 passed / 47 skipped / exit 0 |
| Meaning of the difference 26 | cases previously **skipped** for lack of a compiler now really run |
| Fully isolated (no source tree, no gcc, `PYTHONPATH` pointing at the install directory) | **106 passed / 30 skipped / exit 0** |
| With gcc + source tree full test suite | **126 passed / 10 skipped / exit 0** (no regression) |
| Prebuilt artifacts in the wheel | 11 `_prebuilt/` entries (10 dll + `sources.sha256`), 74 files total in the wheel, tag `py3-none-win_amd64` |

`SUNNY_NO_PREBUILT=1` forces off the prebuilt short-circuit, used for the comparison verification in the first row of the table above.

When binaries are included the wheel is tagged with the platform (`py3-none-win_amd64`) rather than `py3-none-any`, to avoid shipping platform A's
artifacts to platform B; machines with a mismatched platform build from the sdist via pip, with the source path fully usable. The sdist is
produced by a single platform, and `_prebuilt/` is not included in the sdist (only the C sources are).

**Artifact–source consistency check**. In development, if C sources are changed but not recompiled, the loader silently hits the old prebuilt
library — this really happened during development: `sn_mat_scan` compiled fine, yet the Python side reported `AttributeError`,
because what was hit was the pre-change `sunnynet_runtime.dll`. So `build_prebuilt()` additionally writes
`_prebuilt/<tag>/sources.sha256` (source content hashes grouped by library), and `locate()` compares before loading.

The decision uses **content hashes, not mtime**: pip unpacking refreshes the source mtime to the unpack time, always newer than the
artifacts, so judging by time would mark every installed package as stale. After manually tampering with the hash of `runtime/recur.c` in the
manifest, `locate()` returned `None` for all six libraries (falling back to source compilation); after restoring it, they hit again.

Overhead: under Windows real-time protection, the first file open in a process is about 14–20 ms, after which about 0.4 ms; one load
compares only the 2–5 source files used by that library, so one-time about 70 ms, far less than the seconds of source compilation. When
sources are not distributed with the package it cannot be decided, and it is treated as not stale. Contract tests `tests/test_native_prebuilt.py` (5 items).

Isolation verification (no source tree, no gcc, `PYTHONPATH` pointing at the install directory):

| Check | Result |
|---|---|
| `locate()` for the six libraries | all hit `_prebuilt/win-x86_64/` |
| `load_runtime()` actual load path | points at `sunnynet_runtime.dll` under the install directory |
| `prebuilt_stale()` (with manifest) | False — content hashes consistent, not misjudged stale |
| Calling `mat_step` in the isolated environment | max\|diff\| 1.43e-06 (against the numpy reference) |

**Two real build-time defects** (both blow up on Windows, not reproducible on Linux):

One, `build_prebuilt()` originally called each module's `_compile(gcc)` directly, writing to the **shared**
`<pkg>/_build/<dll>`. Once that file has been loaded by this process (or another case run earlier in the same test session),
Windows does not allow overwriting, gcc cannot write, and it reports a compilation failure. Previously the full test suite produced 3
order-dependent `RuntimeError`s, while running that file alone passed entirely. Fix: compilation goes into `tempfile.TemporaryDirectory()`,
and the artifact is then copied to the target directory. A regression case was added — first `model._load()` puts the dll in the loaded state, then it asserts all seven libraries are written out.

Two, `subprocess.run(..., text=True)` decodes subprocess output as UTF-8, while on Windows gcc's diagnostics are output in the
**system code page** (GBK for Simplified Chinese), so when the source path contains non-ASCII, decoding immediately raises
`UnicodeDecodeError` — the real compilation error is masked by the exception itself. Of the six `_compile` sites only `abi.py`
did fault-tolerant decoding separately, the other five did not. Fix: route all of them through `_native.run_cc()`, collecting bytes then
`decode("utf-8", errors="replace")`.

There is one more instance of the same root problem: `deltanet_c._recommended_threads()` directly `CDLL`s a concatenated
`_build/sunnynet_model.dll`, which both bypasses the prebuilt library lookup and is a trigger source for the defect above; it was changed to reuse
`runtime.model.recommended_threads()`. If a dll in the artifact directory has already been loaded, copying it is likewise refused,
so `build_prebuilt()` catches `PermissionError` separately for each library, and an occupied library is listed separately rather than failing the whole run.

Three, `prebuilt_stale()`'s decision result is cached within the process, but `build_prebuilt()` did not clear it after a
rebuild — the normal development flow of "change source → rebuild → load" would keep reading the old stale conclusion, effectively falling back to source compilation every time.
Fix: `_STALE_CACHE.clear()` at the end of the rebuild.

Four, the contract test hardcoded the artifact name `"sunnynet_runtime.dll"`. The manifest keys are platform-suffixed artifact file names,
which on Linux is `sunnynet_runtime.so`, so `man["sunnynet_runtime.dll"]` directly raises `KeyError`.
This machine is Windows and happens to match, so it was always green locally; CI's `architecture-selfcheck` was red only on the
two `ubuntu-latest` matrices. Fix: the test takes `DLL.name` from the module instead of writing a literal.
CI for commit `8a7bfa4` had already failed because of this.

Five, **forgetting to register a new architecture library does not turn any test red**. `build_prebuilt()` only iterates over
`MODULES`; if a newly added `runtime/<arch>_c.py` is not added to `MODULES`, it neither enters the wheel nor is written to the manifest,
and after installation it can only fall back to source compilation (requiring the user's own C compiler), which is completely invisible
on a development machine with a compiler installed. After adding `hgrn2_c`, `sunnynet_hgrn2.dll` was found missing from
`_prebuilt/win-x86_64/`. Fix: a reverse-coverage case was added, requiring every `*_c.py` under `runtime/` to be registered in `MODULES`.

**Contract tests**: `tests/test_native_prebuilt.py` now has 8 items, covering manifest–source consistency, a tampered hash
judged stale, no manifest without misjudgment, missing source without misjudgment, rebuild in the loaded state, cache cleared after
rebuild, every module providing `_link_cmd` and `DLL`, and reverse coverage (every `*_c.py` is in `MODULES`).
Case names are all taken from module attributes, so they run cross-platform.

### 24.14 Attribution check for channel gating: the "softmax Jacobian has a $1/d$" claim does not hold

An early design attributed the depth gradient decay to "the softmax Jacobian contains a $1/d$ factor,
discounting once per layer, so deep stacking must kill the gradient", and on that basis switched channel
competition from softmax to intra-group pairwise competition. This section checks the attribution with an
operator-level and a model-level measurement.

Script `experiments/_gate_originality.py jac|grad`, results `experiments/results/_gate_originality_{jac,grad}.json`.

**(1) Operator-level Jacobian spectral norm** (fp64, `torch.autograd.functional.jacobian`, `||J||₂`):

| d | pairwise | softmax | scalar | none |
|---|---|---|---|---|
| 16 | 0.44 | 0.32 | 0.25 | 0 |
| 64 | 0.46 | 0.35 | 0.25 | 0 |
| 128 | 0.48 | 0.31 | 0.25 | 0 |
| 256 | 0.48 | 0.35 | 0.25 | 0 |
| 512 | 0.48 | 0.39 | 0.25 | 0 |

The softmax Jacobian spectral norm **rises** with d (0.32 → 0.39); there is no 1/d decay, and its median
is about 0.35, not 0. Pairwise competition is 0.44 → 0.48, also O(1). The two have no order-of-magnitude gap,
so the original argument does not hold.

**(2) Model-level embedding gradient norm** (dim=128, 2/4/8/16 layers, one forward-backward):

| gate | L2 | L4 | L8 | L16 |
|---|---|---|---|---|
| pairwise | 1.08 | 1.11 | 1.19 | 1.26 |
| softmax | 1.07 | 1.10 | 1.16 | 1.22 |
| scalar | 1.03 | 1.03 | 1.04 | 1.07 |

The gradient does not decay with depth (at 16 layers it is still slightly above 2 layers), and pairwise vs
softmax differ by ~3%, within the same magnitude. The original "0.018× decay, exactly zero at 16 layers" is
not reproduced.

**Conclusion**: the real causes of the depth gradient decay were the inverted spectral-radius sign (§4) and
the missing residual (§5), both since corrected. The pairwise competition gate is kept, but for the engineering
tradeoff (two sigmoids versus softmax's max-subtract and exp, one fewer reduction), no longer as a "fix for
dying gradients". README (both languages) and ARCHITECTURE §1(c) have been corrected accordingly.

**(3) Quality comparison of gate forms**: the same harness (dim=128, 4 layers, 600 steps, lr 3e-3, zh_ancient, 3 seeds)
with four gates, measured by val ppl:

| gate | seed 1234 | seed 7 | seed 2024 | mean |
|---|---|---|---|---|
| pairwise (current) | 332.11 | 329.67 | 318.06 | 326.61 |
| softmax (original) | 331.37 | 321.75 | 330.77 | 327.96 |
| scalar sigmoid | 326.37 | 318.43 | 331.02 | 325.27 |
| none | 362.43 | 371.98 | 373.87 | 369.42 |

**Conclusion**: the pairwise differences among the three gates (mean spread 2.7) are far smaller than the
seed-to-seed spread (11–14 for a single gate) and the gap to none (~43). That is: **having a gate is what matters
(no gate is ~43 ppl worse), while the quality differences among the three gate forms are within noise and do not
constitute evidence that pairwise beats softmax/scalar**. The pairwise competition gate is kept for the engineering
tradeoff (two sigmoids versus softmax's max-subtract and exp), not a quality gain. With 3 seeds this is a weak
directional read, not a significance claim.

---

## 25. 100M-scale training, weight release, and Agent ecosystem integration

§15 only achieved the **single-step feasibility** of 0.5B (SGD runs, AdamW OOMs) and had no multi-step convergence data.
This section adds the real multi-step training of `p100mw` (stack 93.83M / total 105.36M) and packages the weights into releasable assets.

### 25.1 Training

Protocol: GPU (GTX 1060 6GB, sm_61), bs=2 × seq=256 × 2000 steps = 1,024,000 tokens,
lr 1e-3, warmup 100, cosine decay, AdamW, grad_clip 1.0, seed 1234,
corpus `zh_ancient` 0.4 / `en_literature` 0.35 / `code_python` 0.25.

| Step | train loss | val loss | ppl | ms/tok |
|---|---|---|---|---|
| 200 | 5.8046 | 5.7407 | 311.29 | 0.501 |
| 600 | 5.1452 | 5.3599 | 212.71 | 0.502 |
| 1000 | 4.8508 | 5.1040 | 164.68 | 0.502 |
| 1400 | 5.9248 | 4.8997 | 134.25 | 0.503 |
| 1800 | 6.2203 | 4.8115 | 122.91 | 0.503 |
| 2000 | 2.9600 | 4.7882 | **120.09** | 0.503 |

val decreases monotonically, no divergence; wall 583.5s; peak_rss 1301.6MB; final train loss 2.9600.
train loss fluctuates a lot between cells (val is stable), a normal manifestation of a small batch (×2) combined with cross-domain corpus variation.

**Metric**: 1.024M tokens for 105M parameters is extremely undertrained (about 0.01 token/parameter);
this section **does not claim convergence and makes no cross-architecture quality comparison**, it only provides evidence of "decreases stably + can load and generate".

### 25.2 Configuration boundaries of a 6GB card

| Configuration | Result |
|---|---|
| bs=1 × 256 | Passed, 1.220 ms/tok |
| **bs=2 × 256** | **Passed, 0.744 ms/tok (chosen as the training configuration)** |
| bs=4 × 128 | Passed, 0.818 ms/tok |
| bs=4 × 256 | OOM late in training |
| bs=8 × 128 | OOM late in training |

Three measurement traps (all reproduced by measurement):

1. **`expandable_segments` is unavailable on sm_61**. Adding
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` makes the OOM appear earlier,
   not later. This is consistent with the record in §15.
2. **When VRAM is occupied by other processes, the free figure is not trustworthy**. With a resident Ollama holding 383 MiB,
   the nominal free was 5649 MiB, yet an attempt to allocate 12 MiB failed; unloading first with `keep_alive: 0` brought
   free back to 6031 MiB and it returned to normal. This is WDDM allocator fragmentation, not an arithmetic problem.
3. **Validation batches also consume VRAM**. `val-batches 8` OOMs during the validation phase at bs=2;
   lowering it to 2 is stable throughout. The training configuration and the validation configuration must be bounded separately.

### 25.3 Weight packaging and load verification

`sunnynet-release` (`sunnynet/release.py`) exports three kinds of assets + a model card + a sha256 manifest in one pass:

| Artifact | Size | First 12 digits of sha256 |
|---|---|---|
| `sunnynet-p100mw.pt` | 467.7 MB | `399e604e635b` |
| `sunnynet-p100mw.safetensors` | 421.5 MB | `c7287c76ac8b` |
| `sunnynet-p100mw.runtime.bin` | 467.6 MB | `95c195e47637` |

The safetensors file contains 139 tensors. **When the embedding is tied (`tie_embeddings=True`), `head.weight` and
`embed.weight` share storage, and safetensors refuses to save and reports "Some tensors share memory"**;
the exporter removes `head.weight` and writes `sunnynet.tied_head=1` in metadata, and the loader just reuses embed.

A header normalization is done once more after export: safetensors 0.8.0 has unstable ordering when serializing
multi-key metadata, and consecutive exports of the same checkpoint yield different bytes (the data section is exactly consistent), making the
released sha256 non-reproducible. The exporter rewrites the header JSON with `sort_keys` and updates the length field, leaving the data section untouched. Measured: without normalization six exports gave six hashes; with normalization all six are identical.

Three load paths measured (same weights):

| Path | Output | Speed |
|---|---|---|
| torch backend | `"“H不得你，也不可不得…"` | 143.6 ms/tok |
| C Runtime (`.runtime.bin`) | `"「尚與子之子，雖以一名，於我也。"` | **80.2 ms/tok** |
| OpenAI service (official SDK) | all four call styles pass | — |

The C backend's reading against the torch backend is a single measurement (20 tokens), **not stated as a speedup ratio**; it only proves the exported
blob layout matches the kernel's expectations and can be loaded independently of PyTorch.

### 25.4 Agent ecosystem integration: three pitfalls of Ollama

The RWKV-7 191M GGUF was really imported into Ollama and generated successfully (`family=rwkv7`,
`parameter_size=191.13M`, `runner=llamacpp`, `tokenizer.ggml.model=rwkv`).

1. **TEMPLATE must be written explicitly**. Ollama's built-in `rwkv-world` template wraps another
   `User: / Assistant:` layer around the prompt, and on the same prompt the output degenerates into repetition
   (`"-of-the-world-of-the-world-of-..."`). After writing
   `TEMPLATE """User: {{ .Prompt }}\n\nAssistant:"""` the output returns to
   `" The capital of France is Paris."`, and is **character-for-character identical** to the raw output of the project's own C runtime
   — proving the problem is in the template, not the tokenization.
2. **The real meaning of `Error: could not locate ollama app` is that the server is not running**.
   When the server is offline, `ollama create` silently tries to start the local app and fails. Precisely reproduced:
   `OLLAMA_HOST` pointing at an empty port → this error; server online → normal. A diagnostic script must first probe
   `GET /api/version`, otherwise it misjudges an environment problem as an installation corruption.
3. **`OLLAMA_MODELS` must match the server side**. When only `OLLAMA_HOST` is set, `create`
   writes the blob into the default directory, misaligned with the running server's directory.

There is one more environment-related pitfall: local probing must bypass the system HTTP proxy. An enterprise proxy will also intercept
`http://127.0.0.1:...` as a **502**; on the Python side an opener built with `urllib.request.ProxyHandler({})` is needed to get the real status.

LM Studio is not installed on this machine; only the GGUF directory layout (`<models>/<publisher>/<repo>/<file>.gguf`,
hard links preferred) and integration configuration generation were done, **no real load verification**.

### 25.5 Verifiable boundaries of the SIMD backends

Neither the AVX-512 nor the NEON backend is "done once written" — two hard constraints, the local CPU and the toolchain, determine how far verification can go.

**AVX-512**: this machine's i7-8750H (Coffee Lake) measures `AVX512F=0 / BW=0 / VL=0` in CPUID,
so execution is impossible. Verified: `gcc -mavx512f -mavx512bw -mavx512vl` compiles,
objdump confirms **198 `zmm` registers + 16 `vfmadd132ps`**, and
`load_avx512()` raises on unsupported machines (avoiding an illegal instruction). **Execution correctness not verified.**

**NEON**: no ARM toolchain, no QEMU (`wsl.exe` blocked by security policy). Cross-compiled a real AArch64 ELF64 (`e_machine=183`) with
`zig cc -target aarch64-linux-gnu`, and capstone 5.0.7 confirms 4-lane `fmla v3.4s, v2.4s, v4.4s` and `fmul`/`fsub`/`fadd` are generated.
**Execution correctness not verified**; execution verification is placed on the CI `ubuntu-24.04-arm` runner.

The conclusion of this section is that "the artifacts exist and the instructions are correct", **not a performance conclusion**.

---
## 26. P0 Associative Memory and Inter-Layer Input Dilution (Root Cause of CPU-Native Quality)

`CPU_WAY_OUT.md` concludes: "incoherence" has two independent root causes — insufficient training
budget, and the recall capacity ceiling of the element-level state. This section puts both on
measurement: add a matrix-valued associative memory branch to the main state (recall capacity),
and measure whether inter-layer input specificity is diluted with depth.

### 26.1 Per-Layer Cost of Associative Memory

The cost of `AssocMemory` ∝ the number of enabled layers. Sweep `assoc_every ∈ {off, 4, 3, 2, 1}`
(small preset 1.42M, $L$=4, `experiments/_assoc_layers.py`, 3 independent subprocess rounds):

| Configuration | Enabled layers | Parameters | Train ms | Ratio | µs/token | Decode ms | Ratio |
|---|---|---|---|---|---|---|---|
| off | 0 | 1,414,420 | 225.4 | 1.00× | 110.1 | 3.99 | 1.00× |
| every 4 layers | 1 | 1,445,396 | 284.0 | 1.26× | 138.7 | 3.68 | 0.92× |
| every 2 layers | 2 | 1,476,372 | 329.9 | 1.46× | 161.1 | 4.66 | 1.17× |
| every layer | 4 | 1,538,324 | 434.9 | 1.93× | 212.4 | 6.16 | 1.54× |

Training overhead is **about +22% per layer** (linear); `every4` at 1 layer is only 1.26×, and
decoding does not degrade.

### 26.2 Recall Capacity Is Non-Monotonic in the Number of Enabled Layers

`experiments/_assoc_recall_layers.py` (dim=64, $L$=4, 1500 steps, MQAR):

| Key pairs | off | every 4 layers | every 2 layers | every layer |
|---|---|---|---|---|
| 2 | 0.984 | 0.977 | **0.988** | 0.977 |
| 4 | 0.121 | 0.441 | **0.656** | 0.164 |
| 8 | 0.047 | 0.070 | 0.027 | 0.074 |

On 4 key pairs, **every 2 layers (0.656) is far better than every layer (0.164)**, i.e. enabling
all of them is both slower and worse.
⇒ The current best form is **half enabled**. (1500 steps has not converged; read as sample
efficiency.)

### 26.3 Inter-Layer Input Dilution: Absent in Shallow Models, Present in Deep Models

**First correct an over-strong conclusion.** The original criterion "pairwise cosine 0.50 of
different inputs at the last-layer state = collapse" is a **measurement artifact**: it does not
exclude the common component shared across inputs. After adding the mean-removed cosine
(`experiments/_state_collapse.py`, small, 1200 steps), the per-layer **state** pairwise cosine
`layer_pair_cos_centered` = **[−0.053, −0.047, −0.060, −0.061]** (≈0, the same order as random
initialization); the relative difference between the states of two inputs is 0.96–1.07.

`experiments/_branch_breakdown.py` decomposes the block output `out = W_c(y) + b_gain·sn + sfb + x`
branch by branch: each branch has a **mean-removed cosine of −0.03~−0.01** and a raw cosine of
0.54–0.90 ⇒ the similarity in the raw cosine **comes entirely from the common component**,
not from loss of input specificity. On the 4-layer model the share of input-specific energy
actually **rises** with depth (0.285→0.295→0.430→0.597) — **the shallow model does not compress**.

**The deep model does dilute.** Using the converged state of the 100M preset
(`experiments/ckpts_100m/sunnynet_p100mw.pt`, $L$=8, $d$=1408), measured layer by layer:

| Layer | Common component RMS | Input-specific RMS | Specific energy share |
|---|---|---|---|
| 0 | 2.169 | 1.000 | 0.226 |
| 4 | 2.674 | 0.840 | 0.110 |
| 7 | **3.073** | **0.532** | **0.035** |

The common component **grows monotonically by +42%**, the specific component **drops 47%**, and
the energy share goes 0.226→0.035 (ratio degrades 6.5×). Same ckpt on the output side: 16 inputs
give only **6 distinct argmax**, pairwise JS divergence 0.237 (the 4-layer model under the same
protocol gives 0.40+). **This is the mechanism by which "deep models are less coherent than
shallow models".**

Causal control on a deep-thin model (`deepl8`: $L$=8, $d$=80, 0.53M stack parameters):

| 2000 steps | val ppl | specific share layer0→layer7 | dilution drop | 24-input argmax |
|---|---|---|---|---|
| default | 263.47 | 0.662→0.263 | 0.399 | 19 |
| **residual-stream norm `h_norm`** | **625.16** | 0.005→**0.000** | 0.005 | **1** |
| state norm `state_norm` | **259.09** | 0.602→0.302 | 0.299 | 19 |

| 6000 steps | val ppl | specific share layer0→layer7 | dilution drop | 24-input argmax |
|---|---|---|---|---|
| default | 200.51 | 0.460→0.225 | 0.235 | **12** |
| state norm `state_norm` | **198.33** | 0.503→0.267 | 0.236 | **16** |

- **The residual stream cannot be normalized away**: `h_norm` normalizes the residual stream
  per token, the specific component is pushed to 0, 24 inputs are left with
  1 prediction, and ppl doubles ⇒ falsified. Transformer normalization acts on the sublayer input
  rather than the residual stream itself, and SunnyNet's `norm_in` already corresponds to that;
  changing the residual stream would break the identity passthrough of §5.
- **State norm is the right direction but only partially mitigates**: `state_norm` normalizes the
  state and the feedback source, blocking the route by which DC is injected into the residual
  stream from `sfb`/`sn`. At 2000 steps it moves the dilution drop from 0.399 to 0.299, and at
  6000 steps the common-component growth rate from +72% to +54%; output diversity **16 vs 12**,
  ppl **198.33 vs 200.51** (−1.1%). It is a safe positive change.
- **Dilution hardens with training**: for the default model from 2000→6000 steps, ppl improves
  (263→201) but argmax diversity instead falls from 19 to **12**, common-component RMS goes
  0.198→0.341 (+72%) while the specific component is almost unchanged (0.183→0.184)
  — this is **pure dilution** (common-component inflation), not erasure. This explains "the
  longer a deep model trains, the less coherent it becomes".
- The last layer is still unresolved (specific share 0.267 vs 0.225) ⇒ dilution has sources
  beyond the state, and a complete fix is still not achieved.

### 26.4 Output Side: Common Component Dominates, Cannot Be Read Directly as "Output Collapse"

`experiments/_output_collapse.py` (from the converged checkpoint, 32 mutually distinct inputs):

| Metric | base | +assoc (every layer) | random init |
|---|---|---|---|
| Last-position logit pairwise cosine | 0.98 | 0.96 | 0.006 |
| **Cosine after mean removal** | **−0.03** | **−0.02** | — |
| Common-component energy share | **0.975** | 0.961 | — |
| Distinct argmax count over 32 inputs | 27 | 26 | — |

The raw cosine 0.98 is **about 97% from a common component unrelated to the input** (the unigram
bias common to language models); after mean removal the cosine is ≈0, and 32 inputs give 27
distinct argmax. **This term cannot be read on its own as "output collapse"**; it must be
compared against a control model of the same scale.

### 26.5 Same-Budget Comparison at Convergence: SunnyNet vs Associative Memory vs Transformer

All four models are trained to 16000 steps × B8 × T256 = **32.8M token** (≈19.5 token/parameter,
Chinchilla order; zh_ancient single domain), with stack parameters aligned (≤±1%), the same seed,
and the same lr schedule. Evaluation uses `train/evaluate.py`.

| Model | Stack params | zh_ancient ppl | en ppl | near recall (1-gram) | Generation |
|---|---|---|---|---|---|
| SunnyNet baseline | 365,844 | 143.24 | 8798.69 | 0.023 | recognizable syntax, semantic drift |
| +assoc (every layer) | 489,748 | 133.36 | 10922.02 | 0.019 | incoherent |
| +assoc (every 2 layers) | 427,796 | 138.82 | 8008.08 | 0.019 | **coherent classical-Chinese syntax** |
| Transformer | 369,200 | 140.40 | 71778.23 | 0.023 | — |

**Readings (protocol discipline)**:

- **SunnyNet baseline vs Transformer at the same budget and the same stack parameters: ppl
  143.24 vs 140.40** (a 2.0% gap), with near recall 0.023 for both. The two are **statistically
  close**; neither dominates.
- **Transformer also fails at long-range recall under the same budget** (near 0.023 = SunnyNet
  baseline). ⇒ "Incoherence" is **mainly a scale/budget problem at 32.8M token, not unique to
  SunnyNet**.
- **Associative memory (every 2 layers) lowers zh ppl from 143.24 to 138.82** (−3.1%) with only
  +17% stack parameters, and is the only configuration that improves generation coherence.
- **Output-side diagnostic (`_output_collapse.py`)**:

  | Metric | SunnyNet baseline | +assoc (every layer) | +assoc (every 2 layers) | Transformer |
  |---|---|---|---|---|
  | Last-position logit cosine | 0.98 | 0.96 | 0.94 | 0.973 |
  | **Cosine after mean removal** | **−0.03** | **−0.02** | **−0.03** | **−0.031** |
  | Common-component share | 0.975 | 0.961 | 0.954 | 0.967 |

  The Transformer's mean-removed cosine (−0.031) and common-component share (0.967) are **of
  exactly the same order** as SunnyNet's (−0.03 / 0.975) ⇒ the phenomenon is **common to all
  architectures at low budget** (a large common logit component dominated by unigram bias),
  **not a SunnyNet architectural defect**. Reading the raw cosine 0.98 earlier as "output
  collapse" was an over-strong interpretation, now corrected.

### 26.6 Device Boundary of CPU Training and Inference Speed

The same SunnyNet code run on CPU (i7-8750H / AVX2 / torch 2.14.1+cpu) and GPU (GTX 1060 6GB /
torch 2.7.1+cu118) (`experiments/_gpu_compare.py`, an independent process per cell, synchronized
timing on the GPU side):

| Preset | T | CPU forward ms | GPU forward ms | Forward ratio | CPU train step ms | GPU train step ms | Train ratio |
|---|---|---|---|---|---|---|---|
| small 1.4M | 512 | 138.8 | 17.4 | 7.97× | 569.9 | 57.5 | 9.92× |
| small 1.4M | 1024 | 439.3 | 28.9 | 15.20× | 1186.3 | 99.0 | 11.99× |
| proto20mw 16.8M | 512 | 1697.7 | 121.5 | 13.98× | 4620.8 | 384.6 | 12.02× |
| proto20mw 16.8M | 1024 | 3308.5 | 244.4 | 13.54× | 9408.9 | 752.0 | 12.51× |

**Conclusion (protocol discipline)**:

- **CPU training does not reach GPU speed**: with the same architecture and the same code,
  train steps lag by **9.9–12.5×** (forward 8.0–15.2×). This is a structural hardware gap
  (dense GEMM efficiency CPU/GPU ≈ 8.6×, see `CPU_WAY_OUT.md` §1.2), not an implementation
  problem. **"CPU reaches GPU speed" does not hold for dense training**, recorded as is.
- **The only proven regime where CPU beats GPU**: B=1 long-context single-sequence decoding
  (`CPU_WAY_OUT.md` §4), crossover at T≈2048, and at T=8192 CPU is 2.97× faster. This is where
  the CPU-native model lands.

### 26.7 Associative-memory write rules: soft vs. delta, and write-key alignment

`AssocMemory`'s write has two switchable knobs: **what to write** (soft convex combination / delta
overwrite) and **which key to write with** (current position / previous position). This section
separates the two with the same harness.

**(1) Relative error in the overwrite scenario** (`experiments/_assoc_overwrite.py`, 8 keys written,
then 4 of them overwritten):

| Rule | Overwritten keys (relative error) | Keys left intact |
|---|---|---|
| soft (`p`=0.1 / 0.5 / 0.9) | 0.747 / 0.765 / **0.916** | 0.942–1.004 |
| delta ($β$=1.0) | **0.063** | 0.200 |

The soft write multiplies the injected term by $(1-p)$, so writing the same key repeatedly only
accumulates and reads back a mix of old and new; the delta write first subtracts the key's old
association, then writes the new value, so it overwrites exactly. **This is the one mechanism-based
argument for the delta write that holds up.**

**(2) The chunked form of delta** (`experiments/_chunk_delta.py`): delta was believed to require
per-token serial execution and to destroy the $O(N)$ parallel scan. That was wrong — delta has a WY
chunked form (in-block $S_t = S_{\text{in}} + K_{0:t}^{\top}W_{0:t}$,
$W=(I+L)^{-1}\operatorname{diag}(β)(V-K S_{\text{in}})$, $L_{ij}=β_i(k_i\cdot k_j)$ lower triangular),
implemented as `ops.chunk_delta_scan`. It matches the sequential delta to **1e-16 in fp64** (any
chunk size 4/8/16, with or without decay, all $\le 8.9\times10^{-16}$). The cost is one extra
$C\times C$ triangular solve per block. The delta rule and the WY representation belong to
DeltaNet / FLA, not this project.

**(3) Write-key alignment is a necessary condition** (`experiments/_assoc_ab.py overwrite`, $d$=64,
2 layers, 800 steps, B32/T64, slots 32×m 16, 3 seeds):

| Write key | Rule | seed 1234 | seed 7 | seed 2024 |
|---|---|---|---|---|
| current position | soft | 0.036 | 0.078 | 0.034 |
| **previous position** | soft | **0.808** | **0.895** | **0.902** |
| current position | delta | 0.036 | 0.041 | 0.056 |
| **previous position** | delta | **0.919** | **0.933** | **0.920** |
| — | off (memory disabled) | 0.036 | 0.033 | 0.034 |

Chance is $1/32=0.031$. **Consistent across three seeds**: with the write key taken from the current
position both rules fall back to chance (0.034–0.078); only with the previous position do they rise
(soft 0.81–0.90, delta 0.92–0.93). The mechanism is structural — a write is meant to express a
(key → value) association, but at the moment the key is recognized the value has not been seen; if
k/v/q all come from the current position the memory can only express token → token associations, so
the association cannot be written at all. The `off` row shows the main path cannot solve this task
($\approx$ chance), so the task is hard enough. **Write-key alignment is not an optional
optimization — it is the precondition for either overwrite write to work.**

**(4) MQAR comparison** (same harness, n\_pair=4, 3 seeds):

| Rule | seed 1234 | seed 7 | seed 2024 |
|---|---|---|---|
| off | 0.033 | 0.023 | 0.034 |
| soft | 0.936 | 0.950 | 0.959 |
| delta | 0.961 | **0.014** | 0.955 |

On MQAR, soft and delta are close in two of three seeds (within $<1.2\%$) — **weak discriminative
power**; but delta collapses to **0.014 (below chance)** in seed 7, the same source as the divergence
in §26.7(5), the stability cost of delta.

**(5) Real-corpus ppl and the stability cost** (`experiments/_assoc_ab.py ppl`, $d$=128, 4 layers,
600 steps, zh\_ancient, 3 seeds):

| Rule | seed 1234 | seed 7 | seed 2024 | Mean |
|---|---|---|---|---|
| soft | 332.05 | 324.03 | 354.97 | 337.02 |
| delta | 342.35 | 331.25 | 347.85 | 340.48 |

delta is higher than soft by **3.47 (+1.0%)** on average, but the per-seed direction is not
consistent (+10.3 / +7.2 / −7.1) and is far smaller than the within-group spread (soft 30.94 /
delta 16.60) ⇒ **read as slightly worse, within seed noise**, not a quality advantage.

The stability cost of delta is localized (`ops._safe_chunk_delta`): in fp32, chunk=64 produces
**non-finite decay gradients** in the backward pass (`torch.autograd` anomaly detector reports
`AminBackward0` — `amin`'s backward multiplies inf into non-argmin positions, giving nan), diverging
in 2/3 seeds (steps 237 / 249); lowering the block length to $\le 32$ gives 3/3 pass; fp64 is stable
throughout. The chunked form itself matches the sequential to 1e-16, and the cap does not change the
forward semantics.
**Reading**: the delta write has a real gain on overwrite recall (+12–16% vs. soft, reproduced 3/3),
at the cost of being numerically more fragile in fp32 (needs the block cap) and more seed-sensitive
(MQAR seed 7 collapse). On general language modeling the two do not differ significantly.

**(6) Training-step cost and decoder cost of the delta write** (`experiments/_assoc_ab_speed.py`,
$d$=128, $L$=4, one independent subprocess per cell, interleaved rounds):

| slots×m | B×T | soft ms/step | delta ms/step | delta/soft |
|---|---|---|---|---|
| 32×16 | 8×512 | 1671.8 / 1625.9 / 1624.4 | 1488.3 / 1446.6 / 1372.5 | **0.89 / 0.89 / 0.84** |
| 64×32 | 8×256 | 1546.7 / 1610.0 | 1003.0 / 938.2 | **0.65 / 0.58** |
| 128×64 | 4×256 | 2347.5 / 2304.0 / 2385.7 | 1198.1 / 1287.1 / 1489.3 | **0.51 / 0.56 / 0.62** |

**The chunked delta write is faster than the soft write, and the gap widens with slots×m** (3/3,
2/2, 3/3 rounds). Attribution at operator level (B8/T512, slots 32×m 16, single thread, timing
`AssocMemory.forward`): soft materializes the outer product into a $(B,T,\text{slots}\cdot m)$ tensor
and hands it to `logspace_scan`, whose `log`/`cumsum`/`exp` all run at width 512 — `cumsum` alone
costs 13.05 ms, 38% of soft's 34.27 ms; delta keeps the state at $(\text{slots},m)$, dropping the
elementwise work to width 32 and 16, at the price of one extra $C\times C$ triangular solve per block
($C\le32$). The same operator-level comparison reads soft 34.27 ms vs. delta 21.32 ms (0.62×).
Parameter cost is marginal: delta adds only `W_b` (dim+1) per associative layer — **+516 parameters
(+0.034%)** on the small-like stack (1,538,840 vs. 1,538,324).

**Decoder cost** (B=1, single step, slots 32×m 16, 4 interleaved rounds):
soft 9.006 / 9.717 / 11.559 / 9.104 ms, delta 8.172 / 8.335 / 7.503 / 9.912 ms (3/4 favour delta).
The step path evaluates both rules token-by-token; the difference here is inside the run-to-run
spread, so the decoder reads as **comparable** rather than faster.

**Boundary observation (pre-existing, not from this change)**: `assoc_slots=128, assoc_m=64` with
B=8/T=512 training **segfaults the interpreter**; the same crash reproduces on a clean `git archive`
of the HEAD tree, so it is not introduced here. The 32×16 and 64×32 configurations used above are
unaffected, and B=4/T=512 at 128×64 also passes.

### 26.8 Reproduction

```sh
python -X utf8 experiments/_assoc_layers.py                      # §26.1 per-layer cost
python -X utf8 experiments/_assoc_recall_layers.py               # §26.2 capacity vs layer count
python -X utf8 experiments/_state_collapse.py --preset small --steps 1200   # §26.3
python -X utf8 experiments/_state_collapse.py --preset small --steps 1200 \
    --ablation "assoc=1,assoc_slots=32,assoc_m=16"               # §26.3 control
python -X utf8 experiments/_branch_breakdown.py --ckpt runs/base_small_zh.pt          # §26.3 branch decomposition
python -X utf8 experiments/_branch_breakdown.py \
    --ckpt experiments/ckpts_100m/sunnynet_p100mw.pt --n-in 16 --T 128                # §26.3 deep-model dilution
python -X utf8 experiments/_depth_dilution.py --tag dl8base6k --preset deepl8 --steps 6000   # §26.3 depth ladder
python -X utf8 experiments/_depth_dilution.py --tag dl8norm6k --preset deepl8 \
    --ablation "state_norm=1" --steps 6000                                            # §26.3 state-norm control
python -X utf8 experiments/_output_collapse.py --ckpt runs/base_small_zh.pt # §26.4
# §26.5 same-budget convergence comparison (each training run ~2-4 h, all CPU)
python -X utf8 experiments/_converge_compare.py --tag base_small_zh
python -X utf8 experiments/_converge_compare.py --tag assoc_small_zh  --assoc --every 1
python -X utf8 experiments/_converge_compare.py --tag assoc2_small_zh --assoc --every 2
python -X utf8 experiments/_converge_compare.py --tag tf_small_zh --arch transformer
python -X utf8 train/evaluate.py --ckpt runs/<tag>.pt --gen --out runs/<tag>_eval.json
python -X utf8 experiments/_gpu_compare.py                       # §26.6 (needs GPU env vars)
# §26.7 write rules and write-key alignment
python -X utf8 experiments/_chunk_delta.py                       # chunked delta vs sequential (fp64)
python -X utf8 experiments/_assoc_overwrite.py                   # soft vs delta write overwrite
python -X utf8 experiments/_assoc_ab.py overwrite 800 1234       # overwrite recall A/B (3 seeds)
python -X utf8 experiments/_assoc_ab.py mqar      800 1234       # MQAR A/B
python -X utf8 experiments/_assoc_ab.py ppl       600 1234       # real-corpus ppl A/B
python -X utf8 experiments/_assoc_ab_speed.py delta_scalar 512 32 16 12  # §26.9 write-rule cost
python -X utf8 experiments/_decay_reduce.py                      # §26.9 reduction-direction conditioning
python -X utf8 experiments/_chunk_bounds.py                      # §26.9(7)(8) block bounds and guard
```

### 26.9 Decay-reduction numerical defect: the `amin` scalar collapse and a scalar decay head

The delta write of §26.7 left a pit in one implementation detail: the chunked form only accepts a
**scalar** decay, while the per-channel decay head `W_g` produces $(B,T,m)$, so every step runs
`amin(dim=-1)` to collapse it to $(B,T)$. That single reduction causes three consequences.

**(1) The numerical cost of the reduction direction** (`experiments/_decay_reduce.py`, $d$=128,
$B$=8, $T$=256, slots 32×m 16):

| Reduction | Peak $1/\prod d$ at chunk 32 | Peak at chunk 64 |
|---|---|---|
| `amin` (fastest-forgetting channel) | 3.15e8 | **9.92e16** |
| `amax` (slowest-forgetting channel) | 1.34 | **1.78** |

The block computes $1/\prod_{i\le t} d_i$, of order $d_{\min}^{-L}$. `amin` takes the **smallest**
keep-rate in the group (the fastest-forgetting channel), driving that denominator down and the
condition number up: at chunk 64 it reaches 9.92e16, only 21 dB below the fp32 ceiling of 3.4e38.
With `amax` the same block length gives **1.78**.

**(2) Wasted parameters**: `W_g` is `dim→m` ($m$=16, 2048 parameters per layer) producing 16 channels
of decay, of which the reduction keeps exactly one (the `amin` result) — **15/16 of the output is
discarded**.

**(3) Seed-dependent collapse**: which channel `amin` picks depends on the data and switches during
training; gradients are discontinuous around the switch, showing up as individual seeds (MQAR seed 7)
dropping to 0.014, below chance.

**Fix: `assoc_decay="scalar"`.** The chunked form requires a scalar decay by construction, so output a
scalar directly: a new `W_gd(dim→2)` yields (keep-rate $p$, write strength $\beta$) in one linear, with
no reduction and no per-channel head. Critically, $p$ must use the **same H parameterization** as the
per-channel head:

$$\delta = \frac{1}{h_{\max}} + \left(\frac{1}{h_{\min}} - \frac{1}{h_{\max}}\right)\sigma(g),
\quad p = 1 - \delta \;\ge\; 1 - \frac{1}{h_{\min}}$$

($p\ge0.5$ with $h_{\min}$=2). With a bare $\sigma$ the model can drive $p$ to 0.01 and $1/\prod p$
explodes — exactly why the first implementation moved the collapse from seed 7 to seed 2024. Once the
parameterization is aligned, $p$ has a hard lower bound and the block length needs no conservative cap.

**(4) Three measured comparisons** ($d$=128, 4 layers, 3 seeds, same harness):

Overwrite recall (800 steps, chance 1/32=0.0312):

| Rule | seed 1234 | seed 7 | seed 2024 | Mean |
|---|---|---|---|---|
| soft | 0.8078 | 0.8953 | 0.9016 | 0.8682 |
| delta+`amin` | 0.9187 | 0.9328 | 0.9203 | 0.9239 |
| **delta+scalar** | **0.9203** | **0.9328** | **0.9234** | **0.9255** |

MQAR (800 steps):

| Rule | seed 1234 | seed 7 | seed 2024 | Mean |
|---|---|---|---|---|
| soft | 0.9359 | 0.9500 | 0.9594 | 0.9484 |
| delta+`amin` | 0.9609 | **0.0141** | 0.9547 | 0.6432 |
| **delta+scalar** | **0.9469** | **0.9484** | **0.9406** | **0.9453** |

The `amin` seed collapse (seed 7 dropping to 0.0141) **disappears** under scalar (0.9484); scalar's
three-seed spread is 0.0083 against `amin`'s 0.9468. Its mean is slightly below `amin`'s non-collapsing
seeds (0.9453 vs 0.9578) because scalar constrains $p$ into $[0.5, 1)$ — **trading a little peak for
never collapsing**.

Real-corpus ppl (600 steps, zh_ancient):

| Rule | seed 1234 | seed 7 | seed 2024 | Mean |
|---|---|---|---|---|
| soft | 332.05 | 324.03 | 354.97 | 337.02 |
| delta+`amin` | 342.35 | 331.25 | 347.85 | 340.48 |
| delta+scalar | **328.69** | 343.24 | **329.65** | **333.86** |

scalar's mean is 3.16 below soft (−0.94%), but seed 7 goes the other way (+19.2) and the within-group
spread (soft 30.94 / scalar 14.55) dwarfs the mean gap ⇒ **same reading as §26.7(5): within seed noise,
not a quality advantage**.

**(5) Parameters and consistency**:

| Config | Total params | assoc layers | Δvs off |
|---|---|---|---|
| assoc off | 1,414,420 | 0 | 0 |
| soft (original) | 1,538,324 | 58,368 | +123,904 |
| delta+`amin` | 1,538,840 | 58,884 | +124,420 |
| **delta+scalar** | **1,531,164** | 51,208 | +116,744 |

scalar is **7,160 parameters below soft (−0.47%)** — the wasted per-channel head is gone. `reduce` and
`scalar` agree to 3.8e-15 between `forward` and `step` in fp64.

**(6) Cost** (slots 32×m 16, $d$=128, $L$=4, subprocess per cell, 3 interleaved rounds for training,
5 for decode):

| Rule | Training step ms (B8/T512) | Single-step decode ms (B=1) |
|---|---|---|
| soft | 1886.0 / 1841.0 / 1799.1 | 10.377 / 9.442 / 8.268 / 13.757 / 8.683 |
| delta+`amin` | 1558.0 / 1596.4 / 1692.3 | 8.414 / 11.447 / 9.514 / 10.399 / 9.459 |
| delta+scalar | 1633.9 / 1596.9 / 1546.6 | 8.120 / 9.150 / 9.884 / 9.015 / 7.973 |

Training step: scalar's median of 1596.9 ms against soft's 1841.0 ms is **0.867×**, on par with
`amin` (0.867×). Decode: scalar's median of 9.015 ms against soft's 9.442 ms is **0.955×**, winning
**4/5** paired rounds — slightly better than the "comparable" reading for `amin` in §26.7(6), though
soft's own spread reaches 5.5 ms, so it reads as **slightly better** rather than faster.

**Conclusion**: the `amin` scalar collapse is the common root cause of three §26.7 weaknesses
(parameters, stability, seed sensitivity); a scalar decay head fixes all three at once. The price is
that the decay ceiling is now bound by the $h$ parameterization, so the MQAR peak is slightly below
`amin`'s non-collapsing seeds. The default remains `reduce`; `scalar` is opt-in.

**(7) Block length 32 is set by stability, not by speed** (`chunk` in `chunk_delta_scan`):

`scalar` mode pins $p$'s structural floor at 0.5, so $d_{\min}^{-L}$ only reaches 1.84e19 at $L$=64,
which looks like room to enlarge the block. Two separate checks give different answers:

**On speed every block length ties; no ranking is resolvable.** Per-cell subprocesses, round-interleaved
(`experiments/_chunk_bounds.py chunk`; $B$=8, $T$=512, slots 32×$m$ 16, $p$=0.9988), 22 readings pooled
over two batches:

| chunk | n | median ms | spread | vs 32 |
|---|---|---|---|---|
| **32** | 8 | 61.27 | 11.57 | 1.00× |
| 48 | 5 | 69.27 | 14.09 | 1.13× |
| 64 | 5 | 65.10 | **34.55** | 1.06× |
| 128 | 4 | 66.97 | 13.12 | 1.09× |

The median gap (3.8 ms between 32 and 64) is far below 64's own spread (34.55 ms), and round-paired
results flip between batches (3/4 in one, 2/4 tie in the other). **Measuring block-length ranking inside
a single process produces false differences** — the same 64 configuration was recorded at both 50.67 and
85.22. This is the noise floor for this workload, not a speed advantage.

**On stability 32 is a hard boundary.** At $p$=0.5 the length-64 block already yields non-finite decay
gradients (forward stays OK; $1/\prod d$ overflows the backward), and $L$=128 is NaN even in the forward
pass. Sweeping block length gives a maximum stable value of $L$=62, which offers no gain since speed
there is indistinguishable from 32.

So `cap=32` is set by stability: it sits inside the stable region and is not slower. What had been
treated as a "conservative compromise" is in fact a requirement.

**(8) The guard's headroom was 30 bits wider than the measured safety level**:

The analytic guard reverses $\bar d^{-L}\le 2^{\text{headroom}}$ for $L$ and originally used
headroom=80. The measured threshold sits between $2^{57}$ and $2^{62}$ (the "max safe $L$" column is
swept block length by block length):

| $p$ | max safe $L$ | $2^{\log_2(1/\bar d^L)}$ |
|---|---|---|
| 0.3 | 36 | 62.5 |
| 0.4 | 46 | 60.8 |
| 0.5 | 62 | 62.0 |
| 0.6 | 78 | 57.5 |
| 0.7 | 78 | 40.1 |

With headroom=80 the guard returned $L$=24/34 for $p$=0.1/0.2, and both produce non-finite decay
gradients. With headroom=50 every $p\in[0.05,0.9]$ satisfies $\bar d^{-L}\le 2^{50}$ and is stable in
measurement. On the production path $p\ge 0.5$, so by_cond is always $\ge 50 > \text{cap}=32$ and the
block length is unchanged — this fix only affects direct `chunk_delta_scan` calls with $p<0.5$.

---

## 28. The Cost Mechanism of Layer-Stack Scaling and How to Avoid the "Scaling Cliff"

### 28.1 What the Cliff Is

One metric remains at the top of the report TL;DR: **at the same parameter budget, aligning the
layer count to `L=6` makes train steps 30–40% slower than `L=4`** (§16.5). If "more layers means
slower", scaling to a larger size (more layers) would hit a performance cliff, which directly
affects whether SunnyNet can serve as a general large-model backbone. This section moves this
from "phenomenon" to "mechanism and fix".

### 28.2 Cost Law: Train Steps Are Linear in Layer Count, and the Slope Is a Fixed Tax

`experiments/_layer_scaling.py --mode affine` (fixed `d=128 / i=192 / r=64`, sweeping
`L ∈ {1,2,3,4,6,8,10,12}`, T=256, B=8, 6 threads, independent subprocess per cell, 3 interleaved
rounds taking the median):

| L | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| ms/step | 121.4 | 158.9 | 195.6 | 267.8 | 310.2 | 365.8 | 414.2 | 499.5 |
| ms/token | 0.0593 | 0.0776 | 0.0955 | 0.1308 | 0.1515 | 0.1786 | 0.2022 | 0.2439 |

Least squares: **`step_ms ≈ 32.9·L + 102.4`** (marginal 33 ms per layer, layer-count-independent
intercept 102 ms).
⇒ The depth cost is **linear with a constant slope**, not "exploding after some layer".

![Scale and depth](../figures/perf_scaling.png)

Left is stack parameters vs validation ppl (1.4M / 20M / 100M): **not monotonic**, 20M is the worst and 100M recovers,
but the three tiers use different step budgets (600 / 300 / 100), so read the trend only and never subtract across tiers.
Right is the fit of the cost law above.

### 28.3 The Slope Comes from Per-Layer Fixed Operator Dispatch, Not FLOPs

The same profile counts the aten calls added per layer (L=2/4/6/8):

| Path | L=2 | L=4 | L=6 | L=8 | Per-layer slope |
|---|---|---|---|---|---|
| eager torch | 2326 | 4524 | 6722 | 8820 | **+1099** |
| AVX2 fused kernel | 1558 | 2988 | 4418 | 5848 | **+715** |

Grouped by operator (L=4): **`mm` accounts for only 1.9% (84 calls), `elementwise_other` for
95.7% (4329 calls)**, of which roughly 1082 elementwise/view operators per layer (`as_strided`
535, `mul` 232, `slice` 192, `narrow` 180, `view` 139, `transpose` 121…). **There are only 21
`mm` calls per layer.**

⇒ **Each layer is a fixed "operator dispatch tax" (~1100 calls), largely independent of width
and FLOPs.** The fused kernel lowers it from 1099 to 715 (−35%), the only reduction actually
landed. "More layers = per-layer operator set × more times", consistent with the root-cause
judgments of §13.3/§14.2/§16.5.

### 28.4 Segment Attribution of the Per-Layer Tax: Spread Across 5 Segments, No Single Winner

`experiments/_layer_tax.py` uses replacement experiments (zeroing the time of a segment; not
numerically equivalent, measuring only the **upper bound of the gain**) to give the "per-layer
marginal cost after zeroing that segment" relative to the baseline (baseline 37.9 ms per layer):

| Zeroed segment | Per-layer ms after zeroing | Relative to baseline | Note |
|---|---|---|---|
| `scan` | 24.8 | **0.654** | scan (logspace_scan) |
| `readout` | 25.0 | 0.660 | `W_z`+softwedge+`norm_mix` |
| `compete` | 26.7 | 0.704 | competitive gating+gating probabilities |
| `local` | 26.8 | 0.708 | depthwise separable convolution |
| `emit` | 28.0 | 0.739 | `W_c/W_s/b_gain/sn` assembly |
| `norm` | 34.6 | 0.912 | `norm_in` |

⇒ **The tax is highly dispersed**: fully fusing any single segment only cuts it to 0.65–0.74×,
and no segment can own it; fusing all segments (an unreachable upper bound) only brings a layer
to about 24.8 ms. **"Writing one more fused kernel" is not enough to flatten the cliff.**

### 28.5 Under the Same Parameter Budget: Depth Is Worse on Both the Speed and Quality Axes

The "30–40%" was measured **under the same parameter budget**. `experiments/_scaling_fix.py`
enumerates 6 (L, dim) families at a budget of **365,844 (= the `small` stack parameter count)**,
900 steps, B8×T256, 2 seeds (1234 / 2024), taking the median:

| Family | L | dim | inner | rank | ms/token | zh ppl |
|---|---|---|---|---|---|---|
| **L3 wide** | 3 | 136 | 233 | 97 | **0.0995** | 268.92 |
| **L4 wide** | 4 | 121 | 198 | 77 | 0.1130 | **247.17** |
| L8 narrow | 8 | 67 | 106 | 133 | 0.1132 | 313.94 |
| L4 base | 4 | 128 | 192 | 64 | 0.1184 | 287.84 |
| L6 wide | 6 | 96 | 170 | 64 | 0.1338 | 318.51 |
| L6 narrow | 6 | 92 | 145 | 92 | **0.1346** | 312.62 |

**Readings**:

- **Speed**: the shallowest `L3 wide` is fastest (0.0995), the two `L6` families are slowest
  (≈0.1346), a **span of 1.35×**, in the direction consistent with the cost law. `L8 narrow`
  (d67) is instead faster than `L6` — because each layer is narrower, the per-layer tax is
  smaller; the driver is `L × width`, not bare `L`.
- **Quality**: **depth does not buy quality**. `L4 wide` (d121) with ppl **247.17** is the best
  overall; all three `L6`/`L8` families land in 313–319, 27–29% worse than `L4 wide`. Within-seed
  and between-seed variation is about ±34 ppl (`L4 wide` gives 264.5 / 229.9 across two seeds),
  while the gap of ~60 between `L6/L8` and `L4 wide` exceeds the variation
  ⇒ the direction is credible, the magnitude needs more seeds to tighten.
- **Protocol**: 900 steps is not converged, single machine and single implementation; this
  section is for **relative comparison across families**, not an absolute quality conclusion.

⇒ **The so-called "scaling cliff" is not the architecture failing after a certain layer count,
but the low-rank + narrow `dim` design being forced to stack depth when parameters are aligned,
paying a fixed dispatch tax per layer, while depth does not compensate with quality.** This is
both a performance problem and the same root as the quality problem.

### 28.6 Fix: Configuration Discipline First, Not More Fusion

Combining the mechanisms, the actionable fixes by priority:

| # | Measure | Basis | Status |
|---|---|---|---|
| 1 | **Prefer shallow and wide at the same budget**: put parameter count into `inner/rank`, cap the layer count at the sweet spot (measured 4 layers) | §28.5 better on both axes simultaneously; §14.3 the same direction at 20M (L10→L6 gains 12.1% speed, ppl −10.9%) | **adopted (configuration discipline)** |
| 2 | Raise the `rank/dim` ratio, widen `inner`, reducing the incentive to "stack layers to make up parameters" | §14.2 the cause of exceeding peers by +3~+8 layers is low rank + narrow dim | **adopted (preset reconfiguration)** |
| 3 | Fused kernel (already landed) | per-layer operators 1099→715 (§28.3) | landed |
| 4 | Continue fusing `emit/readout/norm` | §28.4 each segment is only 0.66–0.74×, limited upper bound | P2 (diminishing returns) |

**New preset `p100msw`** (`dim=1536/inner=2624/rank=1152/L=6`, stack parameters **94,470,942**,
differing from `p100mw`'s 93,828,904 by **+0.68%**): pushes the 100M layer count from 8 down to 6
and raises the width by 9%. Isolated A/B (`_scaling_fix.run_cell`, B4×T256, 3 interleaved rounds,
an independent subprocess per cell):

| Preset | L | dim | Stack params | ms/token | Relative |
|---|---|---|---|---|---|
| `p100mw` | 8 | 1408 | 93,828,904 | 4.4111 | — |
| **`p100msw`** | 6 | 1536 | 94,470,942 | **4.2247** | **4.2% faster (3/3 rounds better)** |

⇒ **Removing layers and widening measures 4.2% saved, not the 6–8% estimated on paper** (the part
of the estimate not realized is offset by "widening 9%"). The direction is certain and the
magnitude limited: at this layer count the per-layer tax is about 33 ms, but widening also
lengthens the per-layer matmul. **Depth no longer exceeds the peer band, and speed and quality
(§28.5) improve in the same direction.**

**Honest boundary**: this section rewrites the cliff from an "architectural defect" to a
"cost law avoidable by configuration", but it **does not eliminate** the per-layer dispatch tax
itself. To truly make "more layers" not slow down, the per-layer elementwise operators need to be
largely folded into the kernel — §28.4 already gives the upper bound of the gain on this path
(per layer 37.9→24.8 ms as the limit), and §22.5 has shown that the inference mode yields no gain
because the tensors are small. **On an AVX2-only 2018 mobile CPU, this upper bound is a hard
constraint.**

### 28.7 Reproduction

```sh
python -X utf8 experiments/_layer_scaling.py --mode affine --L 1,2,3,4,6,8,10,12 --rounds 3
python -X utf8 experiments/_scaling_fix.py --budget 365844 --steps 900 --rounds 2 \
    --seeds 1234,2024                                   # §28.5 same-budget layer-width families
python -X utf8 experiments/_layer_tax.py 256 2          # §28.4 per-layer tax segment attribution
```

---

## 29. Full Reproduction

```bash
# 1. Environment
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib pytest

# 2. Data
python prepare_data.py
python build_corpus.py --vocab-size 8192

# 3. Learning-rate calibration
python experiments/run_experiments.py --exp probe --steps 260

# 4. Main experiments (each with its own best lr)
python experiments/run_experiments.py --exp baseline --steps 1200
python experiments/run_experiments.py --exp ablation --steps 600
python experiments/run_experiments.py --exp seeds    --steps 600

# 5. Long sequence / 20M
bash experiments/battery2.sh
python experiments/long_sequence.py --only latency

# 6. Same-complexity O(N) comparison (seven architectures)
bash experiments/battery5.sh                                  # same-protocol quality/training cost
python experiments/on_scaling.py --threads 8 --repeats 9 \
    --out experiments/results/on_scaling_t8.json             # speed scaling with T

# 7. Figures and summary
python experiments/make_figures.py
python experiments/collect_results.py --out docs/_tables.md

# 8. T=1024 audit / cross-distribution / short sequence (2026-10-21)
python experiments/_T1024_audit.py 1024 8                    # §17 training-mode speed + peak memory
python prepare_extra.py                                      # collect external new domains (arxiv/news)
python experiments/_ood.py 600 sunnynet,transformer,...      # §18 cross-distribution generalization
python experiments/_shortseq.py 128,256,384 3                # §19 short-sequence training-mode crossover
python experiments/_shortseq2.py 32,64,128,256 8 3           # §20 very short sequence + narrow config
python experiments/_decode_bench.py 32,64,128,256 3          # §20 decode mode (per token)
```

### Hand-written Kernels (§21/§22, requires a C compiler)

```bash
export SUNNY_W64=F:/devtools/w64devkit    # w64devkit (GCC 16.2.0)
python -X utf8 experiments/_kernel_bench.py                  # §21.3 operator-level speedup
python -X utf8 experiments/_core_e2e.py 256 6 8              # §22.3 fused-kernel train step A/B
python -X utf8 experiments/_scan_share.py 256                # §22.4 train step composition
python -X utf8 experiments/_step_e2e.py 128,256,512 1 4      # §22.5 inference-mode A/B
python -X utf8 experiments/_decode_batch.py 256 1,4,8,16,32 3
python -X utf8 experiments/_decode_dim.py 256 3              # §22.5 dim sweep
python -X utf8 experiments/_decode_longpath.py 256,512,1024,2048,4096 2  # §22.10 single step vs context
```

### General Recurrent Runtime (§24, requires a C compiler)

```bash
export SUNNY_W64=F:/devtools/w64devkit
python -X utf8 experiments/_recur_equiv.py                   # §24.3 kernel self-check
python -X utf8 experiments/_unified_ir_map.py                # §24.2 cross-architecture mapping
python -X utf8 experiments/_runtime_bench.py                 # §24.4 vs vectorized baseline
python -X utf8 experiments/_state_quant.py                   # §24.5 state quantization dynamic range
python -X utf8 experiments/_export_runtime.py \
    experiments/ckpts_ood/sunnynet.pt experiments/ckpts_ood/sunnynet.runtime.bin
python -X utf8 experiments/_runtime_model_equiv.py           # §24.7 C vs torch
python -X utf8 experiments/_runtime_decode.py 256,1024,2048 3  # §24.7 decode latency
python -X utf8 experiments/_ab_fp16.py                        # §24.7c fp16 vs fp32
python -X utf8 experiments/_fp16_threads.py                   # §24.7c fp16 thread scaling
python -X utf8 experiments/_rwkv7_gguf_equiv.py              # §24.9 GGUF vs .pth
python -X utf8 experiments/_rwkv7_gguf_bench.py              # §24.9 numpy generation
python -X utf8 experiments/_export_rwkv7.py                  # §24.10 export C blob (needs GGUF)
python -X utf8 experiments/_rwkv7_c_equiv.py 12              # §24.10 C vs numpy
python -X utf8 experiments/_rwkv7_c_bench.py 5               # §24.10 C kernel speed
python -X utf8 experiments/_quant_i4_bench.py                # §24.11 INT4 matrix speed
python -X utf8 experiments/_rwkv7_int4_e2e.py 5              # §24.11 INT4 end-to-end
python -X utf8 experiments/_rwkv7_int4_quality.py            # §24.11 INT4 quality comparison
python -X utf8 -m pytest tests/test_rwkv7_c.py -v
python -X utf8 -m pytest tests/test_quant_i4.py -v
python -X utf8 experiments/_mat_scan_bench.py                # §24.12 matrix-valued kernel operator level and share
python -X utf8 experiments/_export_deltanet.py MDIR MDIR/deltanet.c.bin  # §24.12 export C blob
python -X utf8 experiments/_deltanet_c_equiv.py              # §24.12 DeltaNet C vs numpy
python -X utf8 experiments/_deltanet_c_bench.py 5            # §24.12 DeltaNet C speed
python -X utf8 experiments/_deltanet_check.py                # §24.12 DeltaNet correctness
python -X utf8 experiments/_export_mamba2.py MDIR MDIR/mamba2.c.bin  # §24.12 export C blob
python -X utf8 experiments/_mamba2_c_equiv.py                # §24.12 Mamba-2 C vs numpy
python -X utf8 experiments/_mamba2_layer_probe.py            # §24.12 Mamba-2 per-layer table
python -X utf8 experiments/_mamba2_c_bench.py 5              # §24.12 Mamba-2 C speed
python -X utf8 experiments/_export_gla.py MDIR MDIR/gla.c.bin    # §24.12 export C blob
python -X utf8 experiments/_gla_c_equiv.py                   # §24.12 GLA C vs numpy
python -X utf8 experiments/_gla_layer_probe.py               # §24.12 GLA per-layer table
python -X utf8 experiments/_gla_stage_probe.py 0              # §24.12 GLA intra-layer stage table
python -X utf8 experiments/_gla_c_bench.py 5                 # §24.12 GLA C speed
python -X utf8 -m pytest tests/test_mat_scan.py -v
python -X utf8 -m pytest tests/test_deltanet_c.py -v
python -X utf8 -m pytest tests/test_mamba2_c.py -v
python -X utf8 -m pytest tests/test_gla_c.py -v
python -X utf8 experiments/_export_hgrn2.py MDIR MDIR/hgrn2.c.bin     # §24.12 HGRN2 fp32 blob
python -X utf8 experiments/_hgrn2_check.py                   # §24.12 HGRN2 numpy consistency
python -X utf8 experiments/_hgrn2_c_equiv.py                 # §24.12 HGRN2 C vs numpy
python -X utf8 experiments/_hgrn2_c_bench.py                 # §24.12 HGRN2 C speed
python -X utf8 experiments/_export_hgrn2.py --int4 MDIR MDIR/hgrn2.q4.bin  # §24.12(13) INT4 blob
python -X utf8 experiments/_hgrn2_int4_equiv.py              # §24.12(13) INT4 vs fp32/numpy
python -X utf8 experiments/_hgrn2_int4_probe.py              # §24.12(13) per-layer error localization
python -X utf8 experiments/_hgrn2_int4_sweep.py              # §24.12(13) variant comparison
python -X utf8 experiments/_hgrn2_int4_quality.py            # §24.12(13) real-text quality
python -X utf8 experiments/_export_gla.py --int4 MDIR MDIR/gla.q4.bin  # §24.12(14) GLA INT4
python -X utf8 experiments/_gla_int4_equiv.py                # §24.12(14) GLA INT4 comparison
python -X utf8 experiments/_export_deltanet.py --int4 MDIR MDIR/deltanet.q4.bin  # §24.12(14)
python -X utf8 experiments/_deltanet_int4_equiv.py           # §24.12(14) DeltaNet INT4 comparison
python -X utf8 -m pytest tests/test_hgrn2_c.py -v
python -X utf8 -m pytest tests/test_gla_c.py -v
python -X utf8 -m pytest tests/test_deltanet_c.py -v
```

### Gate attribution and associative-memory write rules (§24.14 / §26.7)

```bash
python -X utf8 experiments/_gate_originality.py jac               # §24.14(1) operator-level Jacobian spectral norm
python -X utf8 experiments/_gate_originality.py grad              # §24.14(2) model-level embedding gradients
python -X utf8 experiments/_gate_originality.py train 600 1234    # §24.14(3) gate-form quality (3 seeds)
python -X utf8 experiments/_chunk_delta.py                        # §26.8 chunked delta vs sequential
python -X utf8 experiments/_assoc_overwrite.py                    # §26.8 soft vs delta write overwrite
python -X utf8 experiments/_assoc_ab.py overwrite 800 1234        # §26.8 overwrite recall A/B (3 seeds)
python -X utf8 experiments/_assoc_ab.py mqar      800 1234        # §26.8 MQAR A/B
python -X utf8 experiments/_assoc_ab.py ppl       600 1234        # §26.8 real-corpus ppl A/B
python -X utf8 experiments/_assoc_ab_speed.py soft 512 32 16 12   # §26.8 write-rule cost (subprocess per cell)
python -X utf8 -m pytest tests/test_architecture.py -v
```

> GGUF weights and the exported blobs are large, fetched on demand via `experiments/gguf/`, not distributed with the repository.

> **Process isolation requirement**: timing measurements must use one subprocess per cell (`experiments/on_scaling.py` has this built in),
> otherwise model construction accumulated in the same process inflates readings by more than 2×, see §2.4.

All raw results are in `experiments/results/`, figures in `docs/figures/`.
