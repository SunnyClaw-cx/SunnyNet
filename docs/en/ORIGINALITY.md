# SunnyNet Component Originality Check

*[中文](../cn/ORIGINALITY.md)*

For each component this lists the prior work, marking what is established and what is designed here.
The goal is to state the claimable scope precisely — neither overclaiming nor giving away what was
genuinely built here.

## 1. Family

The state update is a gated convex combination:

```
s_t = p_t ⊙ s_{t-1} + (1 - p_t) ⊙ u_t
```

This is a first-order linear recurrence. The equivalence of linear attention and linear RNNs was
proven by Katharopoulos et al., 2020 (arXiv:2006.16236); since then GLA, DeltaNet, RWKV, Mamba, and
HGRN all sit inside this paradigm. SunnyNet belongs to this family and does not claim the paradigm as new.

## 2. Component-by-component

### 2.1 Paradigm and formulas (established)

| Component | Prior work |
|---|---|
| First-order linear recurrence / linear attention = linear RNN | Katharopoulos et al., 2020, arXiv:2006.16236 |
| Convex-combination update `H_t = Z_t⊙H_{t-1} + (1-Z_t)⊙H̃_t` | Cho et al., 2014 (GRU), same update shape |
| Matrix-valued state recurrence | RWKV-7, Mamba-2, DeltaNet all use matrix states |
| Per-channel / per-dimension decay | RWKV-4 (`w`, `u` are d-dim vectors), HGRN2 (hierarchical gating) |
| Log-domain parameterized decay `g ≤ 0, exp(g) ∈ (0,1]` | KDA (Kimi Delta Attention), per-key forget gate |
| Multi-timescale / log-bucketed memory | Log-Linear Attention, Guo et al., 2025, arXiv:2506.04761 (logarithmically growing state set) |
| Softmax forget gate over sequence | Forgetting Transformer, arXiv:2503.02130 |
| Sigmoid gating (head-level optimal, multiplicative over additive) | Qwen Gated Attention, arXiv:2505.06708 (NeurIPS 2025 Best Paper) |
| Group normalization / grouped structure | Group Normalization, Wu & He, 2018, arXiv:1803.08494 |
| Short causal conv for position mixing | Mamba (S6), `d_conv=4` |
| Chunkwise parallel scan (in-block closed form + block-head recovery) | Blelloch 1990 prefix sums; FLA library; Mamba / GLA / DeltaNet / RWKV-7 all use it |
| Delta-rule write (subtract the key's old association, then write the new value) | DeltaNet, Yang et al., 2024, arXiv:2406.06484 |
| Chunked delta rule / WY representation `W=(I+L)⁻¹diag(β)(V−K S_in)` | DeltaNet's chunkwise form; FLA library `chunk_delta_rule`; Gated DeltaNet |

### 2.2 Engineering and runtime (established, industry standard)

| Component | Prior work |
|---|---|
| C / SIMD inference kernel (AVX2 / AVX-512 / NEON) | llama.cpp / ggml, rwkv.cpp |
| INT4 block-wise quantization | llama.cpp k-quant family, ggml generic INT4 GEMV |
| GGUF weight format | Defined by the llama.cpp / GGML team, de-facto local-inference standard |
| OpenAI-compatible API (`/v1/chat/completions` + SSE) | llama.cpp server, Ollama, vLLM |
| mmap read-only load of large weights | llama.cpp, llamafile |
| safetensors / HF weight loading | HuggingFace transformers |
| Decode bound by bandwidth, not FLOPs | memory wall, Wulf & McKee, 1995 |

### 2.3 Designed here (claimable)

| Component | Form | Nearest prior work and difference |
|---|---|---|
| SoftWedge activation | `f(z)=z+a·s(z)`, base `s(z)=½+½·z/(1+\|z\|)` | Mish / SiLU are also self-gated activations; SoftWedge's base and derivative range `f' ∈ (1, 1+a/2]` have no identical form found |
| CenteredScaleNorm | centered variance + smooth floor `var + s_min² + eps` | Idea follows RMSNorm / LayerNorm; the "smooth floor to stop near-constant vectors from being amplified" form is self-designed |
| Spectral-anchoring init | QR orthogonal slices, `σ_max` anchored directly | Orthogonal init from Saxe et al., 2013; anchoring the spectral radius directly is self-designed |
| Intra-group pairwise competition gate | `σ(g_i-g_next(i))·σ(g_i-g_next2(i))`, rolling two-term product in a group | Motivation as in Qwen gating; the specific two-sigmoid rolling-product form has no identical usage found |
| Cross-layer state relay + feature residual stream | state bus + identity residual shortcut | Residual is established (He et al., 2016); the combination and its measured calibration are self-designed |
| Generic first-order-recurrence C kernel | `sn_recur_scan` / `step` / `mat_scan`, one abstraction covering SunnyNet / RWKV-7 / Mamba-2 | ggml has a general kernel concept, but the abstraction granularity for "first-order recurrence" differs |

## 3. Independent observations

Measured data nobody could have held in advance:

- Bandwidth ledger and sequential-read ceiling (`bench_bw.c`).
- Head-to-head of llama.cpp and the SunnyNet C kernel at the same weights and precision.
- Same-parameter cross-architecture BPC comparison and same-domain small-model evaluation.
- Location and fix of the AVX-512 tail-path pointer-binding bug.
- Head-to-head of soft (convex-combination) and delta writes on the overwrite-recall task, plus the
  location of the chunked delta's fp32 long-block divergence (`AminBackward0`) and the block-length cap.
- The structural finding that the write key must come from the previous position for (key→value)
  associations to be stored at all (k/v/q from the current position can only express token→token, and
  both rules stay at chance).
- Measurement that the chunked delta write is cheaper than the soft write: soft's `log`/`cumsum`/`exp`
  all run at width $\text{slots}\cdot m$, while delta keeps the state at the narrow $(\text{slots},m)$
  shape (training step 0.51–0.89×, widening with slots×m).
- **The decay-reduction direction**: collapsing the per-channel decay to a scalar with `amin` (the
  fastest-forgetting channel) drives the block's peak $1/\prod d$ condition number to 9.92e16 at
  chunk=64, against 1.78 for `amax`; `amin` also keeps only 1 of 16 channels. This is the basis for the
  scalar decay head (`assoc_decay='scalar'`) and for $p$ having to reuse the H parameterization's
  lower bound — an independent observation.
- Four design conclusions overturned and corrected by measurement (see ARCHITECTURE.md §3/4/5 and README).

## 4. Claimable boundary

**Ours**: the specific forms of the SoftWedge activation, CenteredScaleNorm, and spectral-anchoring
initialization; the specific form of the intra-group pairwise competition gate; the combination of the
cross-layer state relay; the generic first-order-recurrence C kernel abstraction; and all measured data.

**Not ours**: the convex-combination state update, the delta-rule write, the chunked delta / WY
representation, multi-scale forgetting, log-domain decay, sigmoid gating, chunkwise parallel scan, and
the engineering pieces (SIMD / quantization / GGUF / OpenAI API). `ops.chunk_delta_scan` is an
implementation of the delta rule's chunked form in this codebase; the algorithm itself belongs to the
DeltaNet / FLA line.

In one line: this is a variant of the linear-recurrence architecture with its own tradeoffs in component
selection and combination; the closest comparison is KDA relative to Gated DeltaNet — an in-field
variant, not a new paradigm.

## References

- Katharopoulos et al., *Transformers are RNNs*, 2020. arXiv:2006.16236
- Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder*, 2014. arXiv:1406.1078
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*, 2024. arXiv:2406.06484
- Wu & He, *Group Normalization*, 2018. arXiv:1803.08494
- Qiu et al. (Qwen), *Gated Attention for Large Language Models*, 2025. arXiv:2505.06708
- Guo et al., *Log-Linear Attention*, 2025. arXiv:2506.04761
- Lin et al., *Forgetting Transformer*, 2025. arXiv:2503.02130
- Saxe et al., *Exact solutions to the nonlinear dynamics of learning in deep linear networks*, 2013. arXiv:1312.6120
- Wulf & McKee, *Hitting the Memory Wall*, 1995. ACM SIGARCH Comput. Archit. News
- Blelloch, *Prefix Sums and Their Applications*, 1990