*[中文版](../cn/ARCHITECTURE.md)*

# SunnyNet Architecture

> This document gives all mathematical definitions and design rationale of SunnyNet.
> SunnyNet belongs to the linear-attention / linear-RNN family — its state update is a gated convex
> combination, and first-order linear recurrence is the shared paradigm of that field since 2020.
> What is specific here is the form of the activation, normalization, initialization, and gating.
> A component-by-component comparison against prior work is in [ORIGINALITY.md](ORIGINALITY.md).

## 0. Design Starting Point

On CPU, pretraining bottlenecks are usually not in floating-point throughput but in:

1. **Memory bandwidth**: every training step moves activations, gradients, and optimizer state between DRAM and cache;
2. **Cache hit rate**: once matrix dimensions exceed L2/L3, the hit rate falls off a cliff and effective throughput is dragged down by memory access;
3. **Sequence seriality**: token-by-token recurrence leaves the sequence dimension unparallelizable, so multiple CPU cores are underused.

All of SunnyNet's design is therefore organized around three hard constraints:

| Constraint | Corresponding design |
|---|---|
| Sequence complexity must be strictly O(N) | Rolling state + first-order linear recurrence, no token-to-token pairwise interaction |
| No intermediate activations of the whole sequence are stored | State dimension is fixed, only state is passed across blocks; backward recomputes on demand |
| Matrix dimensions kept small-to-medium, giant dense multiplications avoided | Low-rank factorization + depthwise separable convolution + tridiagonal circulant filtering |

At the same time, **the long-range capability of general language modeling is not sacrificed**; §1.3 describes how this is achieved.

---

## 1. Base State Update Unit (`SunnyBlock`)

### 1.1 Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $x_t$ | input vector at step $t$ | $\mathbb{R}^{d}$ |
| $s_t$ | global rolling state at step $t$ | $\mathbb{R}^{d}$ |
| $z_t$ | concatenated intermediate | $\mathbb{R}^{2d}$ |
| $u_t$ | candidate state | $\mathbb{R}^{d}$ |
| $q_t$ | low-rank projection output | $\mathbb{R}^{2d}$ |
| $S_t$ | state sequence produced by the scan | $\mathbb{R}^{d}$ |
| $d$ | backbone dimension; $d_i$ is the nonlinear branch dimension; $r$ is the projection rank | — |

### 1.2 Forward Computation

**(a) State circulant filtering.** The state first passes through a 3-tap circulant filter for lightweight channel mixing:

$$
\tilde{s}_t \;=\; \tau_0\,\mathrm{roll}(s_t, +1) \;+\; \tau_1\,s_t \;+\; \tau_2\,\mathrm{roll}(s_t, -1),
\qquad \tau \in \mathbb{R}^3 \ \text{learnable}
$$

It is equivalent to right-multiplying $s_t$ by a tridiagonal circulant matrix, but performs only shifts and elementwise multiply-add: **no matrix multiplication and no extra memory**.

**(b) Concatenation and low-rank projection.** After normalization, the input is concatenated with the filtered state along the feature dimension (**no additive bypass**), then projected through a low-rank factorization:

$$
z_t = \big[\,\mathcal{N}(x_t) \,;\, \tilde{s}_t\,\big] \in \mathbb{R}^{2d},
\qquad
q_t = U\big(V(z_t)\big),\quad V \in \mathbb{R}^{r \times 2d},\ U \in \mathbb{R}^{2d \times r}
$$

Parameter count and floating-point cost drop from $O(d^2)$ to $O(dr)$. When $r \ll d$, the memory traffic of this step is far below that of a dense matrix of equivalent capacity.

**(c) Candidate state (bounded).** $q_t$ is split into $(u^{\text{raw}}_t, g_t)$. $g_t$ is chunked into blocks of `group` channels each, and **pairwise competitive gating** is applied within each group:

$$
w_i \;=\; \sigma\!\big(g_i - g_{\text{next}(i)}\big)\cdot\sigma\!\big(g_i - g_{\text{next2}(i)}\big)
\in (0,1),
\qquad
u_t = u^{\text{raw}}_t \odot w_t
$$

$w_i < 1$ makes $u_t$ bounded, which is the precondition for the subsequent "convex combination" to keep the state from diverging.
It is chosen over softmax for an engineering reason: softmax needs a max-subtract and an exp, while pairwise competition needs two sigmoids. Both express the "multiple channels contending for one resource" semantics.

> **A correction, kept on record**: an early design attributed the depth gradient decay to
> "the softmax Jacobian contains a $1/d$", and switched to pairwise competition on that basis.
> Measurement later refuted the attribution: the Jacobian spectral norm of softmax **rises** with
> channel count ($d{=}16 \to 512$: 0.29 → 0.36) and does not decay with depth; pairwise competition is
> 0.46 → 0.47, both $O(1)$. The real causes of the gradient decay were the inverted spectral-radius
> sign and the missing residual, both since corrected (§4 / §5). The pairwise competition gate is kept,
> but for the engineering tradeoff, no longer as a "fix for dying gradients".
> Data in experiments/results/_gate_originality_jac.json.

**(d) Nonlinear branch.** It serves only the output and does not enter the state recurrence, so it does not break the linearity of the scan:

$$
y_t \;=\; \mathcal{N}\!\big(\,\mathcal{A}\big(W_z u_t\big)\big)
$$

**(e) Data-driven protection gate.** The per-channel retention rate is determined by the input itself:

$$
\delta_i \;=\; \frac{1}{H_i},\qquad
p_{t,i} \;=\; 1 - \delta_i\,\big(1 - \sigma(g_{t,i})\big),\qquad
p_{t,i} \in [1-\delta_i,\; 1]
$$

$H_i$ is the **memory horizon** of channel $i$ (unit: tokens), sampled uniformly on a logarithmic scale.
$\theta_i$ is a **learnable** parameter (initialized so that $H$ is log-uniform); the decays in comparable work (Mamba/RWKV/H3) are all learnable, and freezing it into a buffer would prevent the model from choosing its own memory scale:

$$
H_i = \exp\!\Big(\ln H_{\min} + s(\theta_i)\,\big(\ln H_{\max} - \ln H_{\min}\big)\Big),\qquad s=\sigma(\cdot)
$$

$\theta$ is initialized by `horizon_logit` (channel midpoint $s=(i+0.5)/d$, then logit, avoiding $\pm\infty$).

Defaults are $H_{\min} = 2$, $H_{\max} = 8192$ (20M prototype); scaled to 0.5B, $H_{\max} = 32768$.

**(f) State recurrence (convex combination).**

$$
s_{t} \;=\; p_{t} \odot s_{t-1} \;+\; \big(1 - p_{t}\big) \odot u_{t}
$$

Since both coefficients are non-negative and sum to 1, there is an **elementwise convex combination**

$$
|s_{t,i}| \;\le\; \max\big(|s_{t-1,i}|,\, |u_{t,i}|\big)
\;\Longrightarrow\;
\|s_t\|_\infty \le \max\big(\|s_0\|_\infty,\ \max_{k\le t}\|u_k\|_\infty\big)
$$

**The state is inherently bounded and cannot diverge through additive accumulation.** This is the fundamental
structural difference between this architecture and LSTM-like ones ($c_t = f \odot c_{t-1} + i \odot \tilde{c}$,
where new information is injected **additively** and has no upper bound).

**(g) Output.**

$$
S_{1:T} = \mathrm{Scan}\big(u_{1:T},\, p_{1:T},\, s_0\big)
\qquad
o_t = W_c\, y_t + \beta\, S_t + W_s\,\mathrm{circ}(S_{\text{prev}}),\qquad \beta \ \text{learnable}
$$

$S_t$ carries historical information; $y_t$ is a purely local nonlinear transform. The block output is then added to the input as a residual
($x^{(l+1)}_t = o^{(l)}_t + x^{(l)}_t$, see §5). The end-of-block state is handed over: $s_{t+1}^{\text{next}} = S_T$.

### 1.3 How Long-Range Dependencies Are Realized

The **lower bound** of the retention rate $p_{t,i}$ is $1 - \delta_i$, so the single-step forgetting of channel $i$ is at most $\delta_i = 1/H_i$.
Without gating protection, the memory time constant of channel $i$ is

$$
\tau_i \;=\; \frac{1}{1-p_{t,i}} \;\le\; H_i
$$

$H_i$ sweeps $[2,\ 8192]$ on a logarithmic scale, meaning that **within one layer there simultaneously exist** all time scales from "remembering only 2 tokens" to "memory spanning 8192 tokens". This directly matches the needs of general language modeling:

- Syntax / local collocation → handled by channels with small $H_i$ plus the depthwise separable convolution;
- Paragraph-level, cross-code-block, and long-range reference in classical Chinese philology → handled by channels with large $H_i$.

Because $p$ is **elementwise** (rather than one scalar shared per channel) and $\sigma(g)$ is determined by the current input,
the model can dynamically decide "how long to remember at this moment" under different contexts.

### 1.4 Chunked Parallel Scan

The recurrence $s_t = p_t s_{t-1} + (1-p_t)u_t$ is affine, so a **closed-form solution** exists within a chunk. Let

$$
P_j = \sum_{k \le j} \ln p_k,\qquad g_k = (1-p_k)\,u_k
$$

then

$$
\boxed{\;s_j \;=\; \exp(P_j)\,\Big(s_{\text{in}} \;+\; \sum_{k \le j} g_k \exp(-P_k)\Big)\;}
\qquad (\star)
$$

Two inclusive `cumsum` operations yield **all** states of the whole chunk; only $T/\text{chunk}$ serial steps are needed across chunks.
$(\star)$ is **algebraically equivalent** to the step-by-step recurrence, not an approximation.

**Inter-chunk parallelization.** The key is that $(\star)$ is **strictly affine** in the chunk-initial state $s_{\text{in}}$, and

$$
s_j\big|_{j=T} = p_T\, s_{\text{in}} + (1-p_T)\,u_T
\quad\Longrightarrow\quad
s_{\text{in}} \;=\; \frac{S_T - (1-p_T)u_T}{p_T}
$$

that is, **the chunk-initial state can be exactly back-solved from the chunk-final state**. All chunks can therefore be run in
parallel first, then the chunk-initial states are back-filled chunk by chunk, reducing the serial depth along the sequence
dimension from $O(T)$ to $O(T/\text{chunk})$. This matches the "parallel within a chunk, serial across chunks" approach of the
Flash-Linear-Attention family, but here the back-solve is an **exact identity** that holds unconditionally.
See `sunnynet/layers.py` and `tests/test_architecture.py::test_block_chunk_invariance_*` for the implementation.

### 1.5 Reversible Computation

The recurrence for $s_t$ is invertible: $s_{t-1} = (s_t - (1-p_t)u_t)/p_t$ follows directly from $(p_t, u_t, s_t)$.
Backpropagation therefore **does not need to store** the intermediate state tensor $S_{1:T}$; it can be reconstructed during the backward
pass from $(p_t, u_t)$. $(p_t, u_t)$ are determined by $x_t$ and the chunk-initial state, so the cost is one local recomputation,
in exchange for activation memory savings on the order of $T \times d$.

### 1.6 Optional Associative Memory Branch (`AssocMemory`)

The main state $s_t$ (per-channel scalars) has an upper bound on exact recall capacity: each layer has only $d$ scalar slots,
and on MQAR associative recall, 2 key-value pairs reach only about 42% (attention reaches about 93%).
`AssocMemory` adds a **matrix-valued** memory to cover this capacity:

$$S_t = p \odot S_{t-1} + (1-p)\odot (v_t \otimes k_t), \qquad o_t = W_o(S_t^{\top} q_t)$$

$S$ has shape $(B, \text{slots}, m)$; $k/q$ are L2-normalized and then mapped to the slots dimension, and $v$ is mapped to the $m$ dimension.

Design points:

- **The injected term is multiplied by $(1-p)$**: this makes $|S_t| \le \max(|S_0|, |v \otimes k|)$ hold elementwise at all times,
  the same convex combination as the main state. The cost is that writing the same key repeatedly only accumulates and cannot overwrite:
  writing 8 keys then overwriting 4 of them, the convex combination reads back mostly the old value (relative error 0.75–0.92, depending on $p$),
  while the delta write reads back the new value (relative error 0.06). See `experiments/_assoc_overwrite.py`.
- **Optional delta write** (`assoc_write='delta'`): $S_t = p \odot S_{t-1} - \beta k_t(k_t^{\top}S_{t-1}) + \beta (v_t \otimes k_t)$,
  subtracting the key's old association before writing the new value, so it overwrites exactly. $\beta=\sigma(W_b x)$ is learnable.
  It was believed delta had to be per-token serial and would destroy the O(N) scan — that step was wrong:
  delta has a WY chunked form (in-block $S_t = S_{\text{in}} + K_{0:t}^{\top}W_{0:t}$,
  $W=(I+L)^{-1}\operatorname{diag}(\beta)(V-K S_{\text{in}})$, $L_{ij}=\beta_i(k_i\cdot k_j)$, lower triangular),
  implemented as `ops.chunk_delta_scan`, matching the sequential delta to 1e-16 in fp64 (any chunk size, with or without decay);
  see `experiments/_chunk_delta.py` and `tests/test_architecture.py::test_chunk_delta_scan_matches_step_fp64`.
  The cost is one extra $C\times C$ triangular solve per block. In practice it is cheaper than the
  soft write: soft materializes the outer product into a $(B,T,\text{slots}\cdot m)$ tensor and scans
  it, so `log`/`cumsum`/`exp` all run at width $\text{slots}\cdot m$ (`cumsum` alone is 38% of soft's
  time); delta keeps the state at $(\text{slots},m)$ and drops the elementwise work to narrow widths.
  The training step is 0.84–0.89× (slots 32×16), 0.58–0.65× (64×32) and 0.51–0.62× (128×64) of soft,
  at a parameter cost of one `W_b` (+0.034%). See §26.7(6).
- **The direction of the decay reduction matters** (`assoc_decay`): the chunked form only accepts a
  scalar decay, so the per-channel head `W_g`'s $(B,T,m)$ output must be collapsed to $(B,T)$.
  Collapsing with `amin` (fastest-forgetting channel) drives the block's peak $1/\prod d$ condition
  number to **9.92e16** at chunk=64, against **1.78** for `amax`; `amin` also keeps only 1 of 16
  channels and leaves the other 15/16 as dead parameters. `assoc_decay='scalar'` instead outputs
  (keep-rate $p$, write strength $\beta$) directly from a dedicated `W_gd(dim→2)` with no reduction.
  $p$ must follow the same H parameterization as the per-channel head ($p=1-1/H \ge 1-1/h_{\min}$,
  i.e. $p \ge 0.5$); a bare $\sigma$ lets $p$ fall to 0.01 and the run diverges. After the fix:
  7,160 fewer parameters than the soft write (−0.47%), the MQAR seed collapse is gone, and neither
  ppl nor overwrite recall regresses. See §26.9.
- **A block length of 32 is set by stability, not by speed**: once `scalar` pins $p \ge 0.5$,
  $d_{\min}^{-L}$ only reaches 1.84e19 at $L$=64, which looks like room to enlarge. Speed shows no
  resolvable difference — across 22 cross-process readings the median ratio is 0.94× for 32/64 and
  0.91× for 32/128, while 64's own spread (34.55 ms) dwarfs the 3.8 ms median gap (ranking block lengths
  inside one process yields false differences). Stability is the hard result: at $p$=0.5 the length-64
  block already yields non-finite decay gradients, and $L$=128 is NaN in the forward pass. So `cap=32`
  is set by stability; being no slower is incidental. The guard's headroom was $2^{80}$, 30 bits wider
  than the measured safety level ($2^{57}$–$2^{62}$), and returned unstable lengths for $p$=0.1/0.2;
  it is now $2^{50}$. On the production path ($p \ge 0.5$) the block length is unaffected.
  See §26.9(7)(8).
- **The write key comes from the previous position** (`assoc_shift`): the write is meant to express a
  (key → value) association, but at the moment the key is recognized the value has not been seen yet.
  If k/v/q are all taken from the current position, the memory can only express token → token
  associations and structurally cannot store (key → value) — which is exactly why early recall
  experiments left both write rules at chance (0.02 vs. 1/32 ≈ 0.031). When enabled, k is taken from
  the previous position (position 0 uses the most recent entry in the conv history), and the delayed
  write adds no extra state. `forward` and `step` agree to 1e-6 across four write rules × two shift
  settings (`tests/test_architecture.py::test_assoc_shift_forward_step_matches_fp64`).
- **Reuse one scan primitive**: the recurrence for $S$ is isomorphic to the main state (first-order linear convex combination),
  so it calls `logspace_scan` directly and inherits chunked parallelism, single-step vs. whole-segment consistency, and boundedness.
- **Width decoupled from $d$**: taking the full per-head width $H m^2 = d^2/H$ makes the measured training-step multiplier grow
  quadratically with $d$ ($d$=64/128/256 giving 6.9×/8.7×/16.5×), which is unusable at production widths.
  Fixing slots×m decouples the cost from $d$. Measured ($d$=256, $L$=2, B8/T256):
  state=128 → 1.03×, 512 → 1.34×, 1024 → 2.48×.
- **Off by default** (enabled with `ablation={'assoc': True}`). When off, `SunnyState.mem` is `None`,
  the block interface still returns a triple, and existing paths are bit-for-bit unchanged.
- **Can be enabled per layer** (`assoc_every=k`: one enabled every $k$ layers, 0=all enabled). Disabled layers build no
  `AssocMemory`, with no parameters and no computation.

Measured recall (real architecture, $d$=64, $L$=2, 1200 steps; slots 32 × m 16):

| Pairs | Off | On |
|---|---|---|
| 2 | 0.418 | **0.961** |
| 4 | 0.176 | 0.547 |
| 8 | 0.023 | 0.043 |

Capacity-cost frontier ($d$=128, 2 pairs, 1000 steps): state 512 → recall 0.875, training 3.2×;
state 1024 → 0.973, 5.9×. **Recall capacity can be continuously tuned via the state width**,
which the main state cannot do.

Per-layer cost and capacity (small preset 1.42M, $L$=4; `experiments/_assoc_layers.py` /
`_assoc_recall_layers.py`, 3 independent subprocess rounds):

| Config | Enabled layers | Training multiplier | 2 key pairs | 4 key pairs |
|---|---|---|---|---|
| Off | 0 | 1.00× | 0.984 | 0.121 |
| Every 4 layers | 1 | 1.26× | 0.977 | 0.441 |
| Every 2 layers | 2 | 1.46× | **0.988** | **0.656** |
| Every layer | 4 | 1.93× | 0.977 | 0.164 |

Training overhead is **about +22% per layer** (linear). Recall capacity is **non-monotonic** in the number of enabled layers:
on 4 key pairs, every 2 layers (0.656) far outperforms every layer (0.164), i.e. **enabling all is both slower and worse;
half-density enabling is currently the optimal form**.

Converged-state comparison at equal budget (32.8M tokens, `experiments/_converge_compare.py`):

| Model | Stack parameters | zh ppl | Generation |
|---|---|---|---|
| Baseline | 365,844 | 143.24 | recognizable syntax, semantic drift |
| +assoc (every 2 layers) | 427,796 | **138.82** | coherent classical Chinese syntax |
| Transformer | 369,200 | 140.40 | — |

Associative memory every 2 layers lowers zh ppl by 3.1% with only +17% stack parameters, and is the only configuration that
improves generation coherence. Baseline vs Transformer ppl (143.24 / 140.40) and near recall (both 0.023) are statistically
close ⇒ at equal budget, "incoherence" is a scale issue, not unique to the architecture.
(1500 steps is not converged; read as sample efficiency, not converged capacity.)

---

## 2. Original Nonlinear Activation: SoftWedge

$$
\boxed{\;\mathcal{A}(z) \;=\; z \;+\; a \cdot s(z),\qquad s(z) = \tfrac12 + \tfrac12\cdot\frac{z}{1+|z|},\qquad a \in (0,1)\;}
$$

**The derivative has strict bounds on both sides** (a hard design goal):

$$
\frac{\mathrm{d}s}{\mathrm{d}z} = \frac{1}{2\,(1+|z|)^2} \in \Big(0,\ \tfrac12\Big]
\quad\Longrightarrow\quad
\mathcal{A}'(z) = 1 + a\,s'(z) \;\in\; \Big(1,\ 1 + \tfrac{a}{2}\Big]
$$

- **Positive lower bound 1**: gradients never vanish at any scale, so there is no "gradient cliff";
- **Finite upper bound $1+a/2$**: $\mathcal{A}' \le 1.5$, which combined with the spectral anchoring initialization of §4 keeps the multi-layer product gain bounded;
- **Asymptotically linear**: as $z \to +\infty$, $\mathcal{A}(z) \to z+a$; as $z \to -\infty$, $\mathcal{A}(z) \to z$.
  Over large values it is the identity map, so long-range information passes through losslessly;
- **No exponential or trigonometric operations**: $s(z)$ needs only addition, absolute value, and one division, making it cheaper than GELU/SiLU on CPU.

This is shape-similar to GELU (weighted by the Gaussian CDF $\Phi(z)$), SiLU ($z\sigma(z)$), and Mish, but
**the functional form and derivative behavior are entirely different**: the derivative of those three tends to 0 as $z \to -\infty$,
whereas $\mathcal{A}'$ is always $\ge 1$.

Measured (`tests/test_architecture.py::test_softwedge_gradient_bounds_analytic`):
over $z \in [-10^6, 10^6]$ the measured derivative range is $[1.000067,\ 1.249998]$, matching the analytic bound $(1, 1.25]$.

---

## 3. Original Normalization: CenteredScaleNorm

$$
\mathcal{N}(x) \;=\; \gamma \odot \frac{x - \mu(x) + c}{\sqrt{\sigma^2(x) + s_{\min}^2 + \epsilon}}
$$

where $\mu(x)$ and $\sigma^2(x)$ are the mean and variance over the last dimension (biased estimates),
$\gamma, c \in \mathbb{R}^{d}$ are learnable, and $s_{\min} = 0.02$.

Two design points of its own:

1. **$c$ is a per-dimension learnable center, not a statistic.** For a **state** that is repeatedly read and written,
   treating the center as a trainable parameter is more appropriate — it expresses "where the zero point of this channel is".
2. **The denominator has a smooth lower bound $s_{\min}$.** When the input is nearly constant ($\sigma^2 \to 0$),
   existing schemes amplify by $\sqrt{\sigma^2+\epsilon}^{-1} \sim 1/\sqrt{\epsilon}$, blowing floating-point noise up into effective signal;
   this design clamps the maximum amplification to $1/s_{\min} = 50$.

**The value of $s_{\min}$ is the third measurement-driven correction (the old value 0.25 was wrong)**: $s_{\min}$ must be
**below the actual activation scale**, otherwise normalization degenerates into a constant scaling. In the `small` preset,
the measured norm input std of each layer is as low as $0.06$ and the embedding std is only $0.02$, while the old default clamp
threshold of $1/s_{\min} = 4$ ($0.25$) is above these values — so every input is in fact a "near-constant branch", the denominator
is always $0.25$, the output is merely a $4\times$ constant scaling of the input, and **normalization never actually happens**.
The consequence is that deep-layer channel scale drift cannot be suppressed, and ppl was measured to degrade by about 26%
within 600 steps (see experiment report §12). The final choice is $s_{\min} = 0.02$ — a sweep shows ppl decreasing monotonically
with $s_{\min}$, with a plateau in $[0.01, 0.025]$; a value inside the plateau is taken (the $50\times$ clamp then only affects
genuinely constant channels with $\mathrm{std} < 0.02$).

Measured (`test_csn_constant_input_stability`): a constant input $\mathbf{3.14159}$ gives an output magnitude $< 10^{-3}$;
`test_csn_unit_variance`: an input with $\mathrm{std}=7$ is normalized to std $\approx 1$.
Against LayerNorm: under near-constant input ($5.0+10^{-7}$ noise), the CSN output std $=1.0\text{e-}6$ versus $1.6\text{e-}5$ for LN,
i.e. **CSN suppresses noise amplification 15× more strongly** (§12.4).

---

## 4. Original Initialization: Spectral Anchoring

No scheme that infers a variance distribution from fan-in/fan-out is used (Xavier/He/MSRA are all of this kind).
This scheme **directly controls the singular value spectrum of the weight matrix**:

1. Draw a Gaussian matrix $G \in \mathbb{R}^{n\times n}$, compute the QR decomposition $G = QR$, and apply the sign correction from $\mathrm{diag}(R)$ to obtain an orthonormal basis $Q$;
2. Take an orthogonal slice $Q_{[:m,:k]}$ of $Q$ and scale it to the target spectral radius: $W = \rho \cdot Q_{[:m,:k]}$, so that $\sigma_{\max}(W) = \rho$;
3. The spectral radius is a **constant** $\rho = 1.0$, with no depth decay ($\beta = 0$).

**Rationale**: an orthogonal slice has condition number 1, so the forward gain spectrum is flat (all directions equally weighted).
The choice $\rho = 1.0$ is calibrated by measurement — see the table below.

**Why there is no depth decay (the second measurement-driven correction)**:

The first version used $\rho_l = \rho_0 (l+1)^{-\beta}$ ($\beta = 0.15$) to make deeper weights smaller, on the grounds of
"offsetting the product gain of deep stacking". Measurements proved the sign was inverted, and it directly created vanishing gradients:

| $\rho_0$ | $\beta$ | $L{=}8$ embedding gradient | $L{=}16$ | $L{=}24$ | 8→16→24 ratio |
|---|---|---|---|---|---|
| 1.15 | 0.00 | 1.35e+0 | 1.73e+0 | 2.50e+0 | 1.28 / 1.44 (stable) |
| 1.15 | 0.10 | 3.44e-1 | 3.11e-2 | 5.17e-4 | 0.090 / 0.017 (collapse) |

A further sweep over $\rho_0$ ($\beta = 0$):

| $\rho_0$ | $L{=}4$ | $L{=}8$ | $L{=}16$ | $L{=}32$ | Conclusion |
|---|---|---|---|---|---|
| 0.50 | 2.94e-6 | 6.42e-11 | 5.46e-20 | 0 | strong vanishing |
| 0.85 | 1.42e-2 | 2.16e-3 | 4.22e-5 | 1.76e-7 | still vanishes |
| **1.00** | **1.86e-1** | **3.38e-1** | **5.66e-1** | **6.58e-1** | **stable across depth** |
| 1.15 | 9.81e-1 | 1.35e+0 | 1.73e+0 | 4.17e+0 | slight growth |

The final choice is therefore $\rho_0 = 1.0$, $\beta = 0$: the single-block Jacobian gain is calibrated near 1,
so gradients neither explode nor vanish as depth increases. Measured from $L = 2 \to 32$, the embedding gradient
goes $1.46\text{e-}1 \to 6.58\text{e-}1$, and the gradient norms of the first and last block weights are of the same order
(5.1e-2 vs 5.1e-2), with no "only shallow layers are learning" phenomenon.

> **Correction note**: the two tables above were measured under the old CSN ($s_{\min}=0.25$, no real normalization).
> Re-measured after the CSN fix (§3), the collapse boundary shifts down overall: $\rho_0=0.5$ no longer collapses
> (24-layer/4-layer gradient ratio $1.01$), and $\beta=0.15$ no longer triggers collapse on its own ($1.45$);
> collapse appears only on the $\rho_0 < 0.5$ side. The final conclusion of $\rho_0=1$, $\beta=0$ is unchanged.

Measured: the first-step loss after initialization is $= 6.2383$, and $\ln V = \ln 512 = 6.238$ — the initial distribution
is exactly near-uniform, with the correct scale.
See `test_init_loss_near_uniform`, `test_spectral_init_singular_values`.

---

## 5. Cross-Layer Information Transfer (State Relay + Residual)

Two paths carry information between layers:

1. **State relay (main path)**: the end-of-block state $S^{(l)}_T$ of block $l$ serves directly as the block-initial state $s^{(l+1)}_0$ of block $l+1$.
   The state scale is pinned by the convex-combination property of §1.2(f) ($\|s\|_\infty$ never increases), so passing it layer by layer **does not amplify**.
2. **Feature stream (auxiliary path)**: $o^{(l)}_t$ serves as the input $x_t$ of block $l+1$, re-entering through $\mathcal{N}$ and the low-rank projection,
   and at the block output it is **added to the input as a residual**: $x^{(l+1)}_t = o^{(l)}_t + x^{(l)}_t$.

**The residual was added in the fourth correction round (the old design explicitly excluded it, and measurements proved that exclusion wrong)**.
The original design let $o_t$ wholly replace $x_t$, leaving state relay as the only strict inter-layer path. Measurements
(seven architectures under the same protocol, 600 steps, multiple seeds) showed that this "purity" was the **largest single quality loss**:

| Config | ppl (seed 1234 / 2024 / 3407) |
|---|---|
| No residual (old default) | 93.82 |
| **With residual (current default)** | **67.40 / 70.72** |
| Control: LayerNorm + residual | 54.90 / 54.58 / 56.58 |
| Reference: H3-2 | 53.93 |

The residual alone brings a **−28.2% to −32.5%** relative ppl improvement, the largest and most reproducible of all changes
tried (the others: learnable decay gave no gain, stacking state_norm gave no gain, and $h_{\max}$ adjustment and nonlinearity
enhancement were both ineffective). Rationale: the residual provides an **identity shortcut that does not pass through state relay**,
so deep layers need not carry all information on the state bus alone, and "depth gradient collapse caused by under-scaling" no
longer occurs (`test_residual_removes_depth_collapse`: at $\rho_0=0.3$, the 24-layer/4-layer gradient ratio is $1.1\text{e-}4$
without the residual and $1.48$ with it).

The earlier argument that "spectral anchoring alone preserves gradients without a residual" holds only for the old CSN
(which did not truly normalize, see §3); after the CSN fix that guarantee disappears, and the residual changes from "optional" to "necessary".

| Mechanism | Effect |
|---|---|
| $\mathcal{A}' \in (1, 1.5]$ | the nonlinearity never discounts gradients |
| $\mathcal{N}$'s $s_{\min}$ lower bound | removes "variance → 0 ⇒ gradient explosion" while keeping the output scale constant |
| Spectral anchoring $\rho=1$, $\beta=0$ | single-block Jacobian gain calibrated near 1, no depth decay (§4) |
| **Residual shortcut** | gives deep layers an identity path that bypasses state relay, catching the depth gradients |

Measured (no residual, state bus only): for the depth $4 \to 24$ embedding gradient ratio to stay in $[0.1, 10]$,
$\rho_0 \ge 1.0$ is required; under-scaling ($\rho_0=0.3$) collapses to $1.1\text{e-}4$ —
`test_spectral_scale_decides_depth_stability` guards this path.

---

## 6. Output Mapping Head (LM Head)

$$
\mathrm{logits}_t = W_{\text{head}}\;\mathcal{N}\!\big(o_t\big), \qquad
W_{\text{head}} \in \mathbb{R}^{V \times d}
$$

$W_{\text{head}}$ is **weight-tied** to the embedding matrix, and the pretraining objective is standard autoregressive
next-token cross-entropy:

$$
\mathcal{L} = -\frac{1}{NT}\sum_{n=1}^{N}\sum_{t=1}^{T} \ln \frac{\exp(\mathrm{logits}_{n,t,y_{n,t}})}{\sum_{v}\exp(\mathrm{logits}_{n,t,v})}
$$

Continued SFT instruction tuning is supported (same head, same objective, with instruction-response pairs as data).

---

## 7. Input Layer

A BPE subword tokenizer implemented from scratch (`sunnynet/tokenizer.py`), without depending on `tokenizers` / `sentencepiece`.

- **Pre-tokenization**: four classes — CJK runs / Latin words / numbers / symbols — kept separate,
  so Chinese merges by phrase and English by word, while remaining compatible with code symbols;
- **Merging**: standard BPE with highest-frequency-first, heap-optimized incremental updates;
- **Vocabulary**: 4 special symbols + high-frequency characters + merge items.

Measured round-trip consistency: classical Chinese, English literature, Python code, and mixed Chinese-English **6/6 all lossless**.

> The input layer uses standard BPE tokenization and is not a core unit of the network.

---

## 8. Complexity and CPU Adaptation

Let $T$ be the sequence length, $d$ the backbone dimension, $r$ the rank, and $k$ the convolution taps.

| Path | Time complexity | Space complexity |
|---|---|---|
| State scan ($\star$) | $O(T r d)$ | $O(T r + \text{chunk}\cdot d)$ |
| Circulant filtering | $O(T d)$ | $O(1)$ |
| Depthwise separable convolution | $O(T d k)$ | $O(1)$ |
| Nonlinear branch | $O(T d d_i)$ | $O(T d_i)$ |
| **Total** | $\mathbf{O(T)}$ strictly linear | $O(T)$ (backward can drop to $O(\text{chunk})$) |

**There is no token-to-token pairwise interaction term**, so no $O(T^2)$ structure exists.

CPU adaptation points:

- Backbone matrix dimensions are small-to-medium (20M prototype: $d=512$, $r=320$, $d_i=896$, $L=10$, measured 20,955,698 parameters), so the working set of a single read/write fits in L2/L3;
- The low-rank factorization reduces the largest matrix from $2d \times 2d$ to $r \times 2d$ and $2d \times r$;
- Circulant filtering and depthwise separable convolution involve **zero matrix multiplications**;
- Native FP32, no dependence on FP16/BF16 (CPU support for half precision is weak, and software emulation is slower).

### 8.1 Measured Optimization Attempts (Including Rejected Items)

| Attempt | Outcome | Rationale |
|---|---|---|
| `logspace_scan` skipping `p.amin()` (passed in via `p_min_hint` instead) | **Adopted** | single step 0.596ms→0.043ms, values unchanged |
| Scan chunk length tuning (64→128/256/512/1024) | **No gain** | difference within ±10% noise on T=512…4096 |
| `gated_scan` avoiding `torch.cat(outs)` for a single chunk | **Rejected** | A/B gain ±5% with unstable direction, no real benefit |
| Rewriting competitive gating from roll to slicing | **Rejected** | measured slower (4.91ms vs 1.95ms); `roll` retained |
| `CenteredScaleNorm` fused with `var_mean` | **Rejected** | forward 46% faster but backward slower, and the numerical difference 4.8e-7 exceeds the 1e-10 invariant |
| State feedback `_feedback` rewritten to avoid `cat` | **Rejected** | not numerically equivalent (deviation 1.8) |
| `torch.compile` operator fusion | **Environment-limited** | MSVC lacks Windows SDK headers; Inductor's Windows branch requires an MSVC-style CLI and does not accept gcc |
| Hand-written AVX2 kernels (w64devkit GCC 16.2.0) | **Adopted** | scan at the operator level is 4.3–22× faster than torch; fused kernels give training steps of 1.10–1.17× (report §21/§22) |
| Multithreading (1→8 threads) | **Effective** | T=4096: 322ms→147ms (2.2×), concurrent operator scheduling parallelizes |

**Key diagnosis**: operator profiling shows the bottleneck is **scheduling overhead** rather than FLOPs — in a single
forward pass, `aten::mm` 31%, `aten::mul` 13%, `aten::add` 11%, `aten::var` 10%, plus **128 `cumsum` per pass** (from the
chunked scan). No single operator dominates, so single-point rewrites rarely help; the only effective path is reducing the
total operator count (fusion). `torch.compile` is unavailable on this machine, so fusion is achieved by hand-written AVX2
kernels that bypass Inductor.

**Measurement discipline (important)**: timing measurements must use **a separate process per cell**. The same model
measured in a process that "has already built several models" can take **more than 2× as long** as in a clean process
(T=4096: 325ms vs 776–822ms), and the amplification factor varies with architecture. Without isolation, all
cross-architecture timing ratios are distorted; see experiment report §2.4.

---

## 9. Comparison of Differences from Existing Paradigms

| Dimension | Transformer | RNN / LSTM / GRU | SSM / Mamba | **SunnyNet** |
|---|---|---|---|---|
| Sequence complexity | $O(T^2)$ | $O(T)$ | $O(T)$ | $O(T)$ |
| State update | stateless | $c_t = f\odot c_{t-1} + i\odot\tilde c$ (additive, unbounded) | $h_t = \bar A h_{t-1} + \bar B x_t$ (input-independent matrix) | $s_t = p_t\odot s_{t-1} + (1-p_t)\odot u_t$ (**convex combination, bounded**) |
| Exact recall capacity | $O(T)$ unbounded | unbounded | bounded | bounded, extensible on demand via the §1.6 associative memory |
| Source of retention rate | — | sigmoid gate (not a convex combination) | input-independent discretization step size | input self-gating, upper bound clamped to 1 |
| Time scales | none | single scale | mixed (HiPPO etc.) | log-uniform $[2, 8192]$, elementwise multi-scale |
| Inter-layer connection | residual addition | residual / stacking | residual | **state relay + feature residual stream** |
| Activation | GELU/SiLU, negative-side gradient $\to 0$ | tanh/sigmoid | SiLU | SoftWedge, derivative always $\in (1, 1.5]$ |
| Normalization | LayerNorm/RMSNorm | LayerNorm | RMSNorm | CenteredScaleNorm (smooth lower-bounded scale) |
| Initialization | Xavier/He | Xavier | specialized | spectral anchoring (direct control of the singular value spectrum) |

To be clear: **SunnyNet's state recurrence is mathematically related to linear-attention work**
(both are first-order linear recurrences), but the gating construction (convex combination + log-uniform multi-scale +
elementwise protection gate), the two-path inter-layer transfer of state relay and feature residual, and the activation
and normalization are all designed independently, and no code or structure from existing implementations is relied upon.

---

## 10. State Transfer Diagram

```
   input token stream  x_1  x_2  ...  x_T
                        │    │         │
   ┌────────────────────┴────┴─────────┴─────────────────────┐
   │  Block 1                                                │
   │   N(x) ─┐                                               │
   │         ├─ concat ─► low-rank proj ─► u_t ──┐           │
   │   circ(s)┘        │                         ▼           │
   │              └─► σ(g) ─────► p_t ●── convex recurrence  │
   │                                        ╲                │
   │  s_0 ──────────────────────────────────► S_t            │
   │                                 ┌────────┘              │
   │         W_c·A(W_z u) + β·S_t = o_t                      │
   └───────────────────────────┬─────────────────────────────┘
                               │
    end-of-block state S_T ───┼──► as next block's s_0  (state relay)
           o_1..o_T ──────────┘─► +x ─► next input      (feature residual)
                               │
   ┌───────────────────────────┴─────────────────────────────┐
   │  Block 2 … Block L   (same, ρ = 1.0, β = 0)             │
   └───────────────────────────┬─────────────────────────────┘
                               │
                     N(o) ─► W_head (tied to Embedding) ─► logits
```

Chunked parallel scan within a block:

```
    chunk 1    chunk 2    chunk 3      ← all chunks within a block run in parallel
   [parallel] [parallel] [parallel]
        │          │          │
        └──► back-solve chunk-initial state from S_T, back-fill per chunk (only T/chunk serial) ◄──┘
```

---

## 11. Known Limitations and Risks

1. **Long-range dependencies**: the information capacity of the rolling state is limited by the fixed dimension $d$,
   with a theoretical upper bound below full attention ($O(T)$ addressable). $H_{\max} = 8192$ only guarantees
   "not forgotten", not "retrievable". The associative memory branch of §1.6 raises exact recall capacity by about an
   order of magnitude (2 pairs 0.42→0.96), at a cost of about +22% training steps per layer (adjustable per layer,
   1.26–1.93×); whether to enable it by default depends on the task.
2. **Inter-layer input dilution (deep models only)**: the residual stream accumulates, layer by layer, a common component
   shared across inputs, and on deep models it grows monotonically.
   Shallow models do not have this problem — at $L$=4, the pairwise cosine of mean-centered states is ≈0
   (`_state_collapse.py` supplemental measurement gives −0.05), and the input-specific energy fraction **rises** with
   depth (0.29→0.60); the earlier "last-layer cosine 0.50 = collapse" was an artifact of the common component and has
   been corrected. On the deep model (`experiments/ckpts_100m/sunnynet_p100mw.pt`, $L$=8, $d$=1408), the
   common-component RMS is +42%, the input-specific RMS is −47%, the specific energy fraction goes 0.226→0.035, and
   outputs converge (16 inputs give only 6 distinct argmax).
   The source is the DC carried by the state ($\|\mathrm{mean}(s)\|/\mathrm{RMS}$ 0.57–0.80) injected into the residual
   stream through the feedback branch. `state_norm` (normalizing the state and the feedback source) halves the dilution
   without hurting ppl; **the residual stream itself cannot be cleaned** (`h_norm` measured ppl doubles 263→625, and
   24 inputs collapse to a single argmax). See report §26.3.
3. **Pseudo-convergence**: falling loss/ppl does not imply reasoning and instruction-following ability.
   Evaluation must include an independent test set and per-domain perplexity, not just training loss.
   The pairwise cosine of the final logits in the converged state reaches 0.98, but about 97% of it comes from an
   input-independent common component (after mean-centering the cosine is ≈0, and 32 inputs give 27 distinct argmax)
   — this metric cannot be read on its own as "output collapse"; it must be compared against a control model of the same scale.
4. **Magnitude of benefit**: the CPU training speedup must be measured; paper-style claims do not hold. See the measured data in the experiment report.