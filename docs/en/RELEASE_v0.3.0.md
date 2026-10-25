# v0.3.0

First tagged release of SunnyNet. It collects everything since the PyPI 0.2.0
upload: the depth-scaling cost law, the inter-layer dilution localization, three
whole-model C kernels for foreign architectures, the fp16 weight decode path, and
a size-aware decode thread policy.

## New: whole-model decode in C is the fastest path tested

The decode scorecard compared SunnyNet through the torch `forward_step` path and
reported it 1.9–2.6× slower than RWKV. The shipped whole-model C runtime was not in
that table. Adding it (`experiments/_decode_seven.py`, `sunnyC` config, same spec
d80/i160/r244/L4, single thread, B=1):

| decode ms/step | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| C runtime | **1.560** | **1.630** | **1.547** | **1.518** | **1.647** |
| RWKV | 1.779 | 1.879 | 1.673 | 1.675 | 1.674 |
| torch forward_step | 3.778 | 3.973 | 3.166 | 3.624 | 2.889 |
| Transformer | 2.473 | 3.405 | 5.236 | 4.056 | 6.962 |

The C runtime is first of eight at every length, 1.02–1.15× faster than RWKV, and
4.2× faster than a parameter-matched Transformer at T=4096. Numerical agreement with
the torch step is max|diff| 4.5e-07, argmax 48/48.

Decode parallelism now scales with weight size. Small models keep the weights in
cache, so extra threads only add synchronization: ≤23 MB blobs are fastest single
threaded, ≥39 MB blobs are fastest on physical cores (136 MB gains 1.29×). The CLI
picks this automatically from the blob size; `--threads` still overrides.

## Depth scaling

Training-step cost is linear in depth: `step_ms ≈ 32.9·L + 102.4` at d=128, T=256,
B=8, 6 threads. Each layer adds a fixed ~1100 aten calls (mm is 1.9% of them; 95.7%
are elementwise and view ops). At a fixed parameter budget, depth is worse on both
axes: `L4 wide` (d121) reaches zh ppl 247 at 900 steps while `L6`/`L8` land at
313–319 and take longer per step. New preset `p100msw` (d1536/i2624/r1152/L6,
stack 94,470,942, +0.68% of `p100mw`) is 4.2% faster than `p100mw` in an isolated A/B.

## Inter-layer dilution

Branch-level attribution on a 100M checkpoint attributes ~96% of the accumulating
cross-input common component to the `W_c(y)` branch — a slow, input-independent drift
fed back through the residual stream. The prior "0.50 state collapse" reading was a
common-component artifact; centered cosine is ≈0. `state_norm` now also normalizes
the feedback source, which raises 24-input argmax variety from 12 to 16 and lowers
val ppl 200.51 → 198.33 at 6000 steps on `deepl8`. Residual-stream normalization is
falsified (ppl 263 → 625).

## Whole-model C kernels for foreign architectures

DeltaNet, Mamba-2, and GLA now run their full forward pass in C, driven by
`--hf-arch {deltanet|mamba2|gla} --blob`. Per-token logits: DeltaNet 1.335e-05,
Mamba-2 1.678e-04, GLA 1.431e-05, argmax 36/36 each. Speedups are 1.10× / 1.37× /
1.01× — the gain tracks weight bytes over available bandwidth, and a 1.37 GB weight
is already at the single-thread sequential-read ceiling (18.63 GB/s).

## Also

- fp16 weight decode path (`SnModel(..., half=True)`, `--backend c --half`): reads
  467.6 → 222.3 MB per step, 1.33× faster than fp32, argmax 64/64.
- Matrix-valued rank-1 recurrence kernel `sn_mat_scan`, covering RWKV-7 and DeltaNet
  state forms, with an 11-case numerical contract.
- Prebuilt libraries are validated against a per-library source hash, not mtime, so
  a source edit no longer silently loads a stale binary.
- Two Windows build defects fixed: shared `_build` DLL locking, and GBK-decoded gcc
  diagnostics crashing on non-ASCII paths.

## Install

```
pip install sunnynet-cx==0.3.0
```

## Numbers and boundaries

Every figure above is measured on one machine (i7-8750H, AVX2, 6 physical cores,
15.87 GB RAM, torch 2.14.1+cpu) and reproducible from `experiments/`. Decode results
are B=1, single sequence, randomly initialized models, torch for the peer
architectures. Full detail: `docs/en/EXPERIMENT_REPORT.md`. Test suite: 164 passed,
36 skipped (skips need external weights).