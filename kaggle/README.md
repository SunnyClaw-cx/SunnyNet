# SunnyNet training kernels (Kaggle)

One-file, self-contained training kernels that run on Kaggle's free-tier GPU
(no prebuilt C kernel, no dataset upload, no API credentials in-repo). Each
clones this repo, applies the `--save-every` milestone patch to `train/train.py`,
builds its own corpus, smoke-tests, trains, and writes milestone samples to
`/kaggle/output` (pullable with `kaggle kernels pull`).

## The four kernels

| Kernel | `--preset` | Data | Goal |
|--------|-----------|------|------|
| `sunnynet_p100.py` | `p100m` (94M) | ~200 diverse EN books + code | 94M English prose |
| `code_mix/sunnynet_code_mix.py` | `p100m` (94M) | 70% EN prose + 30% Python | stage-2 code mix |
| `big_data/sunnynet_bigdata.py` | `proto20m` (21M) | ~1,000 diverse EN books, seq 1024 | **stage-1 coherent prose** (high tokens/param) |
| `tiny_stories/sunnynet_tinystories.py` | `proto20m` (21M) | TinyStories (roneneldan/TinyStories, HF), seq 512, ~150M tokens | **coherent short stories** (~9 tok/param on simple data) |

## Run one

```sh
pip install kaggle  # then: kaggle login
kaggle kernels push -p kaggle/big_data -t 32400   # -> kitastro/... big-data run
kaggle kernels push -p kaggle/tiny_stories -t 32400   # -> kitastro/... TinyStories run
# point a browser at the slug it prints; ~7h; results under the kernel's "Output" tab
kaggle kernels pull <owner>/<slug>   # grabs /kaggle/output (results.json + ckpts)
```

Set `REPO_URL` at the top of each kernel to your own fork if you want it to
clone a branch other than `SunnyClaw-cx/SunnyNet:main`.

## Design notes

- **Self-contained.** The gutendex book fetcher (`fetch_books.py`) and the code
  source fetcher (`fetch_code.py`) are embedded as base64 blobs and decoded at
  runtime, so the kernel stays a single paste-able file and doesn't depend on
  the upstream repo having those helpers.
- **Corpus via gutendex, not the built-in 35 classics.** The most-read
  public-domain English books (ranked by `download_count`) give a *diverse*
  corpus — the fix for the "5.5 epochs of 35 novels" overfitting that produced
  fluent-but-incoherent prose in the early p100m runs.
- **Epoch-based step budget.** `estimate_steps()` reads the built corpus's token
  count and targets ~`EPOCHS` passes (capped to the 9 h P100 budget), so the run
  lands at a meaningful token/param ratio instead of a fixed arbitrary step count.
- **VRAM probe.** `probe_batch()` starts from the largest batch and steps down
  on OOM, so the same kernel works on T4 / P100 / V100.
- **TinyStories inlines its own streaming encoder.** The upstream
  `build_corpus.encode_domain` builds a whole-domain Python `int` list (peak
  memory ∝ total tokens), which OOMs at TinyStories' ~1.16B tokens. The kernel
  writes its per-file uint16 encoder directly and targets a **token budget**
  (~150M, ≈9 tok/non-embed-param) rather than whole epochs, staying inside the
  9 h P100 cap. The same streaming fix is in the repo's `build_corpus.py` (this
  PR) for local use.

## Reading the outputs

Each `results.json` has `samples` (first / middle / final checkpoint × 3 prompts)
and `val_history`. The samples are the honest quality signal — val loss
understates overfitting when the val set overlaps the training corpus (see
`docs/en/MODEL_SHOWCASE.md` §1, the footnote on non-comparable val loss).

See `docs/en/MODEL_SHOWCASE.md` for the per-model results from the 2026-10-24
session these kernels were built for.
