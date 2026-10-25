"""SunnyNet on Kaggle: TinyStories corpus run (proto20m, own-BPE, ~1 epoch).

TinyStories (roneneldan/TinyStories, HF) is the classic "small model + simple
diverse corpus -> actually coherent" benchmark: ~50k very short self-contained
stories, ~100M+ tokens. Unlike the public-domain book runs (which sat at ~1
token/param and only produced fluent nonsense), this targets ~10-15 tokens/
param on a 21M model, which is the "coherent-but-not-smart" band.

Self-contained: clones the repo, fetches the corpus, trains its own BPE on
the stories (a BPE fitted to simple prose is a better tokenizer for them than
the mixed zh/en/code one), encodes a tiny_stories domain, smoke-tests, trains
with milestone checkpoints, and writes samples to /kaggle/output.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

# ============================== CONFIG ======================================
REPO_URL = "https://github.com/SunnyClaw-cx/SunnyNet"  # edit to your fork if needed
REPO_DIR = "/kaggle/working/SunnyNet"

PRESET = (
    "proto20m"  # ~21M params (16.8M non-embed): small model over a simple big corpus
)
# ~150M training tokens ~= ~9 tokens/non-embed-param: deep past the coherence
# threshold on this simple corpus, and safely inside the 9h P100 budget
# (fetch+encode overhead ~2.5h + ~5.5h training at ~0.12-0.15 ms/token).
TARGET_TRAIN_TOKENS = 150_000_000
BATCH_CANDIDATES = (32, 16, 8, 4, 2)  # probe from largest down until it fits
SEQ_LEN = 512  # a TinyStories story is ~150-300 tokens; 512 holds 2-4 whole stories
MIX = "tiny_stories=1.0"
SAVE_EVERY = 2_000  # milestone checkpoints every N steps
VAL_EVERY = 500
NO_EARLY_STOP = True
CKPT_STEM = "tst"  # runs/proto20m_tst.pt/.json + milestone globs

VOCAB = 8192
PRESET_DIMS = {  # mirrors _preset_cfg in train/train.py
    "proto20m": {"dim": 512, "inner": 896, "rank": 320, "layers": 10, "h_max": 8192.0},
}
# TinyStories corpus: full train split (~1.9 GB, ~100M+ tokens) from HuggingFace.
TINY_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
TINY_EXPECT = 1_924_281_556  # content length of TinyStories-train.txt

SAMPLE_PROMPTS = [
    "Once upon a time, ",
    "One day, ",
    "A little girl named ",
    "The end",
]
# Where results land: /kaggle/output on Kaggle (pullable via `kaggle kernels pull`),
# a local dir when run on your machine.
OUT_DIR = Path("/kaggle/output") if Path("/kaggle").exists() else Path("kaggle_out")
# ==============================================================================

# Self-contained TinyStories fetcher + chunker (decoded into the cloned repo at
# runtime so the kernel stays a single paste-able file).
FETCH_TINY_B64 = """
import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "tiny_stories"
TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
CHUNK_MB = 4
EXPECT_SIZE = 1_924_281_556


def stream_download(url: str, dst: Path) -> int:
    off = dst.stat().st_size if dst.exists() else 0
    headers = {"User-Agent": "sunnynet-corpus"}
    if off:
        headers["Range"] = f"bytes={off}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        got = off
        with dst.open("ab") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                got += len(b)
                if got % (128 << 20) < 1 << 20:
                    print(f"  downloaded {got >> 20} MB (total)", flush=True)
    return got


def chunk_stories(src: Path, out_dir: Path, target_mb: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("ts_*.txt"):
        p.unlink()
    n = 0
    buf: list[str] = []
    size = 0
    target = int(target_mb * 1024 * 1024)
    with src.open("r", encoding="utf-8", errors="replace") as f:
        story: list[str] = []
        for line in f:
            story.append(line)
            if line.strip() == "":
                block = "".join(story)
                story = []
                buf.append(block)
                size += len(block)
                if size >= target:
                    (out_dir / f"ts_{n:04d}.txt").write_text("".join(buf), encoding="utf-8")
                    n += 1
                    buf, size = [], 0
        if story:
            buf.append("".join(story))
        if buf:
            (out_dir / f"ts_{n:04d}.txt").write_text("".join(buf), encoding="utf-8")
    return n + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-mb", type=float, default=CHUNK_MB)
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "proc" / "tiny_stories_train.txt")
    args = ap.parse_args()
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    got = args.cache.stat().st_size if args.cache.exists() else 0
    if got >= EXPECT_SIZE:
        print(f"  using cached raw download {args.cache} ({got >> 20} MB)")
    else:
        print(f"[1/2] downloading {TRAIN_URL} (resume from {got >> 20} MB)")
        for attempt in range(1, 15):
            print(f"  attempt {attempt} ...")
            try:
                stream_download(TRAIN_URL, args.cache)
            except OSError as e:  # URLError / reset / timeout all resume
                print(f"  [warn] interrupted ({type(e).__name__}); resuming next attempt")
            if args.cache.exists() and args.cache.stat().st_size >= EXPECT_SIZE:
                break
        got = args.cache.stat().st_size if args.cache.exists() else 0
        if got < 1_000_000:
            sys.exit(f"  download too small ({got} B) - network blocked?")
        print(f"  raw file {got >> 20} MB / {EXPECT_SIZE >> 20} MB")

    print(f"[2/2] chunking into {RAW_DIR} (~{args.chunk_mb} MB files)...")
    n = chunk_stories(args.cache, RAW_DIR, args.chunk_mb)
    total_mb = sum(p.stat().st_size for p in RAW_DIR.glob("ts_*.txt")) >> 20
    print(f"  done: {n} files, {total_mb} MB in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".strip()


def sh(cmd: list[str], cwd: str | None = None, timeout: int = 3600) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(
        cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False
    )
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr[-4000:]}")


def run_capture(
    cmd: list[str], cwd: str | None = None, timeout: int = 3600
) -> tuple[int, str, str]:
    r = subprocess.run(
        cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False
    )
    return r.returncode, r.stdout, r.stderr


def write_fetch_tiny() -> None:
    (Path(REPO_DIR) / "fetch_tiny_stories.py").write_text(
        FETCH_TINY_B64, encoding="utf-8"
    )
    print(f"  wrote fetch_tiny_stories.py ({len(FETCH_TINY_B64)} bytes)")


def build_corpus_tiny() -> int:
    """Own BPE (trained on the stories) + tiny_stories domain encode.

    Returns total train tokens (for step estimation)."""
    script = textwrap.dedent("""
        import json, sys
        from pathlib import Path
        ROOT = Path(".").resolve()
        sys.path.insert(0, str(ROOT))
        import build_corpus
        from sunnynet.tokenizer import BPETokenizer

        DOM = "tiny_stories"
        V = 8192
        texts = build_corpus.read_texts(DOM)
        if not texts:
            sys.exit("no tiny_stories texts under data/raw/tiny_stories")
        flat = "".join(t for _, t in texts)
        print(f"corpus: {len(texts)} files, {len(flat):,} chars")

        print("training BPE on the stories (own tokenizer)...")
        sample = [t for _, t in texts[:2000]]  # BPE only needs a representative slice
        tk = BPETokenizer.train(sample, vocab_size=V, min_char_freq=3, verbose=True)
        tk.save(ROOT / "data" / "proc" / "tokenizer.json")

        import numpy as np
        # Streaming per-file uint16 encoder (NOT build_corpus.encode_domain, which
        # on main still builds a whole-domain Python int list -> OOM at 1.16B tokens).
        step = max(1, int(round(1.0 / 0.05)))
        tr_parts, va_parts = [], []
        for fi, (_, t) in enumerate(texts):
            ids = np.asarray(tk.encode(t) + [tk.eos_id], dtype=np.uint16)
            (va_parts if fi % step == 0 else tr_parts).append(ids)
        tr = np.concatenate(tr_parts)
        va = np.concatenate(va_parts)
        proc = ROOT / "data" / "proc"
        tr.tofile(proc / f"tokens_{DOM}.bin")
        va.tofile(proc / f"val_{DOM}.bin")
        r = {"train_tokens": int(tr.size), "val_tokens": int(va.size),
             "train_files": len(tr_parts), "val_files": len(va_parts)}
        print(f"tiny_stories train/val tokens = {r['train_tokens']:,} / {r['val_tokens']:,}")
        # keep a record of the val split for the run log
        man = ROOT / "data" / "proc" / "manifest_tiny.json"
        man.write_text(json.dumps({"domains": {DOM: r}, "vocab_size": V}, indent=2))
        print(r["train_tokens"])
    """)
    rc, out, err = run_capture(
        [sys.executable, "-c", script], cwd=REPO_DIR, timeout=5400
    )
    if rc != 0:
        sys.exit(f"corpus build failed:\n{out[-2000:]}\n{err[-2000:]}")
    return int(out.strip().splitlines()[-1])


def estimate_steps(train_tokens: int, bs: int) -> int:
    """~TARGET_TRAIN_TOKENS training tokens (a multiple of epochs), inside the GPU budget."""
    per_step = bs * SEQ_LEN
    want = int(TARGET_TRAIN_TOKENS / max(1, per_step))
    eff = want * per_step / max(1, train_tokens)
    print(
        f"  corpus={train_tokens:,} tokens; targeting {TARGET_TRAIN_TOKENS:,} train tokens "
        f"= {want} steps (~{eff:.2f} epochs over corpus)"
    )
    return want


def main() -> None:
    # ---- 1. clone ------------------------------------------------------------
    print("== [1/6] clone ==")
    if not Path(REPO_DIR).exists():
        sh(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])

    # ---- 2. corpus: TinyStories train split ---------------------------------
    print("== [2/6] corpus ==")
    write_fetch_tiny()
    sh([sys.executable, "fetch_tiny_stories.py"], cwd=REPO_DIR, timeout=5400)

    # ---- 3. own BPE + encode tiny_stories domain ---------------------------
    # (train.py on main already supports --save-every, so no patching needed)
    print("== [3/6] encode ==")
    train_tokens = build_corpus_tiny()

    # ---- 4. smoke test: 60 steps on the real model, catch OOM/NaN early ----
    print("== [4/6] smoke ==")
    sh(
        [
            sys.executable,
            "train/train.py",
            "--arch",
            "sunnynet",
            "--preset",
            PRESET,
            "--device",
            "cuda",
            "--threads",
            "1",
            "--steps",
            "60",
            "--batch-size",
            str(BATCH_CANDIDATES[-1]),
            "--seq-len",
            str(SEQ_LEN),
            "--mix",
            MIX,
            "--val-every",
            "60",
        ],
        cwd=REPO_DIR,
    )

    # ---- 5. probe batch, estimate steps, then train --------------------------
    print("== [5/6] train ==")
    bs = probe_batch()
    print(f"  batch={bs} selected")
    steps = estimate_steps(train_tokens, bs)
    cmd = [
        sys.executable,
        "train/train.py",
        "--arch",
        "sunnynet",
        "--preset",
        PRESET,
        "--device",
        "cuda",
        "--threads",
        "1",
        "--steps",
        str(steps),
        "--batch-size",
        str(bs),
        "--seq-len",
        str(SEQ_LEN),
        "--mix",
        MIX,
        "--val-every",
        str(VAL_EVERY),
        "--save-every",
        str(SAVE_EVERY),
        "--save-ckpt",
        f"runs/{PRESET}_{CKPT_STEM}.pt",
    ]
    if NO_EARLY_STOP:
        cmd.append("--no-early-stop")
    sh(cmd, cwd=REPO_DIR, timeout=32400)

    # ---- 6. sample milestones: watch the text come together -----------------
    print("== [6/6] milestone samples ==")
    sample_milestones()
    write_results()


def probe_batch() -> int:
    """Largest batch whose forward+backward fits in VRAM (O(N) model, no KV cache)."""
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, '.')
        import torch
        from sunnynet.model import SunnyConfig, SunnyNet
        _d = {PRESET_DIMS[PRESET]}
        c = SunnyConfig(vocab_size={VOCAB}, **_d)
        m = SunnyNet(c).cuda(); opt = torch.optim.AdamW(m.parameters())
        for bs in {BATCH_CANDIDATES}:
            x = torch.randint(0, {VOCAB}, (bs, {SEQ_LEN}), device='cuda')
            y = x.roll(-1, 1)
            try:
                opt.zero_grad()
                torch.nn.functional.cross_entropy(m(x).reshape(-1, {VOCAB}), y.reshape(-1)).backward()
                torch.cuda.synchronize()
                print(f"bs={{bs}} OK: {{torch.cuda.max_memory_reserved()/1e6:.0f}}MB reserved")
                print("PICKED " + str(bs)); sys.exit(0)
            except torch.cuda.OutOfMemoryError:
                print(f"bs={{bs}} OOM"); torch.cuda.empty_cache()
        sys.exit(1)
    """)
    r = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    for ln in r.stdout.splitlines():
        if ln.startswith("PICKED "):
            return int(ln.split()[1])
    sys.exit(f"batch probe failed:\n{r.stdout}\n{r.stderr}")


def sample_milestones() -> None:
    """Generate from each milestone checkpoint; store samples for write_results()."""
    cks = [
        p
        for p in sorted(Path(REPO_DIR, "runs").glob(f"{PRESET}_{CKPT_STEM}*.pt"))
        if p.exists()
    ]
    sample_milestones.samples = {}
    sample_milestones.keep = []
    if not cks:
        print("no checkpoints found - training failed, nothing to sample")
        return
    if len(cks) >= 3:
        picks = {cks[0], cks[len(cks) // 2], cks[-1]}
    else:
        picks = set(cks)
    sample_milestones.keep = sorted(picks)
    for ck in sorted(picks):
        print(f"\n### {ck.name}")
        outs = {}
        for p in SAMPLE_PROMPTS:
            _, out, err = run_capture(
                [
                    sys.executable,
                    "-m",
                    "sunnynet.cli",
                    "--ckpt",
                    str(ck),
                    "--prompt",
                    p,
                    "--max-new",
                    "80",
                    "--temperature",
                    "0.7",
                ],
                cwd=REPO_DIR,
            )
            outs[p] = out[-1200:] if out.strip() else f"(empty) {err.strip()[-200:]}"
        sample_milestones.samples[ck.name] = outs


def write_results() -> None:
    """Copy sampled checkpoints + samples + val history to OUT_DIR (pullable)."""
    import shutil

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ck in list(getattr(sample_milestones, "keep", [])) + [
        Path(REPO_DIR, f"runs/{PRESET}_{CKPT_STEM}.pt")
    ]:
        if ck.exists():
            shutil.copy(ck, OUT_DIR / ck.name)
    res = {"samples": getattr(sample_milestones, "samples", {}), "val_history": []}
    log = Path(REPO_DIR, f"runs/{PRESET}_{CKPT_STEM}.json")
    if log.exists():
        hist = json.loads(log.read_text(encoding="utf-8")).get("history", [])
        step = max(1, len(hist) // 60)
        res["val_history"] = hist[::step]
    (OUT_DIR / "results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nresults -> {OUT_DIR} (results.json + checkpoint copies)")


if __name__ == "__main__":
    main()
