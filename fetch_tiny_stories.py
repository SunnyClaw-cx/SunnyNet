"""Fetch the TinyStories corpus (roneneldan/TinyStories on HuggingFace).

Streams TinyStories-train.txt (~1.9 GB) in, splits it on story boundaries into
~N chunk files under data/raw/tiny_stories/, so the rest of the pipeline treats
it like any other domain:

    1. python fetch_tiny_stories.py          # download + chunk (idempotent)
    2. encode with the BPE (existing proc tokenizer, or a fresh one)
    3. python train/train.py --preset proto20m --mix tiny_stories=1.0 ...

Story format: each story is a short multi-line block; blocks are separated by
blank lines, so chunking happens on blank-line boundaries and no story is cut.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "tiny_stories"
TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"

CHUNK_MB = 4  # ~4 MB per chunk file -> ~500 files for the 1.9 GB train split
EXPECT_SIZE = 1_924_281_556  # TinyStories-train.txt content length (bytes)


def stream_download(url: str, dst: Path) -> int:
    """Stream url to dst, resuming from existing partial size via Range header."""
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
    """Split src into ~target_mb files on blank-line (story) boundaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # wipe previous chunks (idempotent re-runs)
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
            if line.strip() == "":  # blank line ends a story
                block = "".join(story)
                story = []
                buf.append(block)
                size += len(block)
                if size >= target:
                    _write(out_dir, n, buf)
                    n += 1
                    buf, size = [], 0
        if story:  # trailing story without final blank line
            buf.append("".join(story))
        if buf:
            _write(out_dir, n, buf)
    return n + 1


def _write(out_dir: Path, i: int, blocks: list[str]) -> None:
    (out_dir / f"ts_{i:04d}.txt").write_text("".join(blocks), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-mb", type=float, default=CHUNK_MB)
    ap.add_argument(
        "--cache", type=Path, default=ROOT / "data" / "proc" / "tiny_stories_train.txt"
    )
    ap.add_argument(
        "--force", action="store_true", help="re-download even if chunks exist"
    )
    args = ap.parse_args()

    # data/proc/ is gitignored and absent on a fresh clone; create it so the
    # download target has a parent dir.
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    done = sorted(RAW_DIR.glob("ts_*.txt"))
    if done and not args.force:
        print(
            f"  cached: {len(done)} chunks, {sum(p.stat().st_size for p in done) >> 20} MB -> {RAW_DIR}"
        )
        return 0

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
                print(
                    f"  [warn] interrupted ({type(e).__name__}); resuming next attempt"
                )
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
    print(
        "  next: encode + train, e.g.\n"
        "  python train/train.py --arch sunnynet --preset proto20m --mix tiny_stories=1.0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
