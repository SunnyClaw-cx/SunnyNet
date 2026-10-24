"""SunnyNet on Kaggle: stage-1 big-data run (proto20m + ~1000 diverse EN books).

v4 design: the model was fine; the *data* was the bottleneck. v3 trained 5.5
epochs over 35 classics (word-soup, low val-ppl that measured fit-to-35-books).
v5 flips the ratio: smaller model + much more data. proto20m (21M total) over ~1000 diversepublic-domain books (~300M+ unique tokens) ~= 10+ tokens/param - the regime where small-modelprose stops being soup. Long context (seq 1024) is nearly free thanks to O(N). (~40-60M unique
tokens) and trains ~1.5 epochs, so it learns *language* instead of memorising a
handful of novels.

Usage:
  1. Kaggle -> New Notebook, accelerator = GPU P100 16GB (free tier 30h/week).
  2. Paste this file (self-contained: the gutendex fetcher is embedded base64).
  3. Results (milestone samples + val history + ckpt copies) -> /kaggle/output.
"""

from __future__ import annotations

import base64
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

# ============================== CONFIG ======================================
REPO_URL = "https://github.com/SunnyClaw-cx/SunnyNet"  # edit to your fork if needed
REPO_DIR = "/kaggle/working/SunnyNet"

PRESET = "proto20m"  # ~21M total params - small model + big data = high tokens/param
EPOCHS = 1.5  # train ~1.5 epochs over the *diverse* corpus
MAX_STEPS = 16_000  # hard cap: bs16*seq1024 at ~0.1ms/tok ~= 7h + overhead, inside the 9h cap
BATCH_CANDIDATES = (16, 8, 4, 2)  # proto20m is light - try bs=16 first  # probe from largest down until forward+backward fits
SEQ_LEN = 1024  # O(N) kernel: long context is nearly free; coheres samples
MIX = "en_literature=1.0"  # stage 1 = coherent English prose
SAVE_EVERY = 2_000  # milestone checkpoints every N steps
VAL_EVERY = 500  # ~54 val checks at 27k steps
NO_EARLY_STOP = True
CKPT_STEM = "p20b"  # runs/proto20m_p20b.pt/.json + milestone globs

VOCAB = 8192
PRESET_DIMS = {  # mirrors _preset_cfg in train/train.py (p100m/p100mw have no static cfg method)
    "p100m": {"dim": 1024, "inner": 1792, "rank": 640, "layers": 14, "h_max": 16384.0},
    "p100mw": {"dim": 1408, "inner": 1920, "rank": 1024, "layers": 8, "h_max": 16384.0},
    "proto20m": {"dim": 512, "inner": 896, "rank": 320, "layers": 10, "h_max": 8192.0},
}
# Diverse EN corpus: most-downloaded public-domain books from gutendex.
GUTENDEX_PAGES = 35  # server caps ~32/page -> ~1100 candidates
GUTENDEX_TOP = 1000  # fetch the most-downloaded N of the candidates
MIN_BOOKS = 300  # fall back to curated classics if the fetch is this thin
# Fallback curated classics (verified via Gutenberg search; used only if gutendex is thin).
EXTRA_BOOKS = [
    15,
    2852,
    7700,
    19640,
    7100,
    19797,
    1260,
    1400,
    18581,
    35,
    1399,
    829,
    521,
    33,
    599,
    1184,
    135,
    3707,
    4300,
    2781,
    768,
    145,
]

SAMPLE_PROMPTS = [
    "It was the best of times, ",
    "Once upon a time, ",
    "The morning sun was ",
]
# Where results land: /kaggle/output on Kaggle (pullable via `kaggle kernels pull`),
# a local dir when run on your machine.
OUT_DIR = Path("/kaggle/output") if Path("/kaggle").exists() else Path("kaggle_out")
# ==============================================================================

# Self-contained gutendex fetcher (repo-root fetch_books.py). Decoded into the
# cloned repo at runtime so the kernel stays a single paste-able file.
FETCH_BOOKS_B64 = "IiIiRGl2ZXJzZSBwdWJsaWMtZG9tYWluIEVuZ2xpc2ggYm9vayBjb3JwdXMgdmlhIHRoZSBndXRlbmRleCBBUEkuCgpTdXBwbGVtZW50cyBwcmVwYXJlX2RhdGEncyBmaXhlZCAxMyBjbGFzc2ljcyB3aXRoIHRoZSBtb3N0LXJlYWQgRW5nbGlzaCBib29rcwpvbiBQcm9qZWN0IEd1dGVuYmVyZyAocmFua2VkIGJ5IGRvd25sb2FkX2NvdW50KSwgc28gdGhlIEVOIHRyYWluaW5nIGNvcnB1cyBpcwpsYXJnZSBhbmQgKmRpdmVyc2UqIGluc3RlYWQgb2YgYSBoYW5kZnVsIG9mIHJlcGVhdGVkIG5vdmVscy4gUmV1c2VzCnByZXBhcmVfZGF0YSdzIGZldGNoL2RlY29kZS9jbGVhbi9zYXZlIHNvIG91dHB1dCBpcyBieXRlLWNvbXBhdGlibGUuCiIiIgoKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBqc29uCmltcG9ydCB1cmxsaWIuZXJyb3IKaW1wb3J0IHVybGxpYi5yZXF1ZXN0CmZyb20gY29uY3VycmVudC5mdXR1cmVzIGltcG9ydCBUaHJlYWRQb29sRXhlY3V0b3IsIGFzX2NvbXBsZXRlZAoKaW1wb3J0IHByZXBhcmVfZGF0YSBhcyBwZAoKVUEgPSB7IlVzZXItQWdlbnQiOiAiTW96aWxsYS81LjAgKHN1bm55bmV0LWNvcnB1cykifQpHVVRFTkRFWCA9ICJodHRwczovL2d1dGVuZGV4LmNvbS9ib29rcy8/bGFuZ3VhZ2VzPWVuJnBlcl9wYWdlPXtwZXJfcGFnZX0mcGFnZT17cGFnZX0iCgoKZGVmIGd1dGVuZGV4X2Jvb2tzKHBlcl9wYWdlOiBpbnQgPSAxMDAsIHBhZ2VzOiBpbnQgPSA1KSAtPiBsaXN0W2RpY3RdOgogICAgb3V0OiBsaXN0W2RpY3RdID0gW10KICAgIGZvciBwIGluIHJhbmdlKDEsIHBhZ2VzICsgMSk6CiAgICAgICAgdXJsID0gR1VURU5ERVguZm9ybWF0KHBlcl9wYWdlPXBlcl9wYWdlLCBwYWdlPXApCiAgICAgICAgdHJ5OgogICAgICAgICAgICBkID0ganNvbi5sb2FkcygKICAgICAgICAgICAgICAgIHVybGxpYi5yZXF1ZXN0LnVybG9wZW4oCiAgICAgICAgICAgICAgICAgICAgdXJsbGliLnJlcXVlc3QuUmVxdWVzdCh1cmwsIGhlYWRlcnM9VUEpLCB0aW1lb3V0PTkwCiAgICAgICAgICAgICAgICApLnJlYWQoKQogICAgICAgICAgICApCiAgICAgICAgICAgIG91dCArPSBkLmdldCgicmVzdWx0cyIsIFtdKQogICAgICAgIGV4Y2VwdCAodXJsbGliLmVycm9yLlVSTEVycm9yLCBqc29uLkpTT05EZWNvZGVFcnJvciwgVGltZW91dEVycm9yKSBhcyBlOgogICAgICAgICAgICAjIGtlZXAgZ29pbmcgLSBvbmUgc2xvdyBwYWdlIHNob3VsZG4ndCBzaW5rIHRoZSBydW4KICAgICAgICAgICAgcHJpbnQoZiIgIFtwYWdlIHtwfV0gRVJSIHtlfSIpCiAgICByZXR1cm4gb3V0CgoKZGVmIGdyYWIoYjogZGljdCkgLT4gdHVwbGVbaW50LCBzdHJdIHwgTm9uZToKICAgIHVybCA9IGIuZ2V0KCJmb3JtYXRzIiwge30pLmdldCgidGV4dC9wbGFpbjsgY2hhcnNldD11dGYtOCIpCiAgICBpZiBub3QgdXJsOgogICAgICAgIHJldHVybiBOb25lCiAgICBkYXRhID0gcGQuZmV0Y2godXJsLCB0aW1lb3V0PTkwKQogICAgaWYgbm90IGRhdGEgb3IgbGVuKGRhdGEpIDwgMjAwMDoKICAgICAgICByZXR1cm4gTm9uZQogICAgdCA9IHBkLmRlY29kZV9hbnkoZGF0YSkKICAgIHJldHVybiAoYlsiaWQiXSwgdCkgaWYgdCBlbHNlIE5vbmUKCgpkZWYgbWFpbigpIC0+IGludDoKICAgIGFwID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAgYXAuYWRkX2FyZ3VtZW50KAogICAgICAgICItLXRvcCIsIHR5cGU9aW50LCBkZWZhdWx0PTMwMCwgaGVscD0i5Y+WIGRvd25sb2FkX2NvdW50IOacgOmrmOeahCBOIOacrCIKICAgICkKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1wYWdlcyIsIHR5cGU9aW50LCBkZWZhdWx0PTUsIGhlbHA9Imd1dGVuZGV4IOe/u+mhteaVsCAo5q+P6aG1IDEwMCkiKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLXdvcmtlcnMiLCB0eXBlPWludCwgZGVmYXVsdD04KQogICAgYXJncyA9IGFwLnBhcnNlX2FyZ3MoKQoKICAgIHByaW50KGYiW2d1dGVuZGV4XSDmipPlj5YgRU4g5Lmm55uuIChwYWdlcz17YXJncy5wYWdlc30pIC4uLiIpCiAgICBib29rcyA9IGd1dGVuZGV4X2Jvb2tzKHBhZ2VzPWFyZ3MucGFnZXMpCiAgICBib29rcyA9IFtiIGZvciBiIGluIGJvb2tzIGlmIG5vdCBiLmdldCgiY29weXJpZ2h0IildCiAgICBib29rcy5zb3J0KGtleT1sYW1iZGEgYjogYi5nZXQoImRvd25sb2FkX2NvdW50IiwgMCksIHJldmVyc2U9VHJ1ZSkKICAgIHRvcCA9IGJvb2tzWzogYXJncy50b3BdCiAgICBwcmludChmIiAg5YCZ6YCJIHtsZW4oYm9va3MpfSDmnKzlhazlvIDpoobln5/oi7HmlofkuaYsIOWPluS4i+i9vemHj+acgOmrmOeahCB7bGVuKHRvcCl9IOacrCIpCgogICAgc3RhdHMgPSB7InpoX2FuY2llbnQiOiAwLCAiZW5fbGl0ZXJhdHVyZSI6IDAsICJkcm9wIjogMH0KICAgIGNoYXJzID0geyJ6aF9hbmNpZW50IjogMCwgImVuX2xpdGVyYXR1cmUiOiAwfQogICAgd2l0aCBUaHJlYWRQb29sRXhlY3V0b3IoYXJncy53b3JrZXJzKSBhcyBleDoKICAgICAgICBmdXRzID0ge2V4LnN1Ym1pdChncmFiLCBiKTogYiBmb3IgYiBpbiB0b3B9CiAgICAgICAgZm9yIGksIGYgaW4gZW51bWVyYXRlKGFzX2NvbXBsZXRlZChmdXRzKSwgMSk6CiAgICAgICAgICAgIHIgPSBmLnJlc3VsdCgpCiAgICAgICAgICAgIGlmIHIgaXMgTm9uZToKICAgICAgICAgICAgICAgIHN0YXRzWyJkcm9wIl0gKz0gMQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgYmlkLCByYXcgPSByCiAgICAgICAgICAgICAgICBib2R5ID0gcGQuc3RyaXBfYm9pbGVycGxhdGUocmF3KQogICAgICAgICAgICAgICAgZG9tID0gcGQuY2xhc3NpZnkoYm9keSkKICAgICAgICAgICAgICAgIGlmIGRvbSA9PSAiZHJvcCI6CiAgICAgICAgICAgICAgICAgICAgc3RhdHNbImRyb3AiXSArPSAxCiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgICAgIGJvZHkgPSBwZC5jbGVhbl96aChib2R5KSBpZiBkb20gPT0gInpoX2FuY2llbnQiIGVsc2UgcGQuY2xlYW5fZW4oYm9keSkKICAgICAgICAgICAgICAgIGlmIGxlbihib2R5KSA8IDgwMDA6CiAgICAgICAgICAgICAgICAgICAgc3RhdHNbImRyb3AiXSArPSAxCiAgICAgICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgICAgIGNoYXJzW2RvbV0gKz0gcGQuc2F2ZShkb20sIGYiZ3V0ZW5kZXhfe2JpZH0udHh0IiwgYm9keSkKICAgICAgICAgICAgICAgIHN0YXRzW2RvbV0gKz0gMQogICAgICAgICAgICBpZiBpICUgNTAgPT0gMDoKICAgICAgICAgICAgICAgIHByaW50KAogICAgICAgICAgICAgICAgICAgIGYiICAuLi57aX0ve2xlbih0b3ApfSAgZW49e3N0YXRzWydlbl9saXRlcmF0dXJlJ119IGRyb3A9e3N0YXRzWydkcm9wJ119IgogICAgICAgICAgICAgICAgKQoKICAgIHByaW50KCIgIOWIhuWfn+e7k+aenDoiLCBzdGF0cykKICAgIHByaW50KCIgIOWtl+espuaVsDoiLCB7azogZiJ7djosfSIgZm9yIGssIHYgaW4gY2hhcnMuaXRlbXMoKX0pCiAgICByZXR1cm4gMAoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICByYWlzZSBTeXN0ZW1FeGl0KG1haW4oKSkK"


def sh(cmd: list[str], cwd: str | None = None, timeout: int = 3600) -> None:
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if r.stdout.strip():
        print(r.stdout[-4000:])
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr[-4000:]}")


def run_capture(
    cmd: list[str], cwd: str | None = None, timeout: int = 3600
) -> tuple[int, str, str]:
    """Like sh() but never exits - fetch/sampling must not kill the run."""
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    print(r.stdout[-3000:])
    return r.returncode, r.stdout, r.stderr


def write_fetch_books() -> None:
    code = base64.b64decode(FETCH_BOOKS_B64)
    (Path(REPO_DIR) / "fetch_books.py").write_bytes(code)
    print(f"  wrote fetch_books.py ({len(code)} bytes)")


def estimate_steps(bs: int) -> int:
    """~EPOCHS passes over the measured EN corpus, capped to the GPU budget."""
    man = json.loads(
        (Path(REPO_DIR, "data", "proc", "manifest.json")).read_text(encoding="utf-8")
    )
    en = man.get("domains", {}).get("en_literature", {}).get("train_tokens", 0)
    per_step = bs * SEQ_LEN
    want = math.ceil(en * EPOCHS / max(1, per_step))
    steps = min(want, MAX_STEPS)
    eff = steps * per_step / max(1, en)
    print(
        f"  EN tokens={en:,} -> {want} steps for {EPOCHS} epochs; "
        f"using {steps} (~{eff:.2f} epochs, cap {MAX_STEPS})"
    )
    return steps


def main() -> None:
    # ---- 1. clone ------------------------------------------------------------
    print("== [1/6] clone ==")
    if not Path(REPO_DIR).exists():
        sh(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])

    # ---- 2. patch train.py: --save-every milestone checkpoints ---------------
    print("== [2/6] patch ==")
    train = Path(REPO_DIR, "train", "train.py")
    src = train.read_text(encoding="utf-8")
    if '"--save-every"' in src:
        print("  already patched, skipping")
    else:
        arg_marker = 'ap.add_argument("--save-ckpt", type=str, default="")'
        hist_marker = 'log["history"].append(rec)'
        assert src.count(arg_marker) == 1, (
            "arg_marker not unique/missing - upstream drifted?"
        )
        assert src.count(hist_marker) == 1, (
            "hist_marker not unique/missing - upstream drifted?"
        )
        src = src.replace(
            arg_marker,
            arg_marker + '\n    ap.add_argument("--save-every", type=int, default=0,\n'
            '                    help="里程碑 checkpoint 间隔步数 (0=关闭)")',
        )
        milestone_block = textwrap.dedent("""
            if args.save_every > 0 and args.save_ckpt and (step + 1) % args.save_every == 0:
                sp = Path(args.save_ckpt).with_name(
                    Path(args.save_ckpt).stem + f"_{step+1:06d}" + Path(args.save_ckpt).suffix)
                sp.parent.mkdir(parents=True, exist_ok=True)
                sd2 = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save({"arch": args.arch, "model": sd2,
                            "config": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
                            "summary": {"step": step + 1, "val_loss": rec["val_loss"]}}, sp)
                print(f"    里程碑 checkpoint -> {sp}")
        """).rstrip("\n")
        ind = " " * 12  # matches 'log["history"].append(rec)' indentation
        src = src.replace(
            hist_marker,
            hist_marker
            + "\n"
            + "\n".join(ind + ln for ln in milestone_block.split("\n")),
        )
        train.write_text(src, encoding="utf-8")
        print("  patched train.py (+save-every)")

    # ---- 3. corpus: diverse EN books (gutendex) + Python code -----------------
    print("== [3/6] corpus ==")
    write_fetch_books()
    sh([sys.executable, "prepare_data.py", "--only-code"], cwd=REPO_DIR)
    rc, _, _ = run_capture(
        [
            sys.executable,
            "fetch_books.py",
            "--top",
            str(GUTENDEX_TOP),
            "--pages",
            str(GUTENDEX_PAGES),
        ],
        cwd=REPO_DIR,
        timeout=5400,
    )
    en_dir = Path(REPO_DIR, "data", "raw", "en_literature")
    en_files = len(list(en_dir.glob("gutendex_*.txt"))) if en_dir.exists() else 0
    if rc != 0 or en_files < MIN_BOOKS:
        print(f"  gutendex thin ({en_files} books) -> curated classics fallback")
        sh(
            [
                sys.executable,
                "-c",
                f"import prepare_data as pd; s,c=pd.collect_books({EXTRA_BOOKS}, 8); print(s); print(c)",
            ],
            cwd=REPO_DIR,
        )
    sh([sys.executable, "build_corpus.py", "--vocab-size", str(VOCAB)], cwd=REPO_DIR)

    # ---- 4. smoke test: 60 steps on the real model, catch OOM/NaN early ------
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

    # ---- 5. probe batch, estimate steps, then train ---------------------------
    print("== [5/6] train ==")
    bs = probe_batch()
    print(f"  batch={bs} selected")
    steps = estimate_steps(bs)
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
    sh(cmd, cwd=REPO_DIR, timeout=32400)  # a full training run can exceed the default

    # ---- 6. sample milestones: watch the text come together ------------------
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
        p for p in sorted(Path(REPO_DIR, "runs").glob(f"{PRESET}_{CKPT_STEM}*.pt")) if p.exists()
    ]
    sample_milestones.samples = {}
    sample_milestones.keep = []
    if not cks:
        print("no checkpoints found - training failed, nothing to sample")
        return
    # sample up to 3 (first / middle / last) to keep output small
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
                    "60",
                    "--temperature",
                    "0.7",
                ],
                cwd=REPO_DIR,
            )
            outs[p] = out[-900:] if out.strip() else f"(empty) {err.strip()[-200:]}"
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
