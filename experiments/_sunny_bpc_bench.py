"""SunnyNet 及项目内架构在 TinyStories 尾部切片上的 BPC, 与 _ext_model_bench.py 同协议.

用 _tiny_bench.py 相同的数据切片 (valid 尾部) 与分词器, 使读数可和外部 HF 模型并列。
BPC = nats_per_token / (chars_per_token * ln2), 按原始字符归一化, 跨分词器可比。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

SRC = ROOT / "data" / "tinystories" / "TinyStories-valid.txt"
VOCAB = 8192
PROMPTS = ["Once upon a time, ", "One day, a little girl named Lily ",
           "The dog was very happy because "]


def load_stories(n: int) -> list[str]:
    t = SRC.read_text(encoding="utf-8")
    st = [s.strip() for s in t.split("<|endoftext|>") if s.strip()]
    return st[-n:]


def build_tok():
    from sunnynet.tokenizer import BPETokenizer
    p = ROOT / "data" / "tinystories" / "tok_fixed_8192.json"
    return BPETokenizer.load(p)


@torch.no_grad()
def eval_bpc(model, tk, stories: list[str], char_budget: int) -> dict:
    tot_nats = 0.0
    tot_tok = 0
    tot_chars = 0
    skipped = 0
    for s in stories:
        text = s[:char_budget]
        ids = tk.encode(text)
        if len(ids) < 2:
            skipped += 1
            continue
        x = torch.tensor([ids[:-1]])
        y = torch.tensor([ids[1:]])
        lg = model(x)
        tot_nats += float(F.cross_entropy(lg[0], y[0], reduction="sum"))
        tot_tok += len(ids) - 1
        tot_chars += len(text)
    npt = tot_nats / max(1, tot_tok)
    cpt = tot_chars / max(1, tot_tok)
    bpc = npt / (cpt * math.log(2))
    return {"bpc": bpc, "nats_per_token": npt, "chars_per_token": cpt,
            "pred_tokens": tot_tok, "chars": tot_chars, "kept": len(stories) - skipped,
            "skipped": skipped}


@torch.no_grad()
def gen(model, tk, prompt, n=60):
    ids = list(tk.encode(prompt))
    for _ in range(n):
        lg = model(torch.tensor([ids]))[0, -1]
        nxt = int(lg.argmax())
        ids.append(nxt)
        if nxt == tk.eos_id:
            break
    return tk.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sunnynet",
                    choices=["sunnynet", "transformer", "linear_attn", "rwkv",
                             "mamba", "hyena", "h3"])
    ap.add_argument("--preset", default="small")
    ap.add_argument("--ckpt", default="", help="已训练 checkpoint; 留空则用随机初始化")
    ap.add_argument("--stories", type=int, default=1500)
    ap.add_argument("--char-budget", type=int, default=900)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tk = build_tok()
    stories = load_stories(args.stories)

    import train as T
    sunny, base, cfg = T.build_models(args.arch, VOCAB, args.preset, None, 512, args.seed)
    model = sunny if args.arch == "sunnynet" else base
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
    model.eval()
    n_all = model.num_params()
    n_ne = model.num_params(non_embedding=True)
    print(f"[{args.arch}/{args.preset}] params={n_all:,} non_embed={n_ne:,} "
          f"vocab={tk.vocab_size} ckpt={args.ckpt or 'random'}")

    t0 = time.perf_counter()
    r = eval_bpc(model, tk, stories, args.char_budget)
    print(f"  bpc={r['bpc']:.4f} nats/tok={r['nats_per_token']:.4f} "
          f"chars/tok={r['chars_per_token']:.3f} kept={r['kept']} "
          f"ppl={math.exp(min(r['nats_per_token'],20)):.2f} {time.perf_counter()-t0:.1f}s")

    samples = {p: gen(model, tk, p) for p in PROMPTS}
    summary = {"model": f"sunnynet-project/{args.arch}", "arch": args.arch,
               "preset": args.preset, "params": n_all, "params_non_embed": n_ne,
               "vocab": tk.vocab_size, "ckpt": args.ckpt,
               "stories": args.stories, "char_budget": args.char_budget, **r,
               "wall_s": round(time.perf_counter() - t0, 1), "samples": samples}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("  ->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())