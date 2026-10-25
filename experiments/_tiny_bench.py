"""TinyStories 归因格: 分词器 vs 架构.

数据用 TinyStories-valid 做故事级 92/8 切分, 不下载 2GB train split,
只为判定"两万步仍不连贯"是输入层还是网络层的问题, 不作规模训练.
每格独立子进程; 四格 = {多域BPE, 英文BPE} x {SunnyNet, Transformer}.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

VOCAB = 8192
SRC = ROOT / "data" / "tinystories" / "TinyStories-valid.txt"
PROMPTS = ["Once upon a time, ", "One day, a little girl named Lily ",
           "The dog was very happy because "]


def load_stories() -> list[str]:
    t = SRC.read_text(encoding="utf-8")
    return [s.strip() for s in t.split("<|endoftext|>") if s.strip()]


def build_tok(mode: str):
    from sunnynet.tokenizer import BPETokenizer
    if mode == "multi":
        from build_corpus import load_tokenizer
        return load_tokenizer()
    if mode == "broken_en":
        return BPETokenizer.load(ROOT / "data" / "ref_kaggle" / "p100m_en_tokenizer.json")
    if mode == "fixed_en":
        p = ROOT / "data" / "tinystories" / "tok_fixed_8192.json"
        if p.exists():
            return BPETokenizer.load(p)
        st = load_stories()
        n_val = max(200, len(st) // 12)
        sample = "\n".join(st[: len(st) - n_val])
        tk = BPETokenizer.train([sample[:8_000_000]], vocab_size=VOCAB, min_char_freq=3)
        tk.save(p)
        return tk
    raise SystemExit(f"unknown tok mode {mode}")


class Stream:
    def __init__(self, arr: np.ndarray, L: int, B: int, seed: int = 0):
        self.a, self.L, self.B = arr, L, B
        self.rng = np.random.default_rng(seed)

    def batch(self):
        B, L = self.B, self.L
        s = self.rng.integers(0, len(self.a) - L - 1, size=B)
        w = np.stack([self.a[i:i + L + 1] for i in s]).astype(np.int64)
        return torch.from_numpy(w[:, :-1]), torch.from_numpy(w[:, 1:])

    def fixed(self, n: int, seed: int = 12345):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            s = rng.integers(0, len(self.a) - self.L - 1, size=self.B)
            w = np.stack([self.a[i:i + self.L + 1] for i in s]).astype(np.int64)
            out.append((torch.from_numpy(w[:, :-1]), torch.from_numpy(w[:, 1:])))
        return out


def lr_at(step, total, base, warmup, min_ratio=0.1):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches:
        lg = model(x)
        tot += float(torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum"))
        n += y.numel()
    model.train()
    return tot / max(1, n)


@torch.no_grad()
def evaluate_bpc(model, va_ids: np.ndarray, chars: int, L: int, B: int = 8) -> float:
    """bits-per-character: 跨分词器唯一可比的指标 (nats / (chars * ln2))."""
    model.eval()
    tot_nats, tot_tok = 0.0, 0
    step = max(1, (len(va_ids) - L - 1) // 32)
    for s in range(0, len(va_ids) - L - 1, step):
        w = va_ids[s:s + L + 1].astype(np.int64)
        x = torch.from_numpy(w[None, :-1])
        y = torch.from_numpy(w[None, 1:])
        lg = model(x)
        tot_nats += float(torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum"))
        tot_tok += y.numel()
    model.train()
    nats_per_tok = tot_nats / max(1, tot_tok)
    chars_per_tok = chars / max(1, len(va_ids))
    return nats_per_tok / (chars_per_tok * math.log(2))


@torch.no_grad()
def generate(model, tk, prompt, n=80):
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
    ap.add_argument("--tok", choices=["multi", "broken_en", "fixed_en"], default="fixed_en")
    ap.add_argument("--arch", default="sunnynet",
                    choices=["sunnynet", "transformer", "linear_attn", "rwkv",
                             "mamba", "hyena", "h3"])
    ap.add_argument("--preset", default="small")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--force-lr", type=float, default=0.0,
                    help="覆盖按架构写死的 lr, 用于跨架构 lr 扫描")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--out", default="")
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    tk = build_tok(args.tok)
    st = load_stories()
    n_val = max(200, len(st) // 12)
    tr_s, va_s = st[: len(st) - n_val], st[len(st) - n_val:]

    def enc(ss):
        out = [np.asarray(tk.encode(s) + [tk.eos_id], dtype=np.uint16) for s in ss]
        return np.concatenate(out)

    tr = enc(tr_s)
    va = enc(va_s)
    cpt = len(tr) / max(1, (len(tr) + len(va)))
    print(f"[{args.tok}] vocab={tk.vocab_size} train_tok={tr.size:,} val_tok={va.size:,} "
          f"chars/token={sum(len(s) for s in tr_s)/max(1,tr.size):.3f}")

    import train as T
    sunny, base, cfg = T.build_models(args.arch, VOCAB, args.preset, None,
                                      args.seq_len, args.seed)
    model = sunny if args.arch == "sunnynet" else base
    n_all = model.num_params()
    n_ne = model.num_params(non_embedding=True)
    print(f"[{args.arch}] params={n_all:,} (non-embed {n_ne:,})")

    mixer = Stream(tr, args.seq_len, args.batch_size, seed=args.seed)
    val_b = Stream(va, args.seq_len, args.batch_size, seed=1).fixed(8)
    val_chars = sum(len(s) for s in va_s)

    mlu = args.force_lr if args.force_lr > 0 else args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=mlu, betas=(0.9, 0.95),
                            weight_decay=0.01)

    hist = []
    t0 = time.perf_counter()
    loss = torch.tensor(float("nan"))
    for step in range(args.steps):
        lr = lr_at(step, args.steps, mlu, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = mixer.batch()
        lg = model(x)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        if not math.isfinite(gn):
            print(f"  [发散] step {step+1}")
            break
        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_b)
            hist.append({"step": step + 1, "train_loss": round(float(loss.detach()), 4),
                         "val_loss": round(vl, 4), "ppl": round(math.exp(min(vl, 20)), 3),
                         "gn": round(gn, 3)})
            print(f"  step {step+1:5d} loss={float(loss):.4f} val={vl:.4f} "
                  f"ppl={math.exp(min(vl,20)):.2f} gn={gn:.3f} {time.perf_counter()-t0:.0f}s")

    samples = {p: generate(model, tk, p) for p in PROMPTS}
    final_bpc = evaluate_bpc(model, va, val_chars, args.seq_len)
    print(f"  [bpc] {final_bpc:.4f} bits/char  (chars/tok={sum(len(s) for s in va_s)/max(1,va.size):.3f})")
    summary = {"tok": args.tok, "arch": args.arch, "preset": args.preset,
               "lr": mlu,
               "params": n_all, "params_non_embed": n_ne, "vocab": tk.vocab_size,
               "steps": args.steps, "tokens_trained": args.steps * args.batch_size * args.seq_len,
               "val_tokens": int(va.size), "val_chars": val_chars,
               "tokens_per_param": round(args.steps * args.batch_size * args.seq_len / n_all, 2),
               "chars_per_token": round(sum(len(s) for s in tr_s) / max(1, tr.size), 4),
               "final_bpc": round(final_bpc, 4),
               "history": hist, "samples": samples,
               "wall_s": round(time.perf_counter() - t0, 1)}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.ckpt:
        p = Path(args.ckpt)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": args.arch, "tok": args.tok,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg}, p)
    print(f"=== {args.tok}/{args.arch} done: ppl={hist[-1]['ppl'] if hist else None} "
          f"wall={summary['wall_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())