"""外部 HF 模型在 TinyStories-valid 上的 BPC 读数.

按故事独立评测: 每个故事单独 tokenize, 前向一次算 auto-regressive loss.
拼接长文本会让故事边界产生困惑, 与官方 ppl 口径不符。
BPC = nats_per_token / (chars_per_token * ln2), 按原始字符归一化, 跨分词器可比。
所有模型用同一段故事切片 (前 N 个长度达标的故事)。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "tinystories" / "TinyStories-valid.txt"
PROMPTS = ["Once upon a time, ", "One day, a little girl named Lily ",
           "The dog was very happy because "]


def load_stories(n: int) -> list[str]:
    """取 valid 尾部的故事: 与 _tiny_bench.py 验证切片同侧(尾部),
    对 SunnyNet 是 held-out; 外部模型训练用的是 train split, 同为 held-out。"""
    t = SRC.read_text(encoding="utf-8")
    st = [s.strip() for s in t.split("<|endoftext|>") if s.strip()]
    return st[-n:]


class SPAdapter:
    """transformers 5.x 的 LlamaTokenizer 对本目录的 sentencepiece 文件返回空串,
    直接用 sentencepiece 本体, 只暴露评测需要的最小接口。"""

    def __init__(self, path: Path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.eos_token_id = self.sp.eos_id()
        self.pad_token_id = self.sp.pad_id()
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, s: str) -> list[int]:
        return self.sp.encode(s)

    def decode(self, ids) -> str:
        return self.sp.decode([int(i) for i in ids])


def encode(tok, s: str) -> list[int]:
    if isinstance(tok, SPAdapter):
        return tok.encode(s)
    return tok(s).input_ids


def decode(tok, ids) -> str:
    if isinstance(tok, SPAdapter):
        return tok.decode(ids)
    return tok.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def eval_bpc(model, tok, stories: list[str], char_budget: int):
    """在字符空间统一截断, 所有模型看到完全相同的文本, BPC 才可直接比。
    按 token 截断会让不同分词器看到不同字符范围, 平均值不在同一总体。"""
    tot_nats = 0.0
    tot_tok = 0
    tot_chars = 0
    skipped = 0
    kept = 0
    for s in stories:
        text = s[:char_budget]
        ids = encode(tok, text)
        if len(ids) < 2:
            skipped += 1
            continue
        x = torch.tensor([ids[:-1]])
        y = torch.tensor([ids[1:]])
        lg = model(x).logits[0]
        tot_nats += float(F.cross_entropy(lg, y[0], reduction="sum"))
        tot_tok += len(ids) - 1
        tot_chars += len(text)
        kept += 1
    npt = tot_nats / max(1, tot_tok)
    cpt = tot_chars / max(1, tot_tok)
    bpc = npt / (cpt * math.log(2))
    return {"bpc": bpc, "nats_per_token": npt, "chars_per_token": cpt,
            "pred_tokens": tot_tok, "chars": tot_chars, "skipped": skipped,
            "kept": kept}


@torch.no_grad()
def gen(model, tok, prompt, n=60):
    if isinstance(tok, SPAdapter):
        ids = torch.tensor([tok.encode(prompt)])
        pad = tok.pad_token_id if tok.pad_token_id >= 0 else 0
    else:
        ids = tok(prompt, return_tensors="pt").input_ids
        pad = tok.pad_token_id or 0
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=pad)
    return decode(tok, out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--local", action="store_true", help="模型是本机目录")
    ap.add_argument("--stories", type=int, default=1500)
    ap.add_argument("--char-budget", type=int, default=1000,
                    help="每个故事截断到的字符数, 所有模型必须一致")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--trust-remote", action="store_true")
    ap.add_argument("--slow-tok", action="store_true",
                    help="缺 tokenizer.json 时走 vocab.json + merges.txt 的 slow 路径")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"trust_remote_code": True} if args.trust_remote else {}
    kw["local_files_only"] = args.local
    sp_path = Path(args.model) / "tokenizer.model"
    if sp_path.exists():
        # llama2.c 转出的目录只有 sentencepiece 文件, HF 封装在本机读不出内容
        tok = SPAdapter(sp_path)
        kw["local_files_only"] = True
    else:
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=not args.slow_tok, **kw)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, **kw)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    cfg = model.config
    stories = load_stories(args.stories)
    print(f"[{args.model}] params={n_par:,} vocab={cfg.vocab_size} "
          f"layers={getattr(cfg,'num_hidden_layers','?')} d={getattr(cfg,'hidden_size','?')}")

    t0 = time.perf_counter()
    r = eval_bpc(model, tok, stories, args.char_budget)
    print(f"  bpc={r['bpc']:.4f} nats/tok={r['nats_per_token']:.4f} "
          f"chars/tok={r['chars_per_token']:.3f} kept={r['kept']} skipped={r['skipped']} "
          f"ppl={math.exp(min(r['nats_per_token'],20)):.2f} {time.perf_counter()-t0:.1f}s")

    samples = {p: gen(model, tok, p) for p in PROMPTS}
    summary = {"model": args.model, "params": n_par, "vocab": cfg.vocab_size,
               "architecture": (cfg.architectures or ["?"])[0],
               "layers": getattr(cfg, "num_hidden_layers", None),
               "hidden_size": getattr(cfg, "hidden_size", None),
               "max_len": args.char_budget, "stories": args.stories, **r,
               "wall_s": round(time.perf_counter() - t0, 1), "samples": samples}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("  ->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())