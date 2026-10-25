"""语料构建: 真实文本 -> BPE token 流 -> 域分片 memmap.

划分策略: 按**文件级**切分验证集 (验证集文件全程不参与训练),
避免同一文本的相邻片段同时落入训练与验证造成的信息泄漏.

产出:
  data/proc/tokens_<domain>.bin     uint16 token 流
  data/proc/val_<domain>.bin        验证集 token 流
  data/proc/manifest.json           各域 token 数 / 文件统计 / 词表信息
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from sunnynet.tokenizer import BPETokenizer, pretokenize

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "proc"
DOMAIN_ZH = "zh_ancient"
DOMAIN_EN = "en_literature"
DOMAIN_CODE = "code_python"
DOMAINS = [DOMAIN_ZH, DOMAIN_EN, DOMAIN_CODE]


def read_texts(domain: str) -> list[tuple[str, str]]:
    d = RAW / domain
    if not d.exists():
        return []
    return [(f.name, f.read_text(encoding="utf-8", errors="ignore"))
            for f in sorted(d.glob("*.txt"))]


def train_tokenizer(vocab_size: int, sample_chars: int) -> BPETokenizer:
    texts: list[str] = []
    budget = sample_chars
    per_domain = max(1, budget // max(1, len(DOMAINS)))
    for dom in DOMAINS:
        used = 0
        for _, t in read_texts(dom):
            if used >= per_domain:
                break
            texts.append(t[:per_domain - used])
            used += len(t[:per_domain - used])
    return BPETokenizer.train(texts, vocab_size=vocab_size, min_char_freq=3, verbose=True)


def encode_domain(tk: BPETokenizer, domain: str, val_ratio: float, verbose: bool = True):
    """按文件切分训练/验证后**拼接**成连续 token 流.

    拼接而非逐文件独立编码, 是为了不因单文件过小而丢弃数据
    (代码类文件普遍很小, 逐文件过滤会损失绝大部分样本).
    训练/验证仍按文件离散切分, 拼接不引入跨集泄漏.
    """
    files = read_texts(domain)
    if not files:
        return None
    texts = sorted(files, key=lambda kv: kv[0])
    step = max(1, int(round(1.0 / max(val_ratio, 1e-6))))
    val_files = texts[::step]
    train_files = [kv for kv in texts if kv not in val_files]
    if not train_files:
        train_files, val_files = texts, texts[:1]

    def enc(pairs):
        # 逐文件编码为 uint16 再拼接: 峰值内存受最大单文件约束, 而非整域 token
        # 数 (旧实现把整域 Python int 列表攒在内存, 大语料 [如 TinyStories 的
        # ~1.2B token] 会 OOM).
        parts: list[np.ndarray] = []
        for _, t in pairs:
            parts.append(np.asarray(tk.encode(t), dtype=np.uint16))
            parts.append(np.asarray([tk.eos_id], dtype=np.uint16))
        return np.concatenate(parts)

    tr = enc(train_files)
    va = enc(val_files)
    PROC.mkdir(parents=True, exist_ok=True)
    tr.tofile(PROC / f"tokens_{domain}.bin")
    va.tofile(PROC / f"val_{domain}.bin")
    if verbose:
        print(f"  {domain:14s} files train/val = {len(train_files)}/{len(val_files)}  "
              f"tokens {tr.size:,} / {va.size:,}")
    return {"train_tokens": int(tr.size), "val_tokens": int(va.size),
            "train_files": len(train_files), "val_files": len(val_files)}


def load_tokens(domain: str, split: str = "train") -> np.ndarray:
    name = f"tokens_{domain}.bin" if split == "train" else f"val_{domain}.bin"
    return np.fromfile(PROC / name, dtype=np.uint16)


def load_manifest() -> dict:
    return json.loads((PROC / "manifest.json").read_text(encoding="utf-8"))


def load_tokenizer() -> BPETokenizer:
    return BPETokenizer.load(PROC / "tokenizer.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=12000)
    ap.add_argument("--val-ratio", type=float, default=0.06)
    ap.add_argument("--tok-sample-chars", type=int, default=6_000_000)
    args = ap.parse_args()

    # 分词器训练前清掉旧产物, 避免词表与 token 流不一致
    PROC.mkdir(parents=True, exist_ok=True)
    for p in PROC.glob("*.bin"):
        p.unlink()

    print("[1/3] 训练 BPE 分词器 ...")
    tk = train_tokenizer(args.vocab_size, args.tok_sample_chars)
    print(f"  vocab_size={tk.vocab_size} merges={len(tk.merges)}")
    tk.save(PROC / "tokenizer.json")

    print("[2/3] 编码各域 ...")
    man = {"vocab_size": tk.vocab_size, "val_ratio": args.val_ratio, "domains": {}}
    for dom in DOMAINS:
        r = encode_domain(tk, dom, args.val_ratio)
        if r:
            man["domains"][dom] = r

    man["total_train_tokens"] = sum(d["train_tokens"] for d in man["domains"].values())
    man["total_val_tokens"] = sum(d["val_tokens"] for d in man["domains"].values())
    (PROC / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

    print("[3/3] 完成")
    for d, v in man["domains"].items():
        print(f"  {d:14s} train={v['train_tokens']:>10,}  val={v['val_tokens']:>8,}")
    print(f"  total train={man['total_train_tokens']:,}  val={man['total_val_tokens']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())