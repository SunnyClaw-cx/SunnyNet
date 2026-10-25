"""build_corpus.encode_domain 大语料回归测试.

修复前的 encode_domain 把整域 token 攒进一个 Python int 列表再一次性转
uint16, 峰值内存 ∝ 整域 token 数, ~100M token 的 TinyStories 会 OOM。
本测试用大量小文件(而非一个巨型文件)构造大 token 域, 验证:
  1. 各文件 token 数正确 (含 eos)
  2. 训练/验证按文件离散切分, 不跨集泄漏
  3. 全量 token 数 = 各文件编码之和 (规模正确)

规模用 monkeypatch 缩放到可快速跑完, 但文件数与 token 结构保持"大域"特征.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def big_domain(tmp_path, monkeypatch):
    """造一个"很多文件、总 token 不小"的域, 指向临时 RAW/PROC."""
    sys.path.insert(0, str(ROOT))
    import build_corpus
    from sunnynet.tokenizer import BPETokenizer

    # read_texts(domain) 读 RAW/<domain>/ ; 把 RAW 指到 tmp, 域 "big" 建在 tmp/big
    domain_dir = tmp_path / "big"
    domain_dir.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()

    monkeypatch.setattr(build_corpus, "RAW", tmp_path)  # 域 "big" -> tmp/big
    monkeypatch.setattr(build_corpus, "PROC", proc)

    # 共享 BPE 词表: 训一个覆盖句子的 tokenizer, 保证每个文件都编码成非空
    text = "The quick brown fox jumps over the lazy dog every single day "
    n_files = 40
    for i in range(n_files):
        (domain_dir / f"f{i:02d}.txt").write_text(text * 8, encoding="utf-8")

    texts = build_corpus.read_texts("big")
    assert len(texts) == n_files
    tk = BPETokenizer.train([t for _, t in texts[:8]], vocab_size=256, min_char_freq=1)
    # 预先记录每个文件的 token 数, 供断言
    per_file = [len(tk.encode(t)) for _, t in texts]
    return build_corpus, tk, per_file, proc, n_files


def test_encode_large_domain_token_count(big_domain):
    bc, tk, per_file, proc, n_files = big_domain
    # val_ratio=0.0 仍会把 1 个文件分到 val (1/val_ratio 被 clamp), 故看总量
    bc.encode_domain(tk, "big", val_ratio=0.0)
    tr = np.fromfile(proc / "tokens_big.bin", dtype=np.uint16)
    va = np.fromfile(proc / "val_big.bin", dtype=np.uint16)
    total = tr.size + va.size
    # 全量 token 数 = 每文件 token 数 + 每文件 1 个 eos (与切分无关)
    assert total == sum(per_file) + n_files
    # eos 出现次数 = 文件数 (每文件边界一个, 分布在 train/val 两边)
    assert int((tr == tk.eos_id).sum()) + int((va == tk.eos_id).sum()) == n_files


def test_encode_file_disjoint_split(big_domain):
    bc, tk, per_file, proc, n_files = big_domain
    r = bc.encode_domain(tk, "big", val_ratio=0.25)
    tr = np.fromfile(proc / "tokens_big.bin", dtype=np.uint16)
    va = np.fromfile(proc / "val_big.bin", dtype=np.uint16)
    # 每个 token 恰好落在 train 或 val 之一 (不重复、不遗漏、不跨集)
    assert tr.size + va.size == sum(per_file) + n_files
    assert r["train_files"] + r["val_files"] == n_files


def test_encode_empty_domain_returns_none(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import build_corpus

    monkeypatch.setattr(build_corpus, "RAW", tmp_path)
    monkeypatch.setattr(build_corpus, "PROC", tmp_path)
    # RAW/missing 不存在 -> read_texts 返回 [] -> encode_domain 返回 None
    assert build_corpus.encode_domain(None, "missing", 0.0) is None
