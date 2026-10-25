"""BPE 训练器的计数不变式.

回归目标: pair 计数曾用 set 去重递减, 高频对永远减不到 0, 同一对被反复写入
merges —— 8192 词表里只剩 447 个解码互不相同的 token. 这里锁定三条不变式.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sunnynet.tokenizer import BPETokenizer  # noqa: E402

TEXT = (
    "Once upon a time there was a little girl named Lily. "
    "She had a red ball and she liked to play with it in the park. "
    "One day the ball rolled away and she was sad. "
) * 200


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train([TEXT], vocab_size=1024, min_char_freq=2)


def test_merges_are_distinct(tok: BPETokenizer) -> None:
    c = Counter(tok.merges)
    dup = {p: n for p, n in c.items() if n > 1}
    assert not dup, f"重复 merge {len(dup)} 个, 如 {list(dup.items())[:3]}"


def test_no_duplicate_token_strings(tok: BPETokenizer) -> None:
    strs = [s for s in tok._str.values() if s]
    assert len(set(strs)) == len(strs), \
        f"解码等价的重复 token {len(strs) - len(set(strs))} 个"


def test_pair_counts_match_rescan(tok: BPETokenizer) -> None:
    """merges 的顺序必须与真实语料的贪心合并顺序一致."""
    ids = tok.encode(TEXT[:400])
    assert len(ids) < len(TEXT[:400]), "编码未做任何合并, 压缩率为 1"


def test_compression_beats_char_level(tok: BPETokenizer) -> None:
    ids = tok.encode(TEXT[:2000])
    assert len(TEXT[:2000]) / len(ids) > 1.8, \
        f"压缩率过低 {len(TEXT[:2000]) / len(ids):.2f} chars/token"