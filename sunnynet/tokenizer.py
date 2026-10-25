"""BPE 子词分词器 (SunnyNet 输入层, 不属于原创网络核心算法).

从零实现, 不依赖 tokenizers / sentencepiece. 预分词把 CJK 连续串、拉丁词、
数字、符号分开, 使中文按短语合并、英文按词合并, 同时兼容代码符号.

词表 = 4 个特殊符号 + 高频字符 + BPE 合并项.
self.merge_tokens[k] 是第 k 次合并产生的新词表项, 用于编码时把 (a,b) 映射到新 id.
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

__all__ = ["BPETokenizer", "pretokenize", "PRETOK_RE"]

PRETOK_RE = re.compile(
    r"[A-Za-z]+|\d+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
    r"|[^\sA-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|\s+"
)
MAX_CHUNK = 48
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def pretokenize(text: str) -> list[str]:
    out: list[str] = []
    for piece in PRETOK_RE.findall(text):
        if len(piece) <= MAX_CHUNK:
            out.append(piece)
        else:
            out.extend(piece[i:i + MAX_CHUNK] for i in range(0, len(piece), MAX_CHUNK))
    return out


class BPETokenizer:
    def __init__(self, merges=None, vocab=None, merge_tokens=None):
        self.merges: list[tuple[int, int]] = list(merges or [])
        self.merge_tokens: list[str] = list(merge_tokens or [])
        self.ranks: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        self.vocab: dict[str, int] = dict(vocab or {})
        self.inv: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self._str: dict[int, str] = {}
        self._build_decode_table()

    def _build_decode_table(self) -> None:
        """把每个 token id 展开为原始字符序列, 用于无歧义解码."""
        spec = set(SPECIALS)
        for tok, i in self.vocab.items():
            if tok in spec:
                self._str[i] = ""
            elif tok.startswith("<m") and tok.endswith(">"):
                continue
            else:
                self._str[i] = tok
        for k, (a, b) in enumerate(self.merges):
            tid = self.vocab[self.merge_tokens[k]]
            self._str[tid] = self._str.get(a, "") + self._str.get(b, "")

    # ---------------- 训练 ----------------
    @classmethod
    def train(cls, texts, vocab_size: int = 9000, min_char_freq: int = 3,
              verbose: bool = False) -> "BPETokenizer":
        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(pretokenize(text))
        records = sorted(chunk_counts.items(), key=lambda kv: -kv[1])

        char_freq: Counter[str] = Counter()
        for chunk, c in records:
            char_freq.update({ch: c for ch in set(chunk)})
        chars = sorted((ch for ch, c in char_freq.items() if c >= min_char_freq),
                       key=lambda ch: (-char_freq[ch], ch))

        vocab = {s: i for i, s in enumerate(SPECIALS)}
        for ch in chars:
            vocab.setdefault(ch, len(vocab))
        unk = vocab["<unk>"]

        seqs: list[list[int]] = []
        freqs: list[int] = []
        for chunk, c in records:
            ids = [vocab[ch] for ch in chunk if ch in vocab]
            if ids:
                seqs.append(ids)
                freqs.append(c)
        if verbose:
            print(f"[bpe] chunks={len(records)} chars={len(chars)} seqs={len(seqs)}")

        pair_counts: Counter = Counter()
        pair_seqs: dict[tuple[int, int], set[int]] = defaultdict(set)
        for si, ids in enumerate(seqs):
            f = freqs[si]
            for p in zip(ids, ids[1:]):
                pair_counts[p] += f
                pair_seqs[p].add(si)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        merge_tokens: list[str] = []
        n_target = max(0, vocab_size - len(vocab))

        while len(merges) < n_target and heap:
            negc, pair = heapq.heappop(heap)
            cur = pair_counts.get(pair, 0)
            if cur <= 0 or -negc != cur:
                if cur > 0:
                    heapq.heappush(heap, (-cur, pair))
                continue
            a, b = pair
            if not any(ids[i] == a and ids[i + 1] == b
                       for si in pair_seqs.get(pair, ())
                       for ids in (seqs[si],)
                       for i in range(len(ids) - 1)):
                pair_counts.pop(pair, None)
                pair_seqs.pop(pair, None)
                continue
            new_id = len(vocab)
            tok = f"<m{new_id}>"
            vocab[tok] = new_id
            merges.append(pair)
            merge_tokens.append(tok)

            for si in list(pair_seqs.get(pair, ())):
                ids = seqs[si]
                f = freqs[si]
                out, i, changed = [], 0, False
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(new_id)
                        i += 2
                        changed = True
                    else:
                        out.append(ids[i])
                        i += 1
                if not changed:
                    continue
                # 计数必须按出现次数增删; 用 set 去重会让高频对永远减不到 0,
                # 同一对反复写入 merges, 词表被解码等价的重复 token 占满.
                old = Counter(zip(ids, ids[1:]))
                new = Counter(zip(out, out[1:]))
                seqs[si] = out
                for p in set(old) | set(new):
                    c = pair_counts.get(p, 0) + f * (new.get(p, 0) - old.get(p, 0))
                    if c > 0:
                        pair_counts[p] = c
                        heapq.heappush(heap, (-c, p))
                    else:
                        pair_counts.pop(p, None)
                for p in old:
                    if p not in new:
                        pair_seqs.get(p, set()).discard(si)
                for p in new:
                    pair_seqs[p].add(si)
            pair_seqs.pop(pair, None)

            if verbose and len(merges) % 1000 == 0:
                print(f"[bpe] merges={len(merges)} vocab={len(vocab)}")

        return cls(merges=merges, vocab=vocab, merge_tokens=merge_tokens)

    # ---------------- 编码 ----------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        unk = self.vocab["<unk>"]
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        for chunk in pretokenize(text):
            ids = [self.vocab.get(ch, unk) for ch in chunk]
            out.extend(self._merge(ids))
        if add_eos:
            out.append(self.eos_id)
        return out

    def _merge(self, ids: list[int]) -> list[int]:
        if len(ids) < 2:
            return ids
        while True:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                return ids
            new_id = self.vocab[self.merge_tokens[best_rank]]
            ids = ids[:best_i] + [new_id] + ids[best_i + 2:]

    def decode(self, ids) -> str:
        if not self._str:
            self._build_decode_table()
        return "".join(self._str.get(int(i), "") for i in ids)

    def tokens_to_text(self, ids) -> list[str]:
        """返回每个 token 的可见文本, 便于逐 token 检查."""
        if not self._str:
            self._build_decode_table()
        return [self._str.get(int(i), "") for i in ids]

    def encode_batch(self, texts, add_bos=False, add_eos=False) -> list[list[int]]:
        return [self.encode(t, add_bos, add_eos) for t in texts]

    # ---------------- 属性 ----------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<eos>"]

    # ---------------- 持久化 ----------------
    def save(self, path) -> None:
        Path(path).write_text(json.dumps(
            {"vocab": self.vocab,
             "merges": [list(m) for m in self.merges],
             "merge_tokens": self.merge_tokens}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(merges=[tuple(m) for m in d["merges"]], vocab=d["vocab"],
                   merge_tokens=d["merge_tokens"])