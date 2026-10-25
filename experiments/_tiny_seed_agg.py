"""跨架构多种子聚合: 判定相邻排名是否超出种子噪声.

单种子读数下相邻架构的 BPC 差可小到 1%, 无法判定排名是否稳定.
本脚本读 results/tiny/ 下同架构多种子 JSON, 给出均值/极差与组间差.

用法: python -X utf8 experiments/_tiny_seed_agg.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "results" / "tiny"

# 归档文件名 -> (架构, 种子, lr). 主格无种子后缀, 视为 seed=1234 (脚本默认).
# transformer 主格是 6e-3 读数, 已由 --force-lr 3e-3 复测取代(1.2692 -> 1.2478),
# 故 transformer 取 lr3e3 那份为规范读数, 与其余六格同 lr.
MAIN = {
    "fixed_rwkv": ("rwkv", 1234, 3e-3),
    "fixed_mamba": ("mamba", 1234, 3e-3),
    "fixed_en_sunnynet": ("sunnynet", 1234, 3e-3),
    "fixed_h3": ("h3", 1234, 3e-3),
    "fixed_hyena": ("hyena", 1234, 3e-3),
    "fixed_transformer_lr3e3": ("transformer", 1234, 3e-3),
    "fixed_linear_attn": ("linear_attn", 1234, 3e-3),
}
SEEDED = re.compile(r"^ms_(?P<arch>.+)_s(?P<seed>\d+)\.json$")


def collect() -> dict[str, list[tuple[int, float, float]]]:
    out: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for stem, (arch, seed, lr) in MAIN.items():
        p = TINY / f"{stem}.json"
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            out[arch].append((seed, float(d["final_bpc"]), lr))
    for p in sorted(TINY.glob("ms_*.json")):
        m = SEEDED.match(p.name)
        if not m:
            continue
        d = json.load(open(p, encoding="utf-8"))
        out[m["arch"]].append((int(m["seed"]), float(d["final_bpc"]), float(d.get("lr", 0.0))))
    return out


def main():
    data = collect()
    if not data:
        print("no readings")
        return 1
    rows = []
    for arch, v in data.items():
        v.sort()
        bpcs = [b for _, b, _ in v]
        lrs = sorted({lr for _, _, lr in v})
        rows.append((arch, len(v), sum(bpcs) / len(bpcs), min(bpcs), max(bpcs), lrs))
    rows.sort(key=lambda r: r[2])
    print(f"{'arch':12s} {'n':>2s} {'mean_bpc':>9s} {'min':>8s} {'max':>8s} {'range%':>7s}  lr")
    for arch, n, mean, lo, hi, lrs in rows:
        rng = (hi - lo) / mean * 100
        print(f"{arch:12s} {n:>2d} {mean:>9.4f} {lo:>8.4f} {hi:>8.4f} {rng:>6.2f}%  {lrs}")
    print()
    multi = [(a, n, mean, lo, hi) for a, n, mean, lo, hi, _ in rows if n >= 2]
    if multi:
        spread = max((hi - lo) / mean for _, _, mean, lo, hi in multi) * 100
        print(f"多种子架构的最大极差: {spread:.2f}%")
    else:
        print("尚无多种子架构")
    for i in range(len(rows) - 1):
        a, _, ma, _, _, _ = rows[i]
        b, _, mb, _, _, _ = rows[i + 1]
        gap = (mb - ma) / mb * 100
        print(f"  {a} -> {b}: 均值差 {gap:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())