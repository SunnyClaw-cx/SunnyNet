"""软写 vs delta 写的训练步耗时对照.

同一 harness 只换 assoc_write, 每格独立子进程 (单进程多格会互相污染缓存).
用法: python -X utf8 experiments/_assoc_ab_speed.py <rule> <T> [slots] [m] [iters]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

VOCAB = 8192


def main(rule: str, T: int, slots: int, m: int, iters: int):
    from sunnynet.model import SunnyConfig, SunnyNet
    write, decay = (rule.split("_", 1) if "_" in rule else (rule, "reduce"))
    ab = {"assoc": True, "assoc_slots": slots, "assoc_m": m, "assoc_every": 1,
          "assoc_write": write, "assoc_shift": True, "assoc_decay": decay}
    cfg = SunnyConfig(vocab_size=VOCAB, dim=128, inner=192, rank=64,
                      layers=4, h_max=2048.0, scan_chunk=64, ablation=ab)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = SunnyNet(cfg)
    assert model.blocks[0].assoc.write_rule == write
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, VOCAB, (8, T))
    y = torch.randint(0, VOCAB, (8, T))
    for _ in range(3):
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"{rule} slots={slots} m={m} T={T} {ms:.1f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]),
         int(sys.argv[3]) if len(sys.argv) > 3 else 32,
         int(sys.argv[4]) if len(sys.argv) > 4 else 16,
         int(sys.argv[5]) if len(sys.argv) > 5 else 12)