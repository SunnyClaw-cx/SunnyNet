"""衰减归约方向的数值代价: amin(最快遗忘通道) vs amax(最慢).

分块 delta 只收标量衰减, 逐通道 (B,T,m) 必须压成 (B,T). 块内要算 1/∏_{i<=t} d_i,
量级 ~ d_min^-L, 所以归约取哪个通道直接决定条件数:
  amin -> 取下界, 1/∏ 爆量最大;  amax -> 取上界, 条件数最小.

用法: python -X utf8 experiments/_decay_reduce.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.ops import chunk_delta_scan  # noqa: E402


def main(B=8, T=256, S=32, M=16, h_min=2.0, h_max=2048.0, seed=0):
    torch.manual_seed(seed)
    span = math.log(h_max / h_min)
    H = h_min * torch.exp(span * torch.sigmoid(torch.randn(B, T, M)))
    d_pc = 1.0 - 1.0 / H
    print(f"逐通道保持率 d ∈ [{float(d_pc.min()):.4f}, {float(d_pc.max()):.4f}]  "
          f"理论 [{1 - 1 / h_min:.4f}, {1 - 1 / h_max:.6f}]")

    k = torch.nn.functional.normalize(torch.randn(B, T, S), dim=-1)
    v = torch.randn(B, T, M)
    beta = torch.sigmoid(torch.randn(B, T))
    s0 = torch.randn(B, S, M) * 0.1

    out = {}
    for name, p in (("amin", d_pc.amin(-1)), ("amax", d_pc.amax(-1))):
        row = {}
        for chunk in (32, 64):
            kk, vv = k.clone().requires_grad_(), v.clone().requires_grad_()
            bb, pp = beta.clone().requires_grad_(), p.clone().requires_grad_()
            Sall, _ = chunk_delta_scan(kk, vv, bb, s0, decay=pp, chunk=chunk)
            Sall.pow(2).mean().backward()
            finite = all(bool(torch.isfinite(t.grad).all()) for t in (kk, vv, bb, pp))
            peak = float((1.0 / p.clamp_min(1e-12) ** chunk).max())
            row[chunk] = peak
            print(f"  reduce={name:4s} chunk={chunk:3d}: 1/min(cumprod)={peak:.2e}  "
                  f"反向有限={finite}{'  [超 fp32 3.4e38]' if peak > 3.4e38 else ''}")
        out[name] = row
    return out


if __name__ == "__main__":
    main()