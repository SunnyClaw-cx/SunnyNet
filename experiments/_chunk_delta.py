"""分块 delta 写入的等价性验证: ops.chunk_delta_scan vs 逐步递推.

递推:  S_t = d_t * S_{t-1} - β_t k_t (k_tᵀ S_{t-1}) + β_t k_t v_tᵀ
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from sunnynet.ops import chunk_delta_scan  # noqa: E402


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def seq_delta(k, v, beta, s0, decay=None):
    """k (B,T,S), v (B,T,M), beta (B,T), s0 (B,S,M), decay (B,T). 返回 (B,T,S,M)."""
    B, T, S = k.shape
    M = v.shape[-1]
    s = s0.clone()
    outs = []
    for t in range(T):
        if decay is not None:
            s = s * decay[:, t].view(B, 1, 1)
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        pred = torch.einsum("bs,bsm->bm", kt, s)
        s = s - bt.view(B, 1, 1) * kt.unsqueeze(-1) * pred.unsqueeze(-2) \
            + bt.view(B, 1, 1) * kt.unsqueeze(-1) * vt.unsqueeze(-2)
        outs.append(s.clone())
    return torch.stack(outs, dim=1)


def run(seed=0, B=2, T=32, S=32, M=16, decay=False, chunk=8):
    torch.manual_seed(seed)
    k = unit(torch.randn(B, T, S, dtype=torch.float64))
    v = torch.randn(B, T, M, dtype=torch.float64)
    beta = torch.rand(B, T, dtype=torch.float64) * 0.9 + 0.05
    s0 = torch.randn(B, S, M, dtype=torch.float64) * 0.1
    dec = torch.rand(B, T, dtype=torch.float64) * 0.2 + 0.79 if decay else None
    ref = seq_delta(k, v, beta, s0, dec)
    got, sT = chunk_delta_scan(k, v, beta, s0, dec, chunk=chunk)
    diff = (ref - got).abs().max().item()
    diff_end = (ref[:, -1] - sT).abs().max().item()
    return diff, diff_end, ref.abs().max().item()


if __name__ == "__main__":
    out = []
    for decay in (False, True):
        for C in (4, 8, 16):
            d, de, rn = run(decay=decay, chunk=C)
            print(f"decay={int(decay)} chunk={C:3d} max|逐位置-参考|={d:.3e} "
                  f"max|末态-参考|={de:.3e} 参考量级={rn:.3f}")
            out.append({"decay": decay, "chunk": C,
                        "max_abs_diff": d, "max_end_diff": de, "ref_scale": rn})
    p = ROOT / "experiments/results/_chunk_delta_equiv.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", p)