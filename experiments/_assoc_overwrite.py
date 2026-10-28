"""关联记忆写入规则的能力对照: 软凸组合 vs delta 覆盖.

AssocMemory 现在用软写入 S = p⊙S + (1-p)⊙(v⊗k), 同一 key 反复写入会累加泄漏,
不能覆盖。delta rule S = S(I - βkkᵀ) + βvkᵀ 能精确覆盖。

本脚本直接从输入投影之外测这两条规则本身: 给定单位 key 序列和 value 序列,
写入 N 个 (k_i, v_i), 再覆盖其中一半为新值, 用 q=k_i 读出, 看读回的是新值还是混合值。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

SLOTS = 256
M = 32
N_KEYS = 8
N_OVER = 4


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def soft_write(S, k, v, p):
    """S (slots,m), k (slots,), v (m,), p (m,) -> S'."""
    inj = v.unsqueeze(0) * k.unsqueeze(1)          # (slots,m)
    return p.unsqueeze(0) * S + (1.0 - p).unsqueeze(0) * inj


def delta_write(S, k, v, beta, p):
    """Gated delta: 先按 p 衰减, 再做 rank-1 覆盖."""
    Sd = p.unsqueeze(0) * S
    kk = k.unsqueeze(1) * k.unsqueeze(0)           # (slots,slots) = k kᵀ
    pred = kk @ Sd                                  # (slots,m) = (k kᵀ) S
    return Sd - beta * pred + beta * (v.unsqueeze(0) * k.unsqueeze(1))


def run(rule, pval, beta=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    keys = unit(torch.randn(N_KEYS, SLOTS, generator=g))
    v_old = torch.randn(N_KEYS, M, generator=g)
    v_new = torch.randn(N_KEYS, M, generator=g)
    p = torch.full((M,), pval)

    S = torch.zeros(SLOTS, M)
    for i in range(N_KEYS):
        k, v = keys[i], v_old[i]
        S = soft_write(S, k, v, p) if rule == "soft" else delta_write(S, k, v, beta, p)
    for i in range(N_OVER):
        k, v = keys[i], v_new[i]
        S = soft_write(S, k, v, p) if rule == "soft" else delta_write(S, k, v, beta, p)

    o = keys @ S
    err_new_norm = (o[:N_OVER] - v_new[:N_OVER]).norm(dim=-1) / \
        v_new[:N_OVER].norm(dim=-1).clamp_min(1e-8)
    err_keep = (o[N_OVER:] - v_old[N_OVER:]).norm(dim=-1) / \
        v_old[N_OVER:].norm(dim=-1).clamp_min(1e-8)
    return {
        "rule": rule, "p": pval if rule == "soft" else None, "beta": beta,
        "overwritten_rel_err": round(float(err_new_norm.mean()), 4),
        "kept_rel_err": round(float(err_keep.mean()), 4),
    }


if __name__ == "__main__":
    out = []
    for pval in (0.1, 0.5, 0.9):
        r = run("soft", pval)
        out.append(r)
        print(f"soft  p={pval:.1f}  覆盖后读回相对误差={r['overwritten_rel_err']:.4f}  "
              f"未覆盖保持相对误差={r['kept_rel_err']:.4f}")
    r = run("delta", 1.0, beta=1.0)
    out.append(r)
    print(f"delta beta=1.0  覆盖后读回相对误差={r['overwritten_rel_err']:.4f}  "
          f"未覆盖保持相对误差={r['kept_rel_err']:.4f}")
    p = ROOT / "experiments/results/_assoc_overwrite.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", p)