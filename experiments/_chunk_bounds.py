"""块长与衰减守卫的数值边界标定.

两种检验:
  1. 块长扫描: 块长对 chunk_delta_scan 前反向耗时与衰减梯度有限性的影响 (定最优点)
  2. 守卫检验: _safe_chunk_delta 按 p 反解的块长是否真的落在安全水位内

块长排名会随进程状态翻转, 故 timing 走独立子进程 + 逐轮配对, 不用单进程内循环的读数.

用法:
    python -X utf8 experiments/_chunk_bounds.py            # 两项全跑
    python -X utf8 experiments/_chunk_bounds.py chunk      # 只跑块长扫描
    python -X utf8 experiments/_chunk_bounds.py guard      # 只跑守卫检验
"""
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import sunnynet.ops as ops  # noqa: E402

PY = sys.executable


def _mk(B, T, S, M, seed=0):
    torch.manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(B, T, S), dim=-1)
    v = torch.randn(B, T, M)
    beta = torch.sigmoid(torch.randn(B, T))
    s0 = torch.randn(B, S, M) * 0.1
    return k, v, beta, s0


def _safe_at(p_val, chunk, B=4, T=256, S=32, M=16):
    """该 (p, chunk) 下衰减梯度是否有限. 绕过 cap 以测真实块长."""
    raw = ops._safe_chunk_delta
    ops._safe_chunk_delta = lambda requested, cap=32, p_min=None, **kw: int(requested)
    torch.manual_seed(0)
    k = torch.nn.functional.normalize(torch.randn(B, T, S), dim=-1)
    v = torch.randn(B, T, M)
    beta = torch.sigmoid(torch.randn(B, T))
    s0 = torch.randn(B, S, M) * 0.1
    try:
        pp = torch.full((B, T), p_val, requires_grad=True)
        S_all, _ = ops.chunk_delta_scan(k, v, beta, s0, decay=pp, chunk=chunk)
        S_all.pow(2).mean().backward()
        return bool(torch.isfinite(pp.grad).all())
    finally:
        ops._safe_chunk_delta = raw


def max_safe_chunk(p_val, hi=80):
    """逐块长扫出该 p 下最大的稳定块长."""
    last = 0
    for chunk in range(8, hi, 2):
        if _safe_at(p_val, chunk):
            last = chunk
        else:
            break
    return last


_CELL = r'''
import sys, time
sys.path.insert(0, r"{root}")
import torch
import sunnynet.ops as ops
chunk, pv, iters = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
raw = ops._safe_chunk_delta
ops._safe_chunk_delta = lambda requested, cap=32, p_min=None, **kw: int(requested)
torch.set_num_threads(1); torch.manual_seed(0)
B, T, S, M = 8, 512, 32, 16
k0 = torch.nn.functional.normalize(torch.randn(B, T, S), dim=-1)
v0 = torch.randn(B, T, M); b0 = torch.sigmoid(torch.randn(B, T))
s0 = torch.randn(B, S, M) * 0.1
def one():
    pp = torch.full((B, T), pv, requires_grad=True)
    Sa, _ = ops.chunk_delta_scan(k0.clone(), v0.clone(), b0.clone(), s0, decay=pp, chunk=chunk)
    Sa.pow(2).mean().backward()
for _ in range(2):
    one()
t0 = time.perf_counter()
for _ in range(iters):
    one()
print((time.perf_counter() - t0) / iters * 1000)
ops._safe_chunk_delta = raw
'''


def _cell(chunk, pv, iters=8):
    """跑一个独立子进程测一格."""
    code = _CELL.format(root=str(ROOT).replace("\\", "/"))
    out = subprocess.run([PY, "-X", "utf8", "-c", code, str(chunk), str(pv), str(iters)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def chunk_scan(chunks=(32, 48, 64, 128), rounds=4, pv=0.9988):
    """跨进程逐轮交错测块长耗时. 返回 {chunk: [ms...]}."""
    acc = {c: [] for c in chunks}
    for _ in range(rounds):
        for c in chunks:
            acc[c].append(_cell(c, pv))
    return acc


def guard_check():
    """守卫 vs 实测: 守卫给出的块长是否落在实测安全区内."""
    rows = []
    for p_val in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        n = ops._safe_chunk_delta(1000, cap=1000, p_min=p_val)
        rows.append((p_val, n, p_val ** -n, _safe_at(p_val, n)))
    return rows


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "chunk"):
        acc = chunk_scan()
        base = statistics.median(acc[32])
        print("=== 块长扫描 (独立子进程 x 4 轮交错, B=8 T=512 slots 32x16, p=0.9988) ===")
        print(f"{'chunk':>6} {'中位 ms':>9} {'极差':>7} {'相对32':>8}")
        for c, vals in acc.items():
            print(f"{c:>6} {statistics.median(vals):>9.2f} "
                  f"{max(vals)-min(vals):>7.2f} {statistics.median(vals)/base:>7.2f}x")
        print()
        print("=== 逐轮配对 (32 vs 其他) ===")
        for c, vals in acc.items():
            if c == 32:
                continue
            wins = sum(1 for a, b in zip(acc[32], vals) if a < b)
            print(f"  32 vs {c:>3}: 32 胜 {wins}/{len(vals)} 轮")
        print()
        print(f"=== p=0.5 下的最大稳定块长: {max_safe_chunk(0.5)} ===")
        print()
    if what in ("all", "guard"):
        print("=== 守卫给出的块长 vs 实测 ===")
        print(f"{'p':>6} {'守卫给L':>8} {'1/cumprod':>11} {'实测':>8}")
        for p_val, n, peak, ok in guard_check():
            print(f"{p_val:>6} {n:>8} {peak:>11.2e} {('安全' if ok else '不安全'):>8}")


if __name__ == "__main__":
    main()