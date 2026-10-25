"""在 x86 上执行真 AVX-512 源码 (experiments/_avx512_shim/immintrin.h 提供 shim).

本机 CPU 只有 AVX2, AVX-512 指令无法执行。这里把 __m512/__m512i 实现成 16 元数组,
逐元素运算与 AVX-512 一致; 组内置换用源码里的真索引向量, 因此"低 8 lane = 组 A、
高 8 lane = 组 B"的 lane 语义与尾部掩码都是真代码路径。只证明索引与语义正确,
不能证明真机上的指令吞吐。
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SHIM = Path(__file__).resolve().parent
SRCS = [ROOT / "sunnynet" / "kernels" / "core_avx512.c",
        ROOT / "sunnynet" / "kernels" / "step_avx512.c"]
OUT = SHIM / "avx512_shim.dll"

FL = ctypes.POINTER(ctypes.c_float)
CI = ctypes.c_int


def compile_shim(cc: str) -> Path:
    cmd = [cc, "-O2", "-shared", "-fPIC", "-std=c11", "-DSN_512_NO_COMMON",
           "-fopenmp", "-Wno-psabi",
           "-I", str(SHIM)] + [str(s) for s in SRCS] + ["-o", str(OUT)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"shim 编译失败:\n{(r.stderr or '')[:3000]}")
    return OUT


def ref_core_fwd(q, delta, s0):
    """参考: 对齐 layers.py::_compete 的组内成对竞争门控与门控概率."""
    B, T, D2 = q.shape
    D = D2 // 2
    NG = D // 8
    g = torch.from_numpy(q[..., D:].copy()).reshape(B, T, NG, 8)
    ur = torch.from_numpy(q[..., :D].copy()).reshape(B, T, NG, 8)
    dl = torch.from_numpy(delta.copy()).reshape(NG, 8)
    sig = torch.sigmoid(g)
    w = torch.sigmoid(g - g.roll(-1, dims=-1)) * torch.sigmoid(g - g.roll(-2, dims=-1))
    states = torch.empty(B, T, NG, 8)
    us = torch.empty(B, T, NG, 8)
    ps = torch.empty(B, T, NG, 8)
    for b in range(B):
        sb = torch.from_numpy(s0[b].copy()).reshape(NG, 8)
        for t in range(T):
            u = ur[b, t] * w[b, t]
            p = dl * sig[b, t] + (1.0 - dl)
            sb = p * sb + (1.0 - p) * u
            states[b, t], us[b, t], ps[b, t] = sb, u, p
    return (states.reshape(B, T, D).numpy(), us.reshape(B, T, D).numpy(),
            ps.reshape(B, T, D).numpy())


def ref_step_core(q, delta, s):
    B, D2 = q.shape
    D = D2 // 2
    NG = D // 8
    ur = q[:, :D].reshape(B, NG * 8)
    g = q[:, D:].reshape(B, NG, 8)
    sig = 1.0 / (1.0 + np.exp(-g))
    w = (1.0 / (1.0 + np.exp(-(g - np.roll(g, -1, axis=-1))))) * \
        (1.0 / (1.0 + np.exp(-(g - np.roll(g, -2, axis=-1)))))
    u = q[:, :D].reshape(B, NG, 8) * w
    p = delta.reshape(NG, 8) * sig + (1.0 - delta.reshape(NG, 8))
    sn = p * s.reshape(B, NG, 8) + (1.0 - p) * u
    return sn.reshape(B, D), u.reshape(B, D)


def ref_core_bwd(q, delta, s0, dstate, du_in):
    """参考: 用 torch autograd 对 core 的前向求导, 免手推反向公式."""
    import torch
    qq = torch.from_numpy(q.copy()).requires_grad_(True)
    dd = torch.from_numpy(delta.copy()).requires_grad_(True)
    ss = torch.from_numpy(s0.copy()).requires_grad_(True)
    u, states = ref_core_fwd_torch(qq, dd, ss)
    (u * torch.from_numpy(du_in.copy())
     + states * torch.from_numpy(dstate.copy())).sum().backward()
    return (qq.grad.numpy(), dd.grad.numpy(), ss.grad.numpy())


def ref_core_fwd_torch(q, delta, s0):
    """与 ref_core_fwd 同一式子, 但保持 torch 张量以支持 autograd."""
    import torch
    B, T, D2 = q.shape
    D = D2 // 2
    NG = D // 8
    g = q[..., D:].reshape(B, T, NG, 8)
    ur = q[..., :D].reshape(B, T, NG, 8)
    dl = delta.reshape(NG, 8)
    sig = torch.sigmoid(g)
    w = torch.sigmoid(g - g.roll(-1, dims=-1)) * torch.sigmoid(g - g.roll(-2, dims=-1))
    states = torch.empty(B, T, NG, 8, dtype=torch.float64)
    us = torch.empty(B, T, NG, 8, dtype=torch.float64)
    for b in range(B):
        sb = s0[b].reshape(NG, 8)
        for t in range(T):
            u = ur[b, t] * w[b, t]
            p = dl * sig[b, t] + (1.0 - dl)
            sb = p * sb + (1.0 - p) * u
            states[b, t], us[b, t] = sb, u
    return us.reshape(B, T, D), states.reshape(B, T, D)


def main() -> int:
    cc = sys.argv[1] if len(sys.argv) > 1 else "gcc"
    compile_shim(cc)
    lib = ctypes.CDLL(str(OUT))
    for fn, argt in (
        ("core512_fwd_all_f32", [FL, FL, FL, FL, FL, FL, CI, CI, CI]),
        ("core512_bwd_all_f32", [FL, FL, FL, FL, FL, FL, FL, FL, FL, FL, FL,
                                 CI, CI, CI]),
        ("core512_fwd_f32", [FL, FL, FL, FL, FL, FL, CI, CI, CI]),
        ("core512_fwd_tail_f32", [FL, FL, FL, FL, FL, FL, CI, CI, CI, CI]),
        ("step_core512_all_f32", [FL, FL, FL, FL, FL, CI, CI]),
        ("step_core512_f32", [FL, FL, FL, FL, FL, CI, CI]),
        ("step_core512_tail_f32", [FL, FL, FL, FL, FL, CI, CI, CI]),
        ("dw_step512_f32", [FL, FL, FL, FL, CI, CI, CI]),
    ):
        getattr(lib, fn).argtypes = argt

    rng = np.random.default_rng(2026)
    res: dict = {"compiled_from": [str(s.relative_to(ROOT)) for s in SRCS],
                 "shim": "experiments/_avx512_shim/immintrin.h",
                 "note": "只验证索引与语义; 真 AVX-512 指令吞吐未经真机验证"}
    TOL = 1e-5

    # --- core fwd: NG 偶数 (全走 16 宽主循环) 与 NG 奇数 (内部补 8 宽尾部) ---
    for D in (32, 24):
        B, T = 2, 5
        q = rng.standard_normal((B, T, 2 * D)).astype(np.float32)
        delta = rng.random((D,), dtype=np.float32)
        s0 = rng.standard_normal((B, D)).astype(np.float32)
        st = np.zeros((B, T, D), np.float32)
        uo = np.zeros((B, T, D), np.float32)
        po = np.zeros((B, T, D), np.float32)
        NG = D // 8
        lib.core512_fwd_all_f32(q.ctypes.data_as(FL), delta.ctypes.data_as(FL),
                                s0.ctypes.data_as(FL), st.ctypes.data_as(FL),
                                uo.ctypes.data_as(FL), po.ctypes.data_as(FL), B, T, D)
        r_st, r_uo, r_po = ref_core_fwd(q, delta, s0)
        res[f"core_fwd_D{D}"] = {
            "ng": NG, "tail_used": bool(NG % 2),
            "maxdiff_states": float(np.abs(st - r_st).max()),
            "maxdiff_u": float(np.abs(uo - r_uo).max()),
            "maxdiff_p": float(np.abs(po - r_po).max())}

    # --- core bwd: 与 torch autograd 对拍, 含 NG 奇数的尾部 group ---
    for D in (32, 24):
        B, T = 2, 5
        q = rng.standard_normal((B, T, 2 * D)).astype(np.float32)
        delta = rng.random((D,), dtype=np.float32)
        s0 = rng.standard_normal((B, D)).astype(np.float32)
        st, uo, po = ref_core_fwd(q, delta, s0)
        dst = rng.standard_normal((B, T, D)).astype(np.float32)
        du_in = rng.standard_normal((B, T, D)).astype(np.float32)
        dq = np.zeros((B, T, 2 * D), np.float32)
        ddelta = np.zeros((D,), np.float32)
        ds0 = np.zeros((B, D), np.float32)
        lib.core512_bwd_all_f32(
            q.ctypes.data_as(FL), delta.ctypes.data_as(FL), s0.ctypes.data_as(FL),
            st.astype(np.float32).ctypes.data_as(FL), uo.astype(np.float32).ctypes.data_as(FL),
            po.astype(np.float32).ctypes.data_as(FL),
            dst.ctypes.data_as(FL), du_in.ctypes.data_as(FL),
            dq.ctypes.data_as(FL), ddelta.ctypes.data_as(FL), ds0.ctypes.data_as(FL),
            B, T, D)
        r_dq, r_ddelta, r_ds0 = ref_core_bwd(q, delta, s0, dst, du_in)
        res[f"core_bwd_D{D}"] = {
            "ng": D // 8, "tail_used": bool((D // 8) % 2),
            "maxdiff_dq": float(np.abs(dq - r_dq).max()),
            "maxdiff_ddelta": float(np.abs(ddelta - r_ddelta).max()),
            "maxdiff_ds0": float(np.abs(ds0 - r_ds0).max())}

    # --- step core ---
    for D in (32, 24):
        B = 2
        q = rng.standard_normal((B, 2 * D)).astype(np.float32)
        delta = rng.random((D,), dtype=np.float32)
        s = rng.standard_normal((B, D)).astype(np.float32)
        sn = np.zeros((B, D), np.float32)
        us = np.zeros((B, D), np.float32)
        NG = D // 8
        lib.step_core512_all_f32(q.ctypes.data_as(FL), delta.ctypes.data_as(FL),
                                 s.ctypes.data_as(FL), sn.ctypes.data_as(FL),
                                 us.ctypes.data_as(FL), B, D)
        r_sn, r_us = ref_step_core(q, delta, s)
        res[f"step_core_D{D}"] = {
            "ng": NG, "tail_used": bool(NG % 2),
            "maxdiff_states": float(np.abs(sn - r_sn).max()),
            "maxdiff_u": float(np.abs(us - r_us).max())}

    # --- dw_step ---
    B, D, taps = 2, 40, 3
    x = rng.standard_normal((B, D)).astype(np.float32)
    hist = rng.standard_normal((B, taps - 1, D)).astype(np.float32)
    w = rng.standard_normal((taps, D)).astype(np.float32)
    acc = np.zeros((B, D), np.float32)
    lib.dw_step512_f32(x.ctypes.data_as(FL), hist.ctypes.data_as(FL),
                       w.ctypes.data_as(FL), acc.ctypes.data_as(FL), B, D, taps)
    r_acc = x * w[taps - 1]
    for j in range(taps - 1):
        r_acc = r_acc + hist[:, j, :] * w[taps - 2 - j]
    res["dw_step"] = {"maxdiff": float(np.abs(acc - r_acc).max())}

    print(json.dumps(res, ensure_ascii=False, indent=2))
    ok = all(v < TOL for d in res.values() if isinstance(d, dict)
             for k, v in d.items() if k.startswith("maxdiff"))
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())