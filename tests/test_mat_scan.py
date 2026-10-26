"""矩阵值 rank-1 递推内核 (sn_mat_scan) 的数值契约.

对应 RWKV-7 的矩阵状态层级: S += v x k 这类外积累加无法用元素级
递推表达。内核按 head 串行、沿 C 向量化, 参数传 NULL 即跳过该项。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(
    not __import__("sunnynet.runtime", fromlist=["x"]).runtime_available(),
    reason="无 C 编译器且无预编译库")


def _ref(S, w=None, u=None, v=None, k=None, b=None, r=None):
    S = S.copy()
    p = np.einsum("hrc,hc->hr", S, u) if u is not None else None
    NS = S * w[:, None, :] if w is not None else S
    if k is not None:
        add = np.einsum("hr,hc->hrc", v, k)
        if b is not None:
            add = add - np.einsum("hr,hc->hrc", p, b)
        NS = NS + add
    y = np.einsum("hrc,hc->hr", NS, r) if r is not None else None
    return NS, y


def _rnd(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


def _case(NH, R, C, seed=0, **kw):
    from sunnynet.runtime import mat_step
    rng = np.random.default_rng(seed)
    S = torch.tensor(_rnd(rng, NH, R, C))
    t = {k: (torch.tensor(v) if v is not None else None) for k, v in kw.items()}
    refS, refY = _ref(S.numpy().copy(), **kw)
    y = mat_step(S, **t)
    relS = float(np.abs(S.numpy() - refS).max()) / max(1.0, float(np.abs(refS).max()))
    if y is None:
        assert refY is None
        return relS, 0.0
    relY = float(np.abs(y.numpy() - refY).max()) / max(1.0, float(np.abs(refY).max()))
    return relS, relY


def test_rwkv7_full_form():
    """w/u/v/k/b/r 全给, 对应 RWKV-7 的衰减 + rank-1 校正 + 读出。"""
    rng = np.random.default_rng(1)
    dS, dY = _case(
        3, 16, 24, seed=1,
        w=(rng.random((3, 24)) * 0.95 + 0.02).astype(np.float32),
        u=_rnd(rng, 3, 24), v=_rnd(rng, 3, 16),
        k=_rnd(rng, 3, 24), b=_rnd(rng, 3, 24), r=_rnd(rng, 3, 24))
    assert dS < 1e-5 and dY < 1e-5


def test_no_decay():
    """w=None: 无衰减门的形式。"""
    rng = np.random.default_rng(2)
    dS, dY = _case(
        3, 16, 24, seed=2, w=None,
        u=_rnd(rng, 3, 24), v=_rnd(rng, 3, 16),
        k=_rnd(rng, 3, 24), b=_rnd(rng, 3, 24), r=_rnd(rng, 3, 24))
    assert dS < 1e-5 and dY < 1e-5


def test_only_outer_product():
    """只给 v/k: 纯外积累加。"""
    rng = np.random.default_rng(3)
    dS, dY = _case(3, 16, 24, seed=3,
                   k=_rnd(rng, 3, 24), v=_rnd(rng, 3, 16), r=_rnd(rng, 3, 24))
    assert dS < 1e-5 and dY < 1e-5


def test_only_decay():
    rng = np.random.default_rng(4)
    dS, dY = _case(3, 16, 24, seed=4,
                   w=(rng.random((3, 24)) * 0.9 + 0.05).astype(np.float32),
                   r=_rnd(rng, 3, 24))
    assert dS < 1e-5 and dY < 1e-5


def test_no_readout():
    """r=None: 不计算 y, 只更新状态。"""
    rng = np.random.default_rng(5)
    dS, dY = _case(
        3, 16, 24, seed=5,
        w=(rng.random((3, 24)) * 0.9 + 0.05).astype(np.float32),
        u=_rnd(rng, 3, 24), v=_rnd(rng, 3, 16),
        k=_rnd(rng, 3, 24), b=_rnd(rng, 3, 24))
    assert dS < 1e-5 and dY == 0.0


@pytest.mark.parametrize("C", [7, 20, 24, 33])
def test_ragged_width_tail(C):
    """C 非 8 倍数时尾部标量路径必须与向量路径一致。"""
    rng = np.random.default_rng(10 + C)
    dS, dY = _case(
        2, 8, C, seed=10 + C,
        w=(rng.random((2, C)) * 0.9 + 0.05).astype(np.float32),
        u=_rnd(rng, 2, C), v=_rnd(rng, 2, 8),
        k=_rnd(rng, 2, C), b=_rnd(rng, 2, C), r=_rnd(rng, 2, C))
    assert dS < 1e-5 and dY < 1e-5


def test_repeated_calls_accumulate():
    """连续多步: 状态逐次累加, 与参考逐步同步。"""
    from sunnynet.runtime import mat_step
    rng = np.random.default_rng(7)
    NH, R, C = 2, 8, 16
    S = torch.zeros(NH, R, C)
    Sn = np.zeros((NH, R, C), dtype=np.float32)
    for step in range(5):
        kw = dict(w=(rng.random((NH, C)) * 0.9 + 0.05).astype(np.float32),
                  u=_rnd(rng, NH, C), v=_rnd(rng, NH, R),
                  k=_rnd(rng, NH, C), b=_rnd(rng, NH, C), r=_rnd(rng, NH, C))
        Sn, _ = _ref(Sn, **kw)
        mat_step(S, **{k: torch.tensor(v) for k, v in kw.items()})
    assert float(np.abs(S.numpy() - Sn).max()) / max(1.0, float(np.abs(Sn).max())) < 1e-5


def test_rejects_non_contiguous():
    from sunnynet.runtime import mat_step
    S = torch.zeros(2, 8, 16).transpose(0, 1)
    with pytest.raises(ValueError):
        mat_step(S, k=torch.zeros(2, 16))