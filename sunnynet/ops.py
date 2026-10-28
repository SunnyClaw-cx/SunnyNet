"""SunnyNet 原创基础算子. 推导见 docs/cn/ARCHITECTURE.md."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "poly_sigmoid",
    "softwedge",
    "CenteredScaleNorm",
    "multiscale_decay",
    "horizon_logit",
    "logspace_scan",
    "gated_scan",
    "chunk_delta_scan",
]


def poly_sigmoid(z: torch.Tensor) -> torch.Tensor:
    """s(z) = 0.5 + 0.5 * z / (1 + |z|),  值域 (0,1),  s'(z) = 0.5/(1+|z|)^2."""
    return 0.5 + 0.5 * (z / (1.0 + z.abs()))


def softwedge(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """f(z) = z + a * s(z),  a in (0,1).  f'(z) in (1, 1+a/2], 恒 > 1."""
    return z + a * poly_sigmoid(z)


class CenteredScaleNorm(nn.Module):
    """y = gamma * ln(x) + c,  ln 的分母为 sqrt(var + s_min^2 + eps).

    s_min 必须低于实际激活尺度 (small 预设低至 0.02), 否则归一化退化为常数缩放.
    forward 走 oneDNN 融合核 (mean/var/sub/mul/add 单 kernel), 手写链实测慢 19x.
    """

    def __init__(self, dim: int, eps: float = 1e-5, s_min: float = 0.02):
        super().__init__()
        self.eps = eps
        self.s_min = s_min
        self.gamma = nn.Parameter(torch.ones(dim))
        self.center = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.gamma, self.center,
                            self.s_min * self.s_min + self.eps)


def multiscale_decay(n: int, h_min: float = 2.0, h_max: float = 8192.0,
                     device=None, dtype=torch.float32) -> torch.Tensor:
    """逐通道遗忘上界 delta_i = 1/H_i, H_i 在 [h_min, h_max] 对数均匀."""
    if n == 1:
        h = torch.tensor([h_max], device=device, dtype=dtype)
    else:
        lo, hi = math.log(h_min), math.log(h_max)
        t = torch.arange(n, device=device, dtype=dtype) / (n - 1.0)
        h = torch.exp(lo + (hi - lo) * t)
    return 1.0 / h


def horizon_logit(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """theta 初值: 使 H = h_min*(h_max/h_min)^sigmoid(theta) 对数均匀, 取通道中点避开 ±inf."""
    s = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    return torch.log(s) - torch.log1p(-s)


def gated_scan(p: torch.Tensor, g: torch.Tensor, s0: torch.Tensor,
               chunk: int = 64, p_min_hint: float | None = None):
    """一阶线性递推 s_t = p_t*s_{t-1} + g_t 的闭式解, 支持任意尾随状态维.

    两级: 块内一次 cumsum 覆盖所有块 (并置), 块间串行 nblk 步.
    块 c 的块内前缀 P, A 由 s_in^c 修正: S^c = exp(P)*(s_in^c + A), s_in^{c+1} = S^c 末行.
    与逐块串行等价 (fp32/fp64 实测 bit-identical).

    p, g: (..., T, N);  s0: (..., N) -> (states (..., T, N), s_T (..., N))
    """
    p, g = torch.broadcast_tensors(p, g)
    T = p.shape[-2]
    p_min = p.detach().amin().item() if p_min_hint is None else p_min_hint
    chunk = _safe_chunk(p_min, chunk)
    nblk = (T + chunk - 1) // chunk
    pad = nblk * chunk - T
    lead, N = p.shape[:-2], p.shape[-1]
    if pad:
        p = torch.cat([p, p.new_ones(*lead, pad, N)], dim=-2)
        g = torch.cat([g, g.new_zeros(*lead, pad, N)], dim=-2)
    lp = torch.log(p).reshape(*lead, nblk, chunk, N)
    gg = g.reshape(*lead, nblk, chunk, N)
    P = torch.cumsum(lp, dim=-2)
    E = torch.exp(P)
    A = torch.cumsum(gg * torch.exp(-P), dim=-2)
    Elast = E[..., :, -1, :]
    Alast = A[..., :, -1, :]
    s = s0
    s_in = []
    for c in range(nblk):
        s_in.append(s)
        s = Elast[..., c, :] * (s + Alast[..., c, :])
    sin = torch.stack(s_in, dim=-2)
    S = (E * (sin.unsqueeze(-2) + A)).reshape(*lead, nblk * chunk, N)
    return S[..., :T, :], s


def logspace_scan(nu: torch.Tensor, p: torch.Tensor, s0: torch.Tensor,
                  chunk: int = 64, p_min_hint: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """一阶线性递推 s_t = p_t*s_{t-1} + (1-p_t)*nu_t, 即 gated_scan 取 g=(1-p)*nu.

    p_min_hint 给定时直接用其限制块长, 跳过 p.amin() 的同步开销; 取值不变.
    nu, p: (B,T,N);  s0: (B,N) -> (states (B,T,N), s_T (B,N))
    """
    return gated_scan(p, (1.0 - p) * nu, s0, chunk=chunk, p_min_hint=p_min_hint)


def chunk_delta_scan(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                     s0: torch.Tensor, decay: torch.Tensor | None = None,
                     chunk: int = 64, p_min: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """delta 写入递推的分块 (WY) 形式, 支持标量衰减.

        状态:  S_t = d_t * S_{t-1} - β_t k_t (k_tᵀ S_{t-1}) + β_t k_t v_tᵀ

    d_t 为逐位置标量衰减 (None 视为 1). 经 T_t = S_t / σ_t (σ_t = ∏_{i≤t} d_i) 变换后
    化为纯 delta: T_t = T_{t-1} - β̃_t k_t(k_tᵀ T_{t-1}) + c_t k_t v_tᵀ,
    β̃_t = β_t/d_t, c_t = β_t/σ_t. 块内 S_t = S_in + K_{0:t}ᵀ W_{0:t},
    W = (I+L)⁻¹ diag(β̃)(V/d - K S_in), L_{ij}=β̃_i(k_i·k_j) 取下三角 j<i.

    k: (..., T, S) 单位化;  v: (..., T, M);  beta: (..., T);  s0: (..., S, M);
    decay: (..., T) 或 None;  p_min: 衰减最小值, 用于按条件数限制块长
    -> (states (..., T, S, M), s_T (..., S, M))
    """
    lead = k.shape[:-2]
    T, S = k.shape[-2], k.shape[-1]
    M = v.shape[-1]
    chunk = _safe_chunk_delta(chunk, p_min=p_min)
    nblk = (T + chunk - 1) // chunk
    pad = nblk * chunk - T
    if pad:
        k = torch.cat([k, k.new_zeros(*lead, pad, S)], dim=-2)
        v = torch.cat([v, v.new_zeros(*lead, pad, M)], dim=-2)
        beta = torch.cat([beta, beta.new_zeros(*lead, pad)], dim=-1)
    if decay is None:
        d = k.new_ones(*lead, nblk * chunk)
    else:
        d = torch.cat([decay, decay.new_ones(*lead, pad)], dim=-1) if pad else decay
    kk = k.reshape(*lead, nblk, chunk, S)
    vv = v.reshape(*lead, nblk, chunk, M)
    bt = beta.reshape(*lead, nblk, chunk)
    dd = d.reshape(*lead, nblk, chunk)
    Dt = torch.cumprod(dd, dim=-1)                           # 块内 D_t = ∏_{i≤t} d_i
    # 块内几何 (与块首无关)
    G = torch.einsum("...is,...js->...ij", kk, kk)           # (..., nblk, chunk, chunk)
    L = torch.tril(G * bt.unsqueeze(-1), diagonal=-1)        # L_ij = β_i (k_i·k_j), j<i
    eye = torch.eye(chunk, dtype=k.dtype, device=k.device)
    Amat = eye + L
    # 块间: 块首状态直接作为块内 H_{-1} (D_{-1}=1), 串行推进
    s = s0
    outs = []
    for c in range(nblk):
        kc, vcc, btc = kk[..., c, :, :], vv[..., c, :, :], bt[..., c, :]
        # H 空间块内递推: H_t = H_{t-1} - β_t k_t(k_tᵀ H_{t-1}) + (β_t/D_t) k_t v_tᵀ
        inj = vcc / Dt[..., c, :].unsqueeze(-1)              # v_t / D_t
        rhs = btc.unsqueeze(-1) * (inj - torch.einsum("...is,...sm->...im", kc, s))
        W = _tri_solve(Amat[..., c, :, :], rhs)
        inc = kc.unsqueeze(-1) * W.unsqueeze(-2)               # k_t ⊗ w_t, (..., chunk, S, M)
        Ht = s.unsqueeze(-3) + torch.cumsum(inc, dim=-3)       # H_t = H_in + Σ_{i≤t} k_i w_iᵀ
        St = Dt[..., c, :].unsqueeze(-1).unsqueeze(-1) * Ht  # 回到 S 空间
        outs.append(St)
        s = St[..., -1, :, :]
    S_all = torch.cat(outs, dim=-3).reshape(*lead, nblk * chunk, S, M)
    return S_all[..., :T, :, :], s


def _tri_solve(A: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """解 A x = rhs, A 单位下三角 (..., chunk, chunk), rhs (..., chunk, M)."""
    lead = A.shape[:-2]
    n, M = A.shape[-1], rhs.shape[-1]
    Af = A.reshape(-1, n, n)
    rf = rhs.reshape(-1, n, M)
    x = torch.linalg.solve_triangular(Af, rf, upper=False)
    return x.reshape(*lead, n, M)


def _safe_chunk(p_min: float, requested: int, headroom_bits: int = 64) -> int:
    """按 p 最小值限制块长, 使 exp(-P) <= 2^headroom_bits. 取值与块长无关.

    p_min 非有限 (上游发散) 时无法约束, 返回保守块长; NaN 与 0/1 比较恒 False,
    不先判 isfinite 会落到 int(nan).
    """
    if not math.isfinite(p_min) or p_min <= 0 or p_min >= 1:
        return min(requested, 64)
    per_bit = -math.log(p_min)
    if per_bit <= 0:
        return requested
    return max(1, min(requested, int(headroom_bits * math.log(2.0) / per_bit)))


def _safe_chunk_delta(requested: int, cap: int = 32, p_min: float | None = None,
                      headroom_bits: int = 50) -> int:
    """delta 分块的块长上限.

    块内算 1/∏_{i≤t} d_i ~ d_min^-L; 超出安全水位后反向给衰减产出非有限梯度
    (实测临界在 2^57~2^62 之间, 随 p 与块数浮动). 按 headroom_bits 反解 L 并
    与 cap 取小. p_min 由构造期解析下界给出, 生产路径恒 >= 0.5 故 cap 生效.
    """
    if p_min is not None and math.isfinite(p_min) and 0.0 < p_min < 1.0:
        per_step = -math.log(p_min)
        if per_step > 0:
            by_cond = int(headroom_bits * math.log(2.0) / per_step)
            return max(1, min(requested, cap, max(1, by_cond)))
    return max(1, min(requested, cap))