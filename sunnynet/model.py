"""SunnyNet 模型主体 (一阶线性递推序列架构).

结构: 嵌入 -> N 个 SunnyBlock (滚动状态逐层接力 + 特征残差流) -> 终态归一 -> 绑定 LM Head.

跨层信息传递: 第 l 块的末态 s_l 直接作为第 l+1 块的初始状态 (状态接力);
块输出 o 与输入 x 残差相加后作为下一块输入 (特征残差流)。
尺度由 CenteredScaleNorm 与谱锚定初始化共同锁定。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .init import spectral_anchored_init
from .layers import SunnyBlock, SunnyState
from .ops import CenteredScaleNorm

__all__ = ["SunnyConfig", "SunnyNet", "SunnyState"]


@dataclass
class SunnyConfig:
    vocab_size: int = 12000
    dim: int = 256
    inner: int = 384
    rank: int = 128
    taps: int = 4
    layers: int = 6
    scan_chunk: int = 64
    rho0: float = 1.0
    decay_beta: float = 0.0
    h_min: float = 2.0
    h_max: float = 8192.0
    tie_embeddings: bool = True
    ablation: dict = field(default_factory=dict)

    @staticmethod
    def small(vocab_size: int) -> "SunnyConfig":
        return SunnyConfig(vocab_size=vocab_size, dim=128, inner=192, rank=64,
                           layers=4, h_max=2048.0)

    @staticmethod
    def deepl6(vocab_size: int) -> "SunnyConfig":
        """深瘦阶梯: 与 small 近似同栈参数, 只把层数 4->6、dim 128->96.
        用于隔离"深度"对层间公共分量累积的影响."""
        return SunnyConfig(vocab_size=vocab_size, dim=96, inner=152, rank=128,
                           layers=6, h_max=4096.0)

    @staticmethod
    def deepl8(vocab_size: int) -> "SunnyConfig":
        """更深: 层数 8、dim 80."""
        return SunnyConfig(vocab_size=vocab_size, dim=80, inner=128, rank=160,
                           layers=8, h_max=4096.0)

    @staticmethod
    def proto20m(vocab_size: int) -> "SunnyConfig":
        return SunnyConfig(vocab_size=vocab_size, dim=512, inner=896, rank=320,
                           layers=10, h_max=8192.0)

    @staticmethod
    def scale05b(vocab_size: int) -> "SunnyConfig":
        return SunnyConfig(vocab_size=vocab_size, dim=2048, inner=3584, rank=1280,
                           layers=18, h_max=32768.0)


class SunnyNet(nn.Module):
    def __init__(self, cfg: SunnyConfig):
        super().__init__()
        self.cfg = cfg
        ab = cfg.ablation
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([
            SunnyBlock(cfg.dim, cfg.inner, rank=cfg.rank, taps=cfg.taps,
                       layer_index=i, rho0=cfg.rho0, decay_beta=cfg.decay_beta,
                       h_min=cfg.h_min, h_max=cfg.h_max,
                       scan_chunk=cfg.scan_chunk, ablation=ab)
            for i in range(cfg.layers)
        ])
        self.norm_out = CenteredScaleNorm(cfg.dim)
        # 残差流逐层累积跨输入共享的 DC (L8/d1408 实测特异能量 22.6%->3.5%).
        # 曾试过在残差流上做归一化去 DC, 实测反而把特异分量压到 0
        # (ppl 263->625, 24 输入塌成单一 argmax) —— 残差流不可清洗, 保留作证伪对照.
        self.h_norm = CenteredScaleNorm(cfg.dim) if ab.get("h_norm", False) else None
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.head.weight.numel()
        return n

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> SunnyState:
        k = max(0, self.cfg.taps - 1)
        mem = None
        for blk in self.blocks:
            if blk.assoc is not None:
                mem = torch.zeros(batch, self.cfg.layers, blk.assoc.slots,
                                  blk.assoc.m, device=device, dtype=dtype)
                break
        return SunnyState(
            s=torch.zeros(batch, self.cfg.layers, self.cfg.dim, device=device, dtype=dtype),
            hist=torch.zeros(batch, self.cfg.layers, k, self.cfg.dim,
                             device=device, dtype=dtype),
            mem=mem)

    def forward(self, idx: torch.Tensor, state: SunnyState | None = None,
                return_state: bool = False):
        """idx (B,T) -> logits (B,T,V); state 为 SunnyState 或 None."""
        B, T = idx.shape
        x = self.embed(idx)
        k = max(0, self.cfg.taps - 1)
        use_mem = any(blk.assoc is not None for blk in self.blocks)
        if state is None:
            s = torch.zeros(B, self.cfg.layers, self.cfg.dim, device=idx.device, dtype=x.dtype)
            hist = torch.zeros(B, self.cfg.layers, k, self.cfg.dim,
                               device=idx.device, dtype=x.dtype)
            mem = None
            if use_mem:
                b0 = next(blk for blk in self.blocks if blk.assoc is not None)
                s0, m0 = b0.assoc.slots, b0.assoc.m
                mem = torch.zeros(B, self.cfg.layers, s0, m0,
                                  device=idx.device, dtype=x.dtype)
        else:
            s, hist, mem = state.s, state.hist, state.mem
        ns = torch.empty_like(s) if return_state else None
        nh = torch.empty_like(hist) if return_state else None
        nm = torch.empty_like(mem) if (return_state and use_mem) else None
        for i, blk in enumerate(self.blocks):
            if blk.assoc is not None:
                x, s_i, h_i, m_i = blk(x, s[:, i], hist[:, i], mem[:, i])
                if return_state:
                    nm[:, i] = m_i
            else:
                x, s_i, h_i = blk(x, s[:, i], hist[:, i])
            if self.h_norm is not None:
                x = self.h_norm(x)
            if return_state:
                ns[:, i] = s_i
                nh[:, i] = h_i
        logits = self.head(self.norm_out(x))
        if return_state:
            return logits, SunnyState(ns, nh, nm)
        return logits

    @torch.no_grad()
    def forward_step(self, idx_t: torch.Tensor, state: SunnyState):
        """单步自回归: idx_t (B,) -> (logits (B,V), SunnyState)."""
        x = self.embed(idx_t)
        ns = torch.empty_like(state.s)
        nh = torch.empty_like(state.hist)
        use_mem = any(blk.assoc is not None for blk in self.blocks)
        nm = torch.empty_like(state.mem) if use_mem else None
        for i, blk in enumerate(self.blocks):
            if blk.assoc is not None:
                x, s_i, h_i, m_i = blk.step(x, state.s[:, i], state.hist[:, i],
                                            state.mem[:, i])
                nm[:, i] = m_i
            else:
                x, s_i, h_i = blk.step(x, state.s[:, i], state.hist[:, i])
            if self.h_norm is not None:
                x = self.h_norm(x)
            ns[:, i] = s_i
            nh[:, i] = h_i
        logits = self.head(self.norm_out(x))
        return logits, SunnyState(ns, nh, nm)

    def state_dict_for_inference(self, batch: int, device=None, dtype=torch.float32):
        return self.init_state(batch, device, dtype)