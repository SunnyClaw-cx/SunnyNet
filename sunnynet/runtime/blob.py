"""GGUF -> C 内核 blob 的包内导出与缓存.

C 内核 `sn_rk_load` 读的是按固定张量顺序排列的 blob, 不直接读 GGUF.
这里把 GGUF 反量化后按该顺序落盘, 并按 mtime 缓存, 使 `--gguf` 一行即可走 C 路径.

blob 布局 (前置 10 个 int32 + 1 个 float + L*4 个 int32 秩表, 其后为带标志张量流):
  magic, version=3, layers, dim, head_size, n_head, ff, g_rank, vocab, q4
  eps(f32); ranks: L*(Rw,Ra,Rv,Rg)
  每张量前置 int32 标志 (1=INT4 载荷 + 每行 f32 scale, 0=fp32 原始字节)
  矩阵存 M^T (行主序 (len(x), len(y))); 嵌入表保持 fp32 (查表非乘加)
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .gguf import GGUFModel

MAGIC = 0x4B57524B
VERSION = 3
BLOCK = 64

_LORA = ("time_mix_w1", "time_mix_w2", "time_mix_a1", "time_mix_a2",
         "time_mix_g1", "time_mix_g2", "time_mix_v1", "time_mix_v2")


def blob_path(gguf: str | Path, quant: str = "int4") -> Path:
    p = Path(gguf)
    return p.with_suffix(".q4.bin" if quant == "int4" else ".runtime.bin")


# general.architecture -> 本包已实现的架构名。
GGUF_ARCHES = {
    "rwkv7": "rwkv7", "rwkv": "rwkv7",
    "mamba2": "mamba2", "mamba": "mamba2",
}


def gguf_arch(gguf: str | Path) -> str:
    """读 general.architecture 并归一到本包的架构名 (未识别的原样返回)."""
    g = GGUFModel(gguf)
    raw = str(g.meta.get("general.architecture", "") or "").strip().lower()
    return GGUF_ARCHES.get(raw, raw)


def require_rwkv7_gguf(gguf: str | Path) -> None:
    """GGUF -> C blob 目前只覆盖 RWKV-7, 喂别的架构要在入口就报清楚."""
    arch = gguf_arch(gguf)
    if arch == "rwkv7":
        return
    hint = (f"该 GGUF 的架构是 {arch!r}, 但 GGUF 直出 C 内核 blob 目前只支持 RWKV-7。\n"
            f"其余架构请用 HF 权重目录:\n"
            f"  sunnynet chat --hf-dir <权重目录> --hf-arch mamba2 --blob <架构>.c.bin")
    raise SystemExit(hint)


# 整模型 C 内核 blob 的头部协议, 由各架构的 .c 定义, 两处必须同步改
# Mamba-2: <8i + f 后是 3 个 int (vocab, conv, tied) -> vocab 在 int 索引 9
_HF_BLOB_LAYOUT = {
    0x324E4D53: ("<8if3i", 48, 9),   # Mamba-2
}


def read_blob_vocab(path: str | Path) -> int:
    """从整模型 C 内核 blob 头部读 vocab, 免为拿一个数加载整份权重."""
    with open(path, "rb") as f:
        head = f.read(max(n for _, n, _ in _HF_BLOB_LAYOUT.values()))
    if len(head) < 4:
        raise ValueError(f"blob 头部不完整: {path}")
    magic = struct.unpack("<i", head[:4])[0]
    fmt = _HF_BLOB_LAYOUT.get(magic)
    if fmt is None:
        raise ValueError(f"不是 C 内核 blob (magic {magic:#x}): {path}")
    fmt, need, idx = fmt
    if len(head) < need:
        raise ValueError(f"blob 头部不完整 ({len(head)} < {need}): {path}")
    return int(struct.unpack(fmt, head[:need])[idx])


def _fresh(blob: Path, gguf: Path) -> bool:
    return blob.exists() and blob.stat().st_mtime >= gguf.stat().st_mtime


def write_rwkv7_blob(gguf: str | Path, dst: str | Path,
                     int4: bool = True) -> dict:
    """GGUF -> RWKV-7 C 内核 blob. 返回统计信息."""
    g = GGUFModel(gguf)
    L = g.get_int("rwkv7.block_count", 12)
    D = g.get_int("rwkv7.embedding_length", 768)
    HS = g.get_int("rwkv7.wkv.head_size", 64)
    NH = D // HS
    F = g.get_int("rwkv7.feed_forward_length", 3072)
    vocab = len(g.tokens())
    eps = g.get_float("rwkv7.attention.layer_norm_epsilon", 1e-5)

    def pth(name: str) -> np.ndarray:
        a = g.tensor(name)
        # GGUF 的 LoRA 分解矩阵方向与官方 .pth 相反, 此处翻回
        if name.split(".")[-2] in _LORA:
            return np.ascontiguousarray(a.T, dtype=np.float32)
        return np.ascontiguousarray(a, dtype=np.float32)

    def t_of(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(a, dtype=np.float32).T)

    ranks = []
    for i in range(L):
        p = f"blk.{i}."
        ranks.append((pth(p + "time_mix_w1.weight").shape[1],
                      pth(p + "time_mix_a1.weight").shape[1],
                      pth(p + "time_mix_v1.weight").shape[1],
                      pth(p + "time_mix_g1.weight").shape[1]))

    parts: list[bytes] = [
        struct.pack("<10i", MAGIC, VERSION, L, D, HS, NH, F,
                    max(r[3] for r in ranks), vocab, 1 if int4 else 0),
        struct.pack("<f", eps),
    ]
    for rw, ra, rv, rg in ranks:
        parts.append(struct.pack("<4i", rw, ra, rv, rg))

    from . import quant as Q
    stats = {"int4_mats": 0, "f32_mats": 0}

    def put(a: np.ndarray, quantize: bool = False):
        a = np.ascontiguousarray(a, dtype=np.float32)
        ok = (quantize and int4 and a.ndim == 2
              and a.shape[0] % BLOCK == 0 and a.shape[1] > 0)
        if ok:
            q, s = Q.quant_i4(np.ascontiguousarray(a.T), BLOCK)
            parts.append(struct.pack("<i", 1))
            parts.append(q.tobytes() + s.astype(np.float32).tobytes())
            stats["int4_mats"] += 1
        else:
            parts.append(struct.pack("<i", 0))
            parts.append(a.reshape(-1).tobytes())
            stats["f32_mats"] += 1

    for i in range(L):
        p = f"blk.{i}."
        put(g.tensor(p + "attn_norm.weight"))
        put(g.tensor(p + "attn_norm.bias"))
        put(g.tensor(p + "attn_norm_2.weight"))
        put(g.tensor(p + "attn_norm_2.bias"))
        put(g.tensor(p + "time_mix_lerp_fused.weight").reshape(6, D))
        put(t_of(pth(p + "time_mix_receptance.weight")), True)
        put(t_of(pth(p + "time_mix_key.weight")), True)
        put(t_of(pth(p + "time_mix_value.weight")), True)
        put(t_of(pth(p + "time_mix_output.weight")), True)
        put(pth(p + "time_mix_w1.weight"), True)
        put(pth(p + "time_mix_w2.weight"), True)
        put(pth(p + "time_mix_a1.weight"), True)
        put(pth(p + "time_mix_a2.weight"), True)
        put(pth(p + "time_mix_v1.weight"), True)
        put(pth(p + "time_mix_v2.weight"), True)
        put(pth(p + "time_mix_g1.weight"), True)
        put(pth(p + "time_mix_g2.weight"), True)
        put(g.tensor(p + "time_mix_w0.weight"))
        put(g.tensor(p + "time_mix_a0.weight"))
        put(g.tensor(p + "time_mix_v0.weight"))
        put(g.tensor(p + "time_mix_k_k.weight"))
        put(g.tensor(p + "time_mix_k_a.weight"))
        put(g.tensor(p + "time_mix_r_k.weight"))
        put(g.tensor(p + "time_mix_ln.weight"))
        put(g.tensor(p + "time_mix_ln.bias"))
        put(t_of(pth(p + "channel_mix_key.weight")), True)
        put(t_of(pth(p + "channel_mix_value.weight")), True)
        put(g.tensor(p + "channel_mix_lerp_k.weight"))

    put(g.tensor("token_embd_norm.weight"))
    put(g.tensor("token_embd_norm.bias"))
    put(g.tensor("output_norm.weight"))
    put(g.tensor("output_norm.bias"))
    put(g.tensor("token_embd.weight"))
    head = "output.weight" if g.has("output.weight") else "token_embd.weight"
    put(t_of(g.tensor(head)), True)

    buf = b"".join(parts)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(buf)
    return {"file": str(dst), "bytes": len(buf), "mb": round(len(buf) / 2**20, 1),
            "layers": L, "dim": D, "vocab": vocab, "int4": bool(int4), **stats}


def ensure_rwkv7_blob(gguf: str | Path, quant: str = "int4",
                      rebuild: bool = False) -> Path:
    """返回可用的 C 内核 blob, 缺失或过期时按需导出."""
    gguf = Path(gguf)
    require_rwkv7_gguf(gguf)
    blob = blob_path(gguf, quant)
    if rebuild or not _fresh(blob, gguf):
        write_rwkv7_blob(gguf, blob, int4=(quant == "int4"))
    return blob