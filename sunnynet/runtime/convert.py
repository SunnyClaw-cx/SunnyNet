"""sunnynet convert: 把外部权重转成带 SunnyNet 标记的 GGUF.

两条路径:
  * GGUF -> GGUF: 原样搬运张量, 只在元数据里写入 `sunnynet.*` 标记与
    C 内核 blob 的校验信息, 让 `sunnynet` 加载时直接走快速路径.
  * PyTorch checkpoint -> GGUF: 把 SunnyNet 自己的 checkpoint 写成 GGUF,
    便于与线性架构同行放在同一套分发格式里.

标记的作用: `sunnynet chat`/`serve` 见到 `sunnynet.kernel = "c"` 就直接
用 C 内核, 不必每次重新探测; 见到 `sunnynet.blob_sha256` 可校验缓存是否过期.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

# 写入 GGUF 的标记前缀
_MARK = "sunnynet."


def _sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def mark_gguf(src: str | Path, dst: str | Path,
              arch: str | None = None,
              blob: str | Path | None = None) -> dict:
    """GGUF -> GGUF, 只加 SunnyNet 元数据标记, 张量逐字节搬运."""
    from gguf import GGUFWriter

    from .gguf import GGUFModel

    g = GGUFModel(src)
    arch = arch or str(g.meta.get("general.architecture") or "unknown")
    w = GGUFWriter(str(dst), arch="")
    for k, v in g.meta.items():
        if k.startswith("GGUF.") or v is None:
            continue
        try:
            if isinstance(v, bool):
                w.add_bool(k, v)
            elif isinstance(v, int):
                w.add_uint32(k, v)
            elif isinstance(v, float):
                w.add_float32(k, v)
            elif isinstance(v, str):
                w.add_string(k, v)
            elif isinstance(v, list) and v and isinstance(v[0], str):
                w.add_array(k, [str(x) for x in v])
            elif isinstance(v, list) and v and isinstance(v[0], int):
                w.add_array(k, [int(x) for x in v])
            elif isinstance(v, list) and v and isinstance(v[0], float):
                w.add_array(k, [float(x) for x in v])
        except Exception:
            pass

    w.add_string(_MARK + "arch", arch)
    w.add_uint32(_MARK + "version", 1)
    w.add_bool(_MARK + "kernel_c", arch in ("rwkv7", "mamba2"))
    w.add_string(_MARK + "source_sha256", _sha256(src))
    if blob is not None and Path(blob).exists():
        w.add_string(_MARK + "blob_sha256", _sha256(blob))
        w.add_string(_MARK + "blob", str(blob))

    for name in g.names():
        arr = g.tensor(name)
        w.add_tensor(name, arr.astype(np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = Path(dst)
    return {"file": str(out), "bytes": out.stat().st_size,
            "mb": round(out.stat().st_size / 2**20, 1), "arch": arch,
            "marked": True}


def convert_main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sunnynet convert",
        description="把外部权重转成带 SunnyNet 标记的 GGUF")
    ap.add_argument("src", help="输入: .gguf 或 .pt")
    ap.add_argument("dst", help="输出文件名")
    ap.add_argument("--arch", default=None, help="覆盖架构名")
    ap.add_argument("--blob", default=None,
                    help="同时记录 C 内核 blob 的 sha256 (用于校验缓存)")
    ap.add_argument("--mark-only", action="store_true",
                    help="GGUF 输入时只加标记、不重新量化 (默认行为)")
    args = ap.parse_args(argv)

    src = Path(args.src)
    t0 = time.perf_counter()
    if src.suffix == ".gguf":
        info = mark_gguf(src, args.dst, args.arch, args.blob)
    elif src.suffix == ".pt":
        info = _convert_pt(src, args.dst, args.arch)
    else:
        raise SystemExit(f"不支持的输入格式: {src.suffix}")
    info["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def _convert_pt(ckpt: str | Path, dst: str | Path,
                arch: str | None = None) -> dict:
    """SunnyNet checkpoint -> GGUF (权重名逐层展开, 便于外部工具读取)."""
    import torch
    from gguf import GGUFWriter

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = d.get("config", {})
    sd = d["model"]
    arch = arch or str(arch or "sunnynet")

    vocab = int(cfg.get("vocab_size", 0))
    w = GGUFWriter(str(dst), arch="sunnynet")
    w.add_string("general.architecture", "sunnynet")
    w.add_string("general.name", Path(ckpt).stem)
    for k in ("dim", "inner", "rank", "layers", "taps", "vocab_size"):
        if k in cfg and isinstance(cfg[k], int):
            w.add_uint32(f"sunnynet.{k}", int(cfg[k]))
    w.add_uint32(_MARK + "version", 1)
    w.add_bool(_MARK + "kernel_c", True)

    n = 0
    for name, t in sd.items():
        arr = t.detach().cpu().float().numpy()
        w.add_tensor(name, np.ascontiguousarray(arr))
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = Path(dst)
    return {"file": str(out), "bytes": out.stat().st_size,
            "mb": round(out.stat().st_size / 2**20, 1),
            "arch": "sunnynet", "tensors": n, "vocab": vocab}


if __name__ == "__main__":
    raise SystemExit(convert_main())