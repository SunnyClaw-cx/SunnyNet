"""C Runtime 的 DeltaNet 整模型绑定 (不依赖 torch).

blob 由 experiments/_export_deltanet.py 从 HF safetensors 导出;
本模块只负责加载与单步调用, 前向全在 C。
"""
from __future__ import annotations

import ctypes
from pathlib import Path

from . import BUILD, HERE, _find_gcc, compile_flags, shlib_name

__all__ = ["SnDeltaNet", "deltanet_available"]

DLL = BUILD / shlib_name("sunnynet_deltanet")


def deltanet_available() -> bool:
    from .. import _native
    return _find_gcc() is not None or _native.locate(DLL.name) is not None


def _link_cmd(gcc: str, dst) -> list[str]:
    return [gcc] + compile_flags() + [
        str(HERE / "deltanet.c"), str(HERE / "quant.c"), "-o", str(dst)]


def _compile(gcc: str) -> None:
    from .. import _native
    BUILD.mkdir(parents=True, exist_ok=True)
    rc, msg = _native.run_cc(_link_cmd(gcc, DLL))
    if rc != 0:
        raise RuntimeError(f"deltanet runtime 编译失败: {msg[-800:]}")


def _needs_rebuild() -> bool:
    if not DLL.exists():
        return True
    t = DLL.stat().st_mtime
    return any(s.stat().st_mtime > t
               for s in (HERE / "deltanet.c", HERE / "deltanet.h",
                         HERE / "quant.c"))


_lib = None


def _bind(lib):
    V, I, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    lib.sn_dn_load.restype = V
    lib.sn_dn_load.argtypes = [P]
    lib.sn_dn_free.argtypes = [V]
    lib.sn_dn_reset.argtypes = [V]
    lib.sn_dn_step.argtypes = [V, I, V]
    lib.sn_dn_prefill.argtypes = [V, V, I, V]
    lib.sn_dn_vocab.argtypes = [V]
    lib.sn_dn_vocab.restype = I
    lib.sn_dn_ready.argtypes = [V]
    lib.sn_dn_ready.restype = I
    lib.sn_dn_set_threads.argtypes = [I]
    lib.sn_dn_get_threads.restype = I
    return lib


def _load():
    global _lib
    if _lib is not None:
        return _lib
    from .. import _native
    pre = _native.locate(DLL.name)
    if pre is not None:
        _lib = _bind(ctypes.CDLL(str(pre)))
        return _lib
    gcc = _find_gcc()
    if gcc is None:
        raise RuntimeError("未找到 C 编译器 (设 SUNNY_W64)")
    if _needs_rebuild():
        _compile(gcc)
    _lib = _bind(ctypes.CDLL(str(DLL)))
    return _lib


def _recommended_threads() -> int:
    """物理核数. 解码访存受限, 超线程不增带宽 (实测 6 核 64.3 ms vs 12 逻辑核 68.6 ms)。"""
    try:
        from . import model
        return model.recommended_threads()
    except Exception:
        return 0


class SnDeltaNet:
    """B=1 单序列 DeltaNet 解码句柄. logits 复用同一缓冲."""

    def __init__(self, path, vocab: int, threads: int | None = None):
        self._lib = _load()
        if threads is None:
            threads = _recommended_threads()
        if threads:
            self._lib.sn_dn_set_threads(int(threads))
            from . import model as _model
            _model.pin_to_big_cores(int(threads))
        self._logits = (ctypes.c_float * vocab)()
        self._h = self._lib.sn_dn_load(str(path).encode())
        if not self._h:
            raise RuntimeError(f"sn_dn_load 失败: {path}")
        if not self._lib.sn_dn_ready(self._h):
            raise RuntimeError("sn_dn_load 返回但模型非就绪")
        self.vocab = vocab

    def reset(self):
        self._lib.sn_dn_reset(self._h)

    def step(self, token: int) -> list[float]:
        self._lib.sn_dn_step(self._h, int(token), ctypes.byref(self._logits))
        return list(self._logits)

    def step_raw(self, token: int) -> None:
        """只推进状态, 不把 logits 拷成 Python list (计时用)."""
        self._lib.sn_dn_step(self._h, int(token), ctypes.byref(self._logits))

    def step_np(self, token: int):
        """推进一步并返回 logits 的 numpy 零拷贝视图 (解码循环用, 免 list 转换)."""
        import numpy as np
        self._lib.sn_dn_step(self._h, int(token), ctypes.byref(self._logits))
        return np.frombuffer(self._logits, dtype=np.float32)

    def logits_np(self):
        import numpy as np
        return np.frombuffer(self._logits, dtype=np.float32)

    def logits(self) -> list[float]:
        return list(self._logits)

    def prefill(self, tokens) -> list[float]:
        toks = list(tokens)
        buf = (ctypes.c_int * len(toks))(*toks)
        self._lib.sn_dn_prefill(self._h, buf, len(toks), ctypes.byref(self._logits))
        return list(self._logits)

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                self._lib.sn_dn_free(self._h)
        except Exception:
            pass