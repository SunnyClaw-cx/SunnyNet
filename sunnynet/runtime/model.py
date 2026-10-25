"""C Runtime 整模型绑定的 ctypes 层 (不依赖 torch)."""
from __future__ import annotations

import ctypes
from pathlib import Path

from . import BUILD, HERE, _find_gcc, compile_flags, shlib_name

__all__ = ["SnModel", "model_available", "set_threads", "get_threads",
           "recommended_threads", "threads_for_blob"]

DLL = BUILD / shlib_name("sunnynet_model")


def model_available() -> bool:
    from .. import _native
    return _find_gcc() is not None or _native.locate(DLL.name) is not None


def _link_cmd(gcc: str, dst) -> list[str]:
    return [gcc] + compile_flags() + [str(HERE / "sunnynet.c"), "-o", str(dst)]


def _compile(gcc: str) -> None:
    from .. import _native
    BUILD.mkdir(parents=True, exist_ok=True)
    rc, msg = _native.run_cc(_link_cmd(gcc, DLL))
    if rc != 0:
        raise RuntimeError(f"model runtime 编译失败: {msg[-800:]}")


def _needs_rebuild() -> bool:
    if not DLL.exists():
        return True
    t = DLL.stat().st_mtime
    return any(s.stat().st_mtime > t for s in (HERE / "sunnynet.c", HERE / "sunnynet.h"))


_lib = None


def _bind(lib):
    V, I, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    lib.sn_load.restype = V
    lib.sn_load.argtypes = [P, I, I, I, I, I, I, I, I]
    if hasattr(lib, "sn_load_h"):
        lib.sn_load_h.restype = V
        lib.sn_load_h.argtypes = [P, I, I, I, I, I, I, I, I]
    lib.sn_free.argtypes = [V]
    lib.sn_reset.argtypes = [V]
    lib.sn_step.argtypes = [V, I, V]
    lib.sn_prefill.argtypes = [V, V, I, V]
    lib.sn_vocab.argtypes = [V]
    lib.sn_vocab.restype = I
    lib.sn_ready.argtypes = [V]
    lib.sn_ready.restype = I
    lib.sn_set_threads.argtypes = [I]
    lib.sn_get_threads.restype = I
    if hasattr(lib, "sn_recommended_threads"):
        lib.sn_recommended_threads.restype = I
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


def set_threads(n: int) -> int:
    """设置 C runtime 的 GEMV 并行度; n<=0 维持默认。返回生效线程数。"""
    lib = _load()
    lib.sn_set_threads(int(n))
    return lib.sn_get_threads()


def get_threads() -> int:
    return _load().sn_get_threads()


def recommended_threads() -> int:
    """物理核数; C 侧探测失败时回退到逻辑核的一半。

    big.LITTLE 设备上按大核数收敛 (解码线程落到小核会拖慢整条流水线),
    非 ARM 平台该分支不生效。
    """
    try:
        from ..kernels.arm import big_cores, is_arm
        if is_arm():
            big = big_cores()
            if big:
                return len(big)
    except Exception:
        pass
    lib = _load()
    if not hasattr(lib, "sn_recommended_threads"):
        return 0
    n = int(lib.sn_recommended_threads())
    if n <= 0:
        import os
        n = max(1, (os.cpu_count() or 2) // 2)
    return n


def pin_to_big_cores(threads: int = 0) -> list[int]:
    """ARM 上把当前进程绑到大核; 其他平台空操作, 返回实际使用的 CPU 列表."""
    try:
        from ..kernels.arm import set_affinity
        return set_affinity(threads=threads)
    except Exception:
        return []


# 小模型权重随每步读完后驻留缓存, 并行无带宽收益、只多同步开销. 实测 (B=1, T=256,
# 单步, 各规模独立子进程): <=23MB 单线程最快, >=39MB 物理核最快 (d768 快 1.29x).
_SMALL_BLOB = 32 * 1024 * 1024


def threads_for_blob(path) -> int:
    """按权重字节数选解码并行度: 小 blob 单线程, 大 blob 物理核。"""
    try:
        size = Path(path).stat().st_size
    except OSError:
        return recommended_threads() or 1
    if size < _SMALL_BLOB:
        return 1
    return max(1, recommended_threads())


class SnModel:
    """B=1 单序列解码句柄. logits 复用同一缓冲."""

    def __init__(self, path, vocab, dim, inner, rank, taps, layers,
                 group=8, tied=1, half=False, threads=None):
        self._lib = _load()
        self._logits = (ctypes.c_float * vocab)()
        self._path = str(path).encode()
        if threads is None:
            threads = threads_for_blob(path)
        self.threads = set_threads(threads)
        pin_to_big_cores(self.threads)
        load = self._lib.sn_load_h if half else self._lib.sn_load
        self._h = load(self._path, vocab, dim, inner, rank,
                       taps, layers, group, int(tied))
        if not self._h:
            raise RuntimeError(f"sn_load 失败: {path} (维度/魔数不匹配)")
        if not self._lib.sn_ready(self._h):
            raise RuntimeError("sn_load 返回但模型非就绪状态")
        self.vocab = vocab
        self._prefill_buf = (ctypes.c_int * max(1, 1))()

    def reset(self):
        self._lib.sn_reset(self._h)

    def step(self, token: int) -> list[float]:
        self._lib.sn_step(self._h, int(token), ctypes.byref(self._logits))
        return list(self._logits)

    def prefill(self, tokens) -> list[float]:
        toks = list(tokens)
        buf = (ctypes.c_int * len(toks))(*toks)
        self._lib.sn_prefill(self._h, buf, len(toks), ctypes.byref(self._logits))
        return list(self._logits)

    def __del__(self):
        try:
            if getattr(self, "_h", None):
                self._lib.sn_free(self._h)
        except Exception:
            pass