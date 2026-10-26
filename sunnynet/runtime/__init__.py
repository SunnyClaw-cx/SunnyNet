"""SunnyNet Runtime: 与 PyTorch 解耦的独立 C 推理内核 (AVX2/FMA + OpenMP).

只依赖 ctypes 与 C 编译器, 不依赖 torch。递推形式:
    s_t = a_t * s_{t-1} + b_t          逐通道独立 (SN_FMA)
    s_t = a_t * s_{t-1} + (1-a_t)*u_t  门控形式   (SN_GATED)

外部权重的单步递推走递推式而非分块扫描:
    rwkv7     逐 head 矩阵 + rank-1 校正, GGUF 权重
    mamba2    逐 head 矩阵但衰减外积可分解, 退化为 SN_FMA
矩阵值递推 (rank-1) 不能由 SN_FMA 表达, 走 sn_mat_scan。
"""
from __future__ import annotations

import ctypes
from pathlib import Path

__all__ = [
    "runtime_available", "load_runtime", "FMA", "GATED",
    "scan", "scan_bwd", "step", "mat_step", "quant_i8", "dequant_i8",
    "TOOLCHAIN_HINT", "compile_flags", "shlib_name", "MODELS", "load_model",
]

# 外部权重架构注册表: 名字 -> (模块, 类)
MODELS = {
    "rwkv7": ("sunnynet.runtime.rwkv7", "RWKV7"),
    "mamba2": ("sunnynet.runtime.mamba2", "Mamba2"),
}


def load_model(arch: str, path, **kw):
    """按架构名加载外部权重, 返回可直接 step(token, state) 的对象."""
    if arch not in MODELS:
        raise KeyError(f"未知架构 {arch!r}; 可用: {sorted(MODELS)}")
    mod_name, cls_name = MODELS[arch]
    import importlib
    return getattr(importlib.import_module(mod_name), cls_name)(path, **kw)

FMA = 0
GATED = 1

HERE = Path(__file__).resolve().parent
SRC = HERE / "recur.c"
BUILD = HERE / "_build"

TOOLCHAIN_HINT = (
    "未找到 C 编译器。设置环境变量 SUNNY_W64 指向 w64devkit 根目录, 例如:\n"
    "  set SUNNY_W64=<w64devkit 解压目录>\n"
    "或把 gcc 加入 PATH。"
)

_lib = None

_COMMON_FLAGS = ["-O3", "-mavx2", "-mfma", "-march=x86-64-v3", "-funroll-loops"]


def compile_flags(std: bool = True) -> list[str]:
    """生成共享库的编译与链接参数.

    Windows (mingw) 需要 `-static`/`-shared` 才能产出独立 dll; Linux 的 GNU ld
    不接受 `-static -shared` 组合, 且必须显式 `-fPIC`, 否则报
    "attempted static link of dynamic object" 与 "recompile with -fPIC"。
    """
    import sys
    f = list(_COMMON_FLAGS)
    if sys.platform == "win32":
        f += ["-fopenmp", "-static", "-shared"]
    else:
        f += ["-fopenmp", "-fPIC", "-shared"]
    if std:
        f += ["-std=c11"]
    return f


def shlib_name(stem: str) -> str:
    """共享库文件名; ctypes 按平台后缀查找更稳。"""
    import sys
    return f"{stem}.dll" if sys.platform == "win32" else f"{stem}.so"


DLL = BUILD / shlib_name("sunnynet_runtime")


def _find_gcc() -> str | None:
    from shutil import which
    env = __import__("os").environ.get("SUNNY_W64")
    if env:
        for name in ("gcc.exe", "gcc"):
            c = Path(env) / "bin" / name
            if c.exists():
                return str(c)
    return which("gcc")


def runtime_available() -> bool:
    """有编译器, 或有可用的预编译库。"""
    from .. import _native
    return _find_gcc() is not None or _native.locate(DLL.name) is not None


def _link_cmd(gcc: str, dst) -> list[str]:
    return [gcc] + compile_flags() + [str(HERE / "recur.c"), "-o", str(dst)]


def _compile(gcc: str) -> None:
    from .. import _native
    BUILD.mkdir(parents=True, exist_ok=True)
    rc, msg = _native.run_cc(_link_cmd(gcc, DLL))
    if rc != 0:
        raise RuntimeError(f"runtime 编译失败: {msg[-800:]}")


def _needs_rebuild() -> bool:
    if not DLL.exists():
        return True
    dll_t = DLL.stat().st_mtime
    return any(s.stat().st_mtime > dll_t
               for s in (SRC, HERE / "recur.h"))


def _bind(lib):
    V, I = ctypes.c_void_p, ctypes.c_int
    lib.sn_set_threads.argtypes = [I]
    lib.sn_threads.restype = I
    lib.sn_recur_scan.argtypes = [V, V, V, V, V, I, I, I, I]
    lib.sn_recur_bwd.argtypes = [V, V, V, V, V, V, V, V, I, I, I, I]
    lib.sn_recur_step.argtypes = [V, V, V, I, I, I]
    lib.sn_quant_i8.argtypes = [V, V, V, I, I, I]
    lib.sn_dequant_i8.argtypes = [V, V, V, I, I, I]
    lib.sn_mat_scan.argtypes = [V, V, V, V, V, V, V, V, V, I, I, I]
    return lib


def load_runtime(force_rebuild: bool = False):
    global _lib
    if _lib is not None and not force_rebuild:
        return _lib
    if not force_rebuild:
        from .. import _native
        pre = _native.locate(DLL.name)
        if pre is not None:
            _lib = _bind(ctypes.CDLL(str(pre)))
            return _lib
    gcc = _find_gcc()
    if gcc is None:
        raise RuntimeError(TOOLCHAIN_HINT)
    if force_rebuild or _needs_rebuild():
        _compile(gcc)
    _lib = _bind(ctypes.CDLL(str(DLL)))
    return _lib


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def scan(a, b, s0, gated=False):
    """a, b: (B,T,N); s0: (B,N) -> (out (B,T,N), send (B,N))"""
    import torch
    lib = load_runtime()
    B, T, N = a.shape
    out = torch.empty_like(a)
    send = torch.empty_like(s0)
    lib.sn_recur_scan(_ptr(a), _ptr(b), _ptr(s0), _ptr(out), _ptr(send),
                      B, T, N, GATED if gated else FMA)
    return out, send


def scan_bwd(a, b, s0, out, dout, gated=False):
    """-> (da, db, ds0)"""
    import torch
    lib = load_runtime()
    B, T, N = a.shape
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    ds0 = torch.empty_like(s0)
    lib.sn_recur_bwd(_ptr(a), _ptr(b), _ptr(s0), _ptr(out), _ptr(dout),
                     _ptr(da), _ptr(db), _ptr(ds0),
                     B, T, N, GATED if gated else FMA)
    return da, db, ds0


def step(a, b, s, gated=False):
    """a, b, s: (B,N); s 原地更新。"""
    lib = load_runtime()
    B, N = a.shape
    lib.sn_recur_step(_ptr(a), _ptr(b), _ptr(s), B, N, GATED if gated else FMA)
    return s


def mat_step(S, w=None, u=None, v=None, k=None, b=None, r=None):
    """矩阵值 rank-1 递推单步 (RWKV-7 的矩阵状态层级).

    S: (NH,R,C) 原地更新; w/u/k/b/r: (NH,C); v: (NH,R)。任一项传 None 即跳过,
    语义见 recur.h 的 sn_mat_scan。返回 y (NH,R)(r 为 None 时返回 None)。
    """
    import torch
    lib = load_runtime()
    NH, R, C = S.shape
    y = torch.empty((NH, R), dtype=torch.float32) if r is not None else None
    pb = torch.empty((NH, R), dtype=torch.float32)
    if S.dtype != torch.float32 or not S.is_contiguous():
        raise ValueError("S 必须是连续 fp32")
    def _p(t):
        if t is None:
            return 0
        if t.dtype != torch.float32 or not t.is_contiguous():
            raise ValueError("参数必须是连续 fp32")
        return t.data_ptr()
    lib.sn_mat_scan(_ptr(S), _p(w), _p(u), _p(v), _p(k), _p(b), _p(r),
                    _ptr(y) if y is not None else 0, _ptr(pb), NH, R, C)
    return y


def quant_i8(x, block: int = 0):
    """x: (B,N) fp32 -> (q int8 (B,N), scale (B,nb) fp32)"""
    import torch
    lib = load_runtime()
    B, N = x.shape
    nb = (N + block - 1) // block if block > 0 else 1
    q = torch.empty((B, N), dtype=torch.int8)
    scale = torch.empty((B, nb), dtype=torch.float32)
    lib.sn_quant_i8(_ptr(x), _ptr(q), _ptr(scale), B, N, block)
    return q, scale


def dequant_i8(q, scale, block: int = 0):
    import torch
    lib = load_runtime()
    B, N = q.shape
    x = torch.empty((B, N), dtype=torch.float32)
    lib.sn_dequant_i8(_ptr(q), _ptr(scale), _ptr(x), B, N, block)
    return x