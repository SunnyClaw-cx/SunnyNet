"""AVX-512 与 ARM NEON 递推内核的加载层.

AVX-512: 需要 CPU 支持 AVX512F。加载前用 CPUID 探测, 不支持则拒绝加载,
避免非法指令崩溃。编译需要能生成 AVX-512 的 GCC (>=5) 或 clang。

NEON: AArch64 上 NEON 是必备特性, 无需运行期探测。在 x86 主机上只能交叉
编译产出 .so, 不能加载执行——本层在架构不匹配时明确报错而非静默失败。
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from pathlib import Path

__all__ = [
    "avx512_available", "avx512_supported", "load_avx512",
    "load_avx512_core", "avx512_core_compiles",
    "neon_available", "load_neon", "cross_compile_neon",
    "arch_name",
]

HERE = Path(__file__).resolve().parent
BUILD = HERE / "_build"

_AVX512_FLAGS = ["-O3", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mfma",
                 "-funroll-loops", "-fopenmp", "-std=c11"]
# AArch64 的 NEON 是基线特性; -mfpu 是 ARM32 专有选项, AArch64 的 gcc 会直接拒绝。
_NEON_FLAGS = ["-O3", "-fopenmp", "-std=c11"]


def arch_name() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    if m.startswith("arm"):
        return "arm"
    return m


def _find_gcc() -> str | None:
    from shutil import which
    env = os.environ.get("SUNNY_W64")
    if env:
        for name in ("gcc.exe", "gcc"):
            c = Path(env) / "bin" / name
            if c.exists():
                return str(c)
    return which("gcc")


def _find_zig() -> str | None:
    """zig 可作为交叉编译器; 装在 python 包目录时也能找到."""
    from shutil import which
    z = which("zig")
    if z:
        return z
    try:
        import ziglang
        p = Path(ziglang.__file__).parent / ("zig.exe" if os.name == "nt"
                                             else "zig")
        if p.exists():
            return str(p)
    except Exception:
        pass
    return None


# ------------------------------------------------------------- AVX-512
def avx512_supported() -> bool:
    """运行期探测 CPU 是否支持 AVX512F (CPUID leaf 7 EBX bit16)."""
    if arch_name() != "x86_64":
        return False
    if os.name != "nt":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                return "avx512f" in f.read()
        except OSError:
            pass
    return _avx512_via_probe()


def _avx512_via_probe() -> bool:
    """编译一段 CPUID 探测程序并运行; 失败保守返回 False."""
    gcc = _find_gcc()
    if gcc is None:
        return False
    BUILD.mkdir(parents=True, exist_ok=True)
    src = BUILD / "_cpuid_probe.c"
    exe = BUILD / ("_cpuid_probe.exe" if os.name == "nt" else "_cpuid_probe")
    src.write_text(
        "#include <stdio.h>\n"
        "int main(void){\n"
        "  unsigned eax,ebx,ecx,edx,maxl;\n"
        "  __asm__ volatile(\"cpuid\":\"=a\"(maxl):\"a\"(0):\"ebx\",\"ecx\",\"edx\");\n"
        "  if(maxl<7){printf(\"0\");return 0;}\n"
        "  __asm__ volatile(\"cpuid\":\"=a\"(eax),\"=b\"(ebx),\"=c\"(ecx),\"=d\"(edx)\n"
        "                   :\"a\"(7),\"c\"(0));\n"
        "  printf(\"%d\", (ebx>>16)&1);\n"
        "  return 0;\n"
        "}\n", encoding="utf-8")
    try:
        r = subprocess.run([gcc, "-O2", str(src), "-o", str(exe)],
                           capture_output=True)
        if r.returncode != 0:
            return False
        out = subprocess.run([str(exe)], capture_output=True, text=True)
        return out.stdout.strip() == "1"
    except Exception:
        return False


def avx512_available() -> bool:
    """能编译且本机可执行."""
    return _find_gcc() is not None and avx512_supported()


def _compile_avx512(gcc: str) -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    dll = BUILD / ("sunnynet_scan512.dll" if os.name == "nt"
                   else "sunnynet_scan512.so")
    cmd = [gcc] + _AVX512_FLAGS
    if os.name == "nt":
        cmd += ["-static", "-shared"]
    else:
        cmd += ["-fPIC", "-shared"]
    cmd += [str(HERE / "scan_avx512.c"), "-o", str(dll)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"AVX-512 编译失败: {err[-700:]}")
    return dll


def _compile_avx512_core(gcc: str) -> Path:
    """融合内核 + 单步内核; 两者编译到一个库里."""
    BUILD.mkdir(parents=True, exist_ok=True)
    dll = BUILD / ("sunnynet_core512.dll" if os.name == "nt"
                   else "sunnynet_core512.so")
    cmd = [gcc] + _AVX512_FLAGS
    if os.name == "nt":
        cmd += ["-static", "-shared"]
    else:
        cmd += ["-fPIC", "-shared"]
    cmd += [str(HERE / "core_avx512.c"), str(HERE / "step_avx512.c"),
            "-DSN_512_NO_COMMON", "-o", str(dll)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"AVX-512 融合内核编译失败: {err[-700:]}")
    return dll


_lib512_core = None


def load_avx512_core(force_rebuild: bool = False):
    """加载 AVX-512 融合内核与单步内核; 本机不支持则抛 RuntimeError.

    即使运行不了, 也可用 build_only=True 先确认能编译通过。
    """
    global _lib512_core
    if _lib512_core is not None and not force_rebuild:
        return _lib512_core
    if not avx512_supported():
        raise RuntimeError(
            f"本机 CPU 不支持 AVX512F (arch={arch_name()}), 拒绝加载以免非法指令")
    gcc = _find_gcc()
    if gcc is None:
        raise RuntimeError("未找到 C 编译器 (设 SUNNY_W64)")
    dll = BUILD / ("sunnynet_core512.dll" if os.name == "nt"
                   else "sunnynet_core512.so")
    if force_rebuild or not dll.exists() or \
            (HERE / "core_avx512.c").stat().st_mtime > dll.stat().st_mtime or \
            (HERE / "step_avx512.c").stat().st_mtime > dll.stat().st_mtime:
        dll = _compile_avx512_core(gcc)
    lib = ctypes.CDLL(str(dll))
    # 指针参数绑 c_void_p: core_auto 与 AVX2 路径一样传 data_ptr() 整数,
    # POINTER(c_float) 只接受 ctypes 指针对象, 传整数会 ArgumentError。
    V, I, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    lib.sn512_isa.restype = P
    lib.sn512_set_threads.argtypes = [I]
    lib.core512_fwd_all_f32.argtypes = [V, V, V, V, V, V, I, I, I]
    lib.core512_bwd_all_f32.argtypes = [V, V, V, V, V, V, V, V, V, V, V, I, I, I]
    lib.core512_fwd_f32.argtypes = [V, V, V, V, V, V, I, I, I]
    lib.core512_bwd_f32.argtypes = [V, V, V, V, V, V, V, V, V, V, V, I, I, I]
    lib.core512_fwd_tail_f32.argtypes = [V, V, V, V, V, V, I, I, I, I]
    lib.core512_bwd_tail_f32.argtypes = [V, V, V, V, V, V, V, V, V, V, V, I, I, I, I]
    lib.step_core512_all_f32.argtypes = [V, V, V, V, V, I, I]
    lib.step_core512_f32.argtypes = [V, V, V, V, V, I, I]
    lib.step_core512_tail_f32.argtypes = [V, V, V, V, V, I, I, I]
    lib.dw_step512_f32.argtypes = [V, V, V, V, I, I, I]
    _lib512_core = lib
    return lib


def avx512_core_compiles(force: bool = False) -> tuple[bool, str]:
    """只验证融合内核能否编译 (无 AVX-512 机器上唯一可做的检查).

    返回 (ok, 说明). 编译产物落在 _build/, 不加载执行。
    """
    gcc = _find_gcc()
    if gcc is None:
        return False, "未找到 C 编译器"
    try:
        dll = _compile_avx512_core(gcc) if force else None
        if dll is None:
            dll = BUILD / ("sunnynet_core512.dll" if os.name == "nt"
                           else "sunnynet_core512.so")
            if not dll.exists() or \
                    (HERE / "core_avx512.c").stat().st_mtime > dll.stat().st_mtime or \
                    (HERE / "step_avx512.c").stat().st_mtime > dll.stat().st_mtime:
                dll = _compile_avx512_core(gcc)
        return True, str(dll)
    except RuntimeError as e:
        return False, str(e)


_lib512 = None


def load_avx512(force_rebuild: bool = False):
    """加载 AVX-512 内核; 本机不支持则抛 RuntimeError."""
    global _lib512
    if _lib512 is not None and not force_rebuild:
        return _lib512
    if not avx512_supported():
        raise RuntimeError(
            f"本机 CPU 不支持 AVX512F (arch={arch_name()}), 拒绝加载以免非法指令")
    gcc = _find_gcc()
    if gcc is None:
        raise RuntimeError("未找到 C 编译器 (设 SUNNY_W64)")
    dll = BUILD / ("sunnynet_scan512.dll" if os.name == "nt"
                   else "sunnynet_scan512.so")
    if force_rebuild or not dll.exists() or \
            (HERE / "scan_avx512.c").stat().st_mtime > dll.stat().st_mtime:
        dll = _compile_avx512(gcc)
    lib = ctypes.CDLL(str(dll))
    F, V, I, P = ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    lib.sn512_cpu_supported.restype = I
    lib.sn512_isa.restype = P
    lib.sn512_set_threads.argtypes = [I]
    lib.scan512_f32.argtypes = [F, F, F, F, F, I, I, I]
    lib.scan512_bwd_f32.argtypes = [F, F, F, F, F, F, F, F, I, I, I]
    lib.step512_f32.argtypes = [F, F, F, I, I, I]
    _lib512 = lib
    return lib


# ----------------------------------------------------------------- NEON
def neon_available() -> bool:
    """本机是 AArch64 且有 C 编译器时可加载执行."""
    return arch_name() == "aarch64" and _find_gcc() is not None


_libn = None


def load_neon(force_rebuild: bool = False):
    """加载 NEON 内核. 非 AArch64 主机上会明确报错."""
    global _libn
    if _libn is not None and not force_rebuild:
        return _libn
    if arch_name() != "aarch64":
        raise RuntimeError(
            f"NEON 内核只能在 AArch64 上执行, 当前 arch={arch_name()}。"
            " 交叉编译请用 cross_compile_neon()。")
    gcc = _find_gcc()
    if gcc is None:
        raise RuntimeError("未找到 C 编译器")
    BUILD.mkdir(parents=True, exist_ok=True)
    so = BUILD / "sunnynet_scan_neon.so"
    if force_rebuild or not so.exists():
        cmd = [gcc] + _NEON_FLAGS + ["-fPIC", "-shared",
                                     str(HERE / "scan_neon.c"), "-o", str(so)]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            err = (r.stderr or r.stdout or b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"NEON 编译失败: {err[-700:]}")
    lib = ctypes.CDLL(str(so))
    F, I, P = ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_char_p
    lib.sn_neon_isa.restype = P
    lib.scan_neon_f32.argtypes = [F, F, F, F, F, I, I, I]
    lib.scan_neon_bwd_f32.argtypes = [F, F, F, F, F, F, F, F, I, I, I]
    lib.step_neon_f32.argtypes = [F, F, F, I, I, I]
    lib.core_neon_fwd_f32.argtypes = [F, F, F, F, F, F, I, I, I]
    _libn = lib
    return lib


def cross_compile_neon(out_path: str | Path | None = None,
                       target: str = "aarch64-linux-gnu") -> Path:
    """用 zig 交叉编译 NEON 内核, 产出可在 AArch64 上加载的 .so.

    主机不是 AArch64 时这是唯一能验证 NEON 代码正确编译的途径;
    产出的 .so 无法在本机执行。
    """
    zig = _find_zig()
    if zig is None:
        raise RuntimeError("未找到 zig; 安装: pip install ziglang")
    BUILD.mkdir(parents=True, exist_ok=True)
    out = Path(out_path) if out_path else BUILD / f"sunnynet_scan_neon.{target}.so"
    cmd = [zig, "cc", "-target", target, "-O3", "-shared", "-fPIC",
           "-std=c11", str(HERE / "scan_neon.c"), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"NEON 交叉编译失败: {err[-700:]}")
    return out