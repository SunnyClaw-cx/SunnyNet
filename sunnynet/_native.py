"""预编译共享库的定位与构建.

包内 sunnynet/_prebuilt/<平台标签>/ 存放构建期编好的产物。运行期优先加载它,
命中就不再要求本机有 C 编译器; 未命中或 CPU 指令集不达标时退回源码编译。

预编译库一律以 -mavx2 -mfma -march=x86-64-v3 编出, 只对 x86-64 且支持 AVX2 的
机器可用, 故按指令集判定而非仅按平台。
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

__all__ = [
    "PREBUILT", "platform_tag", "cpu_flags", "isa_ok",
    "locate", "build_prebuilt", "MODULES", "SOURCES", "run_cc",
]

HERE = Path(__file__).resolve().parent
PREBUILT = HERE / "_prebuilt"

# 构建期需要产出预编译库的模块; 每个模块需有 _compile(gcc) 与 DLL 属性。
MODULES = (
    "sunnynet.runtime",
    "sunnynet.runtime.model",
    "sunnynet.runtime.rwkv7_c",
    "sunnynet.runtime.mamba2_c",
    "sunnynet.runtime.quant",
    "sunnynet.runtime.abi",
    "sunnynet.kernels",
)

# 每个预编译库的输入源码 (相对 HERE)。随包分发, 用于判定产物是否过期。
SOURCES = {
    "sunnynet.runtime": ("runtime/recur.c", "runtime/recur.h"),
    "sunnynet.runtime.model": ("runtime/sunnynet.c", "runtime/sunnynet.h"),
    "sunnynet.runtime.rwkv7_c": ("runtime/rwkv7.c", "runtime/rwkv7.h",
                                 "runtime/quant.c"),
    "sunnynet.runtime.mamba2_c": ("runtime/mamba2.c", "runtime/mamba2.h"),
    "sunnynet.runtime.quant": ("runtime/quant.c", "runtime/quant.h"),
    "sunnynet.runtime.abi": ("runtime/sn_abi.c", "runtime/sn_abi.h",
                             "runtime/recur.c", "runtime/rwkv7.c",
                             "runtime/quant.c"),
    "sunnynet.kernels": ("kernels/scan_avx2.c", "kernels/core_avx2.c",
                         "kernels/step_avx2.c"),
}


def _machine() -> str:
    m = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(m, m)


def run_cc(cmd: list[str]):
    """跑编译器。Windows 下 gcc 的诊断按系统代码页输出, 中文路径会撞非 UTF-8 字节;
    带 text=True 会抛 UnicodeDecodeError 把真实编译错误盖掉, 故收 bytes 再容错解码。
    返回 (returncode, 合并后的诊断文本)。
    """
    import subprocess

    r = subprocess.run(cmd, capture_output=True)
    msg = (r.stderr or r.stdout or b"").decode("utf-8", errors="replace")
    return r.returncode, msg


def platform_tag() -> str:
    """对应 _prebuilt 下的子目录名, 如 win-x86_64 / linux-x86_64。"""
    if sys.platform == "win32":
        return f"win-{_machine()}"
    if sys.platform == "darwin":
        return f"macos-{_machine()}"
    return f"linux-{_machine()}"


def cpu_flags() -> set[str]:
    """本机 CPU 特性名集合 (小写)。"""
    if sys.platform != "win32":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8",
                                                  errors="replace")
        except OSError:
            return set()
        for line in text.splitlines():
            low = line.lower()
            if low.startswith("flags") or low.startswith("features"):
                return set(line.split(":", 1)[1].split())
        return set()
    try:
        import ctypes
        import ctypes.wintypes as wt

        k = ctypes.windll.kernel32
        k.IsProcessorFeaturePresent.argtypes = [wt.DWORD]
        k.IsProcessorFeaturePresent.restype = wt.BOOL
        # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40。Windows 无独立的 FMA 查询位,
        # 但 FMA3 与 AVX2 同期引入 (Haswell), 有 AVX2 即有 FMA。
        return {"avx2", "fma"} if k.IsProcessorFeaturePresent(40) else set()
    except Exception:
        return set()


_ISA_CACHE = None


def isa_ok() -> bool:
    """预编译库用 x86-64-v3 编出, 只有 AVX2+FMA 的 x86-64 机器能跑。"""
    global _ISA_CACHE
    if _ISA_CACHE is None:
        if _machine() != "x86_64":
            _ISA_CACHE = False
        else:
            f = cpu_flags()
            _ISA_CACHE = "avx2" in f and "fma" in f
    return _ISA_CACHE


def find(name: str) -> Path | None:
    p = PREBUILT / platform_tag() / name
    return p if p.is_file() else None


def _sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


def _manifest_path() -> Path:
    return PREBUILT / platform_tag() / "sources.sha256"


def all_sources() -> tuple[str, ...]:
    return tuple(sorted({s for srcs in SOURCES.values() for s in srcs}))


def _read_manifest() -> dict:
    import json

    p = _manifest_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


_STALE_CACHE = {}


def prebuilt_stale(shlib_name: str | None = None) -> bool:
    """包内产物是否与同包源码不一致。

    比对内容哈希而非 mtime: pip 解包会把源码 mtime 刷成解包时刻, 恒新于产物,
    按时间判会把装好的包一律判成过期。源码没随包分发时无法判定, 按不过期处理。

    shlib_name 为 None 时检查全部库 (需 import 各模块), 否则只查清单里该库的条目
    ——不 import 任何模块, 少读文件: Windows 实时防护下每次打开约 6 ms。
    """
    if shlib_name is None:
        return any(prebuilt_stale(n) for n in _dll_names())
    if shlib_name in _STALE_CACHE:
        return _STALE_CACHE[shlib_name]
    man = _read_manifest().get(shlib_name) or {}
    stale = False
    for rel, want in man.items():
        p = HERE / rel
        if p.is_file() and _sha256(p) != want:
            stale = True
            break
    _STALE_CACHE[shlib_name] = stale
    return stale


def _dll_names() -> tuple[str, ...]:
    out = []
    for name in MODULES:
        d = _dll_of(name)
        if d:
            out.append(d)
    return tuple(out)


def _dll_of(mod_name: str) -> str | None:
    import importlib

    try:
        return importlib.import_module(mod_name).DLL.name
    except Exception:
        return None


def locate(shlib_name: str) -> Path | None:
    """预编译库可加载则返回路径, 否则 None (调用方退回源码编译)。

    设 SUNNY_NO_PREBUILT=1 可强制走源码路径, 用于对照验证。
    开发态下源码改过则判过期, 避免静默加载旧库 (见 prebuilt_stale)。
    """
    if os.environ.get("SUNNY_NO_PREBUILT") == "1" or not isa_ok():
        return None
    if prebuilt_stale(shlib_name):
        return None
    return find(shlib_name)


def build_prebuilt(out_dir: str | Path | None = None):
    """编译全部原生库并复制到 _prebuilt/<平台标签>/。

    同时写出 sources.sha256 清单, 供运行期判定产物是否与源码一致。
    返回 (产物列表, 错误说明)。当前 C 运行时用 -mavx2 编出, 只有 x86-64 能编;
    其他架构返回空列表并说明, 由调用方决定是否视为失败。

    编译落在临时目录: 各模块的 _compile 写的是共享的 <pkg>/_build/<dll>, 该文件
    一旦被本进程 (或别的测试) 加载过, Windows 不允许覆盖写, 编译必失败。
    """
    import importlib
    import json
    import tempfile

    if _machine() != "x86_64":
        return [], f"{_machine()} 不是 x86-64, 该架构无预编译产物"
    if not isa_ok():
        return [], "构建机不支持 AVX2+FMA, 跳过预编译"

    out = Path(out_dir) if out_dir else PREBUILT / platform_tag()
    made: list[Path] = []
    skipped: list[str] = []
    with tempfile.TemporaryDirectory(prefix="sunnynet-build-") as td:
        tmp = Path(td)
        for name in MODULES:
            mod = importlib.import_module(name)
            gcc = mod._find_gcc()
            if gcc is None:
                return made, f"未找到 C 编译器, 无法生成预编译库 ({name})"
            dst = tmp / mod.DLL.name
            cmd = mod._link_cmd(gcc, dst)
            rc, msg = run_cc(cmd)
            if rc != 0:
                return made, f"{name} 编译失败: {msg[-500:]}"
            out.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(dst, out / dst.name)
            except PermissionError:
                # 目标 dll 已被本进程加载, Windows 不允许覆盖。编译产物仍是新的,
                # 单独说明而不整体失败 —— 其余库照常写出。
                skipped.append(f"{dst.name} (已加载, 未覆盖)")
                continue
            made.append(out / dst.name)

    man = {}
    for name in MODULES:
        dll = importlib.import_module(name).DLL.name
        man[dll] = {rel: _sha256(HERE / rel) for rel in SOURCES[name]
                    if (HERE / rel).is_file()}
    man_p = out / "sources.sha256"
    man_p.write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    made.append(man_p)
    _STALE_CACHE.clear()      # 产物已刷新, 旧的过期判定失效
    if skipped:
        return made, "部分库被占用未覆盖: " + ", ".join(skipped)
    return made, None