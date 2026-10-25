"""端侧 ARM 设备的平台适配: 大小核识别与线程亲和性.

树莓派 / RK3588 这类 big.LITTLE 设备上, 解码线程若被调度到小核会明显变慢。
本模块从 sysfs 读各 CPU 的最高频率, 把频率高的一层当作大核, 只让工作线程
跑在大核上; 频率信息缺失时退回按 CPU 编号的后半段 (常见布局是 0-3 小核)。

只在 Linux 上有实际效果; 其他平台上线程集合为空, 调用方应视为"不限制"。
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

__all__ = ["is_arm", "cpu_freqs", "big_cores", "affinity_available",
           "set_affinity", "describe"]


def is_arm() -> bool:
    m = platform.machine().lower()
    return m in ("aarch64", "arm64") or m.startswith("arm")


def _read_int(p: Path) -> int | None:
    try:
        return int(p.read_text().strip())
    except Exception:
        return None


def cpu_freqs() -> dict[int, int]:
    """{cpu_id: 最高频率 kHz}; 读不到返回空 dict."""
    base = Path("/sys/devices/system/cpu")
    out: dict[int, int] = {}
    if not base.exists():
        return out
    for d in sorted(base.glob("cpu[0-9]*")):
        name = d.name
        if not name[3:].isdigit():
            continue
        cid = int(name[3:])
        freq = _read_int(d / "cpufreq" / "cpuinfo_max_freq")
        if freq is None:
            freq = _read_int(d / "cpufreq" / "scaling_max_freq")
        if freq:
            out[cid] = freq
    return out


def big_cores(threshold: float = 0.9) -> list[int]:
    """频率在最高档 threshold 以上的 CPU 编号; 无频率信息时取编号后半段.

    只在 ARM 上返回结果: x86 的"后半段"并不代表更快的核 (同构), 若返回会被
    调用方误当成大核集合。
    """
    if not is_arm():
        return []
    fr = cpu_freqs()
    if fr:
        top = max(fr.values())
        big = sorted(c for c, f in fr.items() if f >= top * threshold)
        if big:
            return big
    n = os.cpu_count() or 0
    if n <= 1:
        return list(range(n))
    return list(range(n // 2, n))


def affinity_available() -> bool:
    """Linux 且有 sched_setaffinity 才能绑核."""
    return platform.system() == "Linux" and hasattr(os, "sched_setaffinity")


def set_affinity(cpus: list[int] | None = None, threads: int = 0) -> list[int]:
    """把当前进程绑到大核; 返回实际使用的 CPU 列表 (平台不支持时为空).

    threads>0 时只取前 threads 个大核。
    """
    if not affinity_available():
        return []
    cpus = cpus or big_cores()
    if threads > 0:
        cpus = cpus[:threads]
    if not cpus:
        return []
    try:
        os.sched_setaffinity(0, set(cpus))
    except Exception:
        return []
    return cpus


def describe() -> dict:
    """给日志/文档用的平台画像."""
    fr = cpu_freqs()
    big = big_cores()
    return {
        "machine": platform.machine(),
        "is_arm": is_arm(),
        "ncpu": os.cpu_count(),
        "freqs_khz": fr,
        "big_cores": big,
        "affinity_available": affinity_available(),
    }