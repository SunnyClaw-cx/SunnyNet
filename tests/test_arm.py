"""ARM 平台适配: 大小核识别、线程数与绑核.

本机是 x86, 真实 ARM 行为只能靠模拟 sysfs 与 monkeypatch 验证。测的是
"给定频率表会怎么选", 不是真机调度效果。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from sunnynet.kernels import arm


def test_big_cores_empty_on_non_arm(monkeypatch):
    """x86 上不该给出"大核"集合, 否则会被调用方误用."""
    monkeypatch.setattr(arm, "is_arm", lambda: False)
    assert arm.big_cores() == []
    assert arm.set_affinity() == []


def test_big_cores_by_freq(monkeypatch):
    """RK3588 式布局: 4 小核 1.8GHz + 4 大核 2.4GHz."""
    freqs = {0: 1_800_000, 1: 1_800_000, 2: 1_800_000, 3: 1_800_000,
             4: 2_400_000, 5: 2_400_000, 6: 2_400_000, 7: 2_400_000}
    monkeypatch.setattr(arm, "is_arm", lambda: True)
    monkeypatch.setattr(arm, "cpu_freqs", lambda: freqs)
    assert arm.big_cores() == [4, 5, 6, 7]
    # 三档 (A76 2.4G / A76 2.0G / A55 1.8G) 时只取最高一档
    monkeypatch.setattr(arm, "cpu_freqs",
                        lambda: {**freqs, 4: 2_000_000, 5: 2_000_000})
    assert arm.big_cores() == [6, 7]


def test_big_cores_falls_back_to_upper_half(monkeypatch):
    """没有 cpufreq 信息时取编号后半段 (树莓派这类常见布局)."""
    monkeypatch.setattr(arm, "is_arm", lambda: True)
    monkeypatch.setattr(arm, "cpu_freqs", lambda: {})
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert arm.big_cores() == [4, 5, 6, 7]


def test_set_affinity_truncates_to_threads(monkeypatch):
    """threads 只取前 N 个大核; 并核调用需收到这些编号."""
    monkeypatch.setattr(arm, "is_arm", lambda: True)
    monkeypatch.setattr(arm, "cpu_freqs", lambda: {})
    monkeypatch.setattr(arm, "affinity_available", lambda: True)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    seen = []
    monkeypatch.setattr(os, "sched_setaffinity",
                        lambda pid, cpus: seen.append((pid, set(cpus))),
                        raising=False)
    assert arm.set_affinity(threads=2) == [4, 5]
    assert seen == [(0, {4, 5})]


def test_set_affinity_swallows_oserror(monkeypatch):
    """绑核被拒 (容器/权限不足) 时不得抛, 只当没有大核."""
    monkeypatch.setattr(arm, "is_arm", lambda: True)
    monkeypatch.setattr(arm, "cpu_freqs", lambda: {})
    monkeypatch.setattr(arm, "affinity_available", lambda: True)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    def _boom(pid, cpus):
        raise OSError("EPERM")

    monkeypatch.setattr(os, "sched_setaffinity", _boom, raising=False)
    assert arm.set_affinity(threads=2) == []


def test_threads_for_blob_uses_big_core_count_on_arm(monkeypatch):
    """ARM 上线程数收敛到大核数, 而非物理核总数."""
    from sunnynet.runtime import model as M
    monkeypatch.setattr(arm, "is_arm", lambda: True)
    monkeypatch.setattr(arm, "cpu_freqs", lambda: {})
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert M.recommended_threads() == 4
    monkeypatch.setattr(arm, "is_arm", lambda: False)
    assert M.recommended_threads() > 0


def test_describe_reports_platform(monkeypatch):
    monkeypatch.setattr(arm, "is_arm", lambda: True)
    monkeypatch.setattr(arm, "cpu_freqs", lambda: {0: 1_800_000, 4: 2_400_000})
    d = arm.describe()
    assert d["is_arm"] is True
    assert d["big_cores"] == [4]
    assert d["freqs_khz"] == {0: 1_800_000, 4: 2_400_000}