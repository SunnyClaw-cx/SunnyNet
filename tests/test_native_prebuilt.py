"""预编译库的过期判定契约。

包内 _prebuilt/<平台>/ 的产物可能与同包 C 源码不一致 (开发态改过源码但没重编)。
比对必须是内容哈希而非 mtime: pip 解包会把源码 mtime 刷成解包时刻, 恒新于产物,
按时间判会把装好的包一律判成过期。
"""
from __future__ import annotations

import json

import pytest

from sunnynet import _native


def _dll(mod_name):
    """产物名带平台后缀 (.dll / .so), 不能硬编码。"""
    return __import__(mod_name, fromlist=["x"]).DLL.name


def test_sources_cover_all_modules():
    """每个待编译模块都要有源码清单与 _link_cmd, 否则构建期才发现缺。"""
    for name in _native.MODULES:
        assert name in _native.SOURCES, name
        assert _native.SOURCES[name]
        mod = __import__(name, fromlist=["x"])
        assert hasattr(mod, "_link_cmd"), f"{name} 缺 _link_cmd"
        assert hasattr(mod, "DLL"), f"{name} 缺 DLL"


def test_every_runtime_c_module_is_registered():
    """runtime/ 下的每个 *_c.py 都要登记进 MODULES。

    漏登记不会让任何测试变红: build_prebuilt 只按 MODULES 遍历, 缺登记的库
    既不进 wheel 也不写清单, 装机后只能退回源码编译 (需自带 C 编译器)。
    """
    rt = _native.HERE / "runtime"
    for p in sorted(rt.glob("*_c.py")):
        name = f"sunnynet.runtime.{p.stem}"
        assert name in _native.MODULES, f"{name} 未登记进 MODULES"


def test_manifest_written_and_matches(tmp_path):
    made, err = _native.build_prebuilt(tmp_path)
    if err:
        pytest.skip(f"无编译器: {err}")
    man_p = tmp_path / "sources.sha256"
    assert man_p.is_file()
    man = json.loads(man_p.read_text(encoding="utf-8"))
    # 按库分组, 键为产物文件名
    for name in _native.MODULES:
        dll = _dll(name)
        assert dll in man, dll
        assert set(man[dll]) == set(_native.SOURCES[name])
        for rel, h in man[dll].items():
            assert h == _native._sha256(_native.HERE / rel)


def test_stale_detects_edited_source(tmp_path, monkeypatch):
    """源码内容一变, 即使 mtime 更旧也要判过期。"""
    made, err = _native.build_prebuilt(tmp_path)
    if err:
        pytest.skip(f"无编译器: {err}")
    name = _dll("sunnynet.runtime")
    monkeypatch.setattr(_native, "PREBUILT", tmp_path.parent)
    monkeypatch.setattr(_native, "platform_tag", lambda: tmp_path.name)
    monkeypatch.setattr(_native, "_STALE_CACHE", {})
    assert _native.prebuilt_stale(name) is False

    rel = "runtime/recur.c"
    src = _native.HERE / rel
    man = json.loads((tmp_path / "sources.sha256").read_text(encoding="utf-8"))
    man[name][rel] = "0" * 64           # 伪造: 源码哈希与清单不符
    (tmp_path / "sources.sha256").write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(_native, "_STALE_CACHE", {})
    assert _native.prebuilt_stale(name) is True
    assert src.is_file()                    # 只读判定, 不动源码


def test_missing_manifest_is_not_stale(tmp_path, monkeypatch):
    """没有清单 (sdist 装出来的包) 时无法判定, 不应误判过期。"""
    monkeypatch.setattr(_native, "PREBUILT", tmp_path)
    monkeypatch.setattr(_native, "platform_tag", lambda: tmp_path.name)
    monkeypatch.setattr(_native, "_STALE_CACHE", {})
    assert _native.prebuilt_stale(_dll("sunnynet.runtime")) is False
    assert _native.prebuilt_stale() is False


def test_missing_source_is_not_stale(tmp_path, monkeypatch):
    """源码没随包分发时无法判定, 不应误判过期 (退回编译器路径即可)。"""
    made, err = _native.build_prebuilt(tmp_path)
    if err:
        pytest.skip(f"无编译器: {err}")
    name = _dll("sunnynet.runtime")
    man = json.loads((tmp_path / "sources.sha256").read_text(encoding="utf-8"))
    man[name]["runtime/recur.c"] = "0" * 64
    (tmp_path / "sources.sha256").write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(_native, "PREBUILT", tmp_path.parent)
    monkeypatch.setattr(_native, "platform_tag", lambda: tmp_path.name)
    monkeypatch.setattr(_native, "_STALE_CACHE", {})
    monkeypatch.setattr(_native, "HERE", tmp_path / "nonexistent")
    assert _native.prebuilt_stale(name) is False


def test_rebuild_clears_stale_cache(tmp_path, monkeypatch):
    """重建后应清掉过期缓存, 否则改源码->重建->加载会一直判过期。"""
    made, err = _native.build_prebuilt(tmp_path)
    if err:
        pytest.skip(f"无编译器: {err}")
    monkeypatch.setattr(_native, "PREBUILT", tmp_path.parent)
    monkeypatch.setattr(_native, "platform_tag", lambda: tmp_path.name)
    monkeypatch.setattr(_native, "_STALE_CACHE",
                        {_dll("sunnynet.runtime"): True})
    _native.build_prebuilt(tmp_path)
    assert _native._STALE_CACHE == {}


def test_build_prebuilt_while_libs_are_loaded(tmp_path):
    """已加载的共享库不能被覆盖写 (Windows 会拒绝), 构建须落在别处。

    回归: build_prebuilt 曾直接调各模块的 _compile, 写的是共享的 _build/<lib>;
    该文件一旦被本进程加载过, 编译必失败。
    """
    from sunnynet.runtime import model
    if not model.model_available():
        pytest.skip("无编译器且无预编译产物")
    model._load()                       # 让 model 库处于已加载状态
    made, err = _native.build_prebuilt(tmp_path)
    if err:
        pytest.skip(f"无编译器: {err}")
    for name in _native.MODULES:
        assert (tmp_path / _dll(name)).is_file(), name