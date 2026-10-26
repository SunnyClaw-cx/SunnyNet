"""把 SunnyNet / RWKV-7 权重接入 Ollama 与 LM Studio.

两者都基于 llama.cpp, 因此接入的前提是权重为 GGUF 且架构被内置运行时支持。
本模块生成配置文件与导入命令, 并做前置校验; 不代替用户执行安装。

Ollama: 用 Modelfile + `ollama create` 导入 GGUF。
LM Studio: 直接把 GGUF 放模型目录, 或把 SunnyNet 的 OpenAI 兼容服务配为
         自定义 endpoint (无需 GGUF 转换)。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["write_modelfile", "find_ollama", "ollama_import",
           "write_lmstudio_config", "stage_lmstudio_gguf",
           "find_lmstudio_models_dir", "probe_gguf_arch", "server_alive",
           "main"]

DEFAULT_OLLAMA_HINTS = [
    r"D:\Ollama\ollama.exe",
    r"C:\Program Files\Ollama\ollama.exe",
    r"/usr/local/bin/ollama",
]

# RWKV-7 的 <s> 约定: "User: ...\n\nAssistant:" 是官方 BlinkDL 提示格式。
# Ollama 内置的 rwkv-world 模板会再包一层 (User:/Assistant: 重复),
# 导致输出退化成复读; 必须显式覆盖 TEMPLATE。
DEFAULT_TEMPLATE = "User: {{ .Prompt }}\n\nAssistant:"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")


def _host_url(host: str) -> str:
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return "http://" + host.rstrip("/")


def _opener() -> urllib.request.OpenerDirector:
    """本地请求直连, 绕开 HTTP(S)_PROXY (企业代理会把 127.0.0.1 也拦成 502)."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def server_alive(host: str | None = None, timeout: float = 3.0) -> tuple[bool, str]:
    """探活 Ollama server. 返回 (是否在线, 版本或错误信息).

    `ollama create` 在 server 未运行时会去拉起本地 app, 失败即报
    "could not locate ollama app"; 先探活能把这个诊断提前。
    """
    url = _host_url(host or OLLAMA_HOST) + "/api/version"
    try:
        with _opener().open(url, timeout=timeout) as r:
            return True, json.loads(r.read().decode("utf-8")).get("version", "?")
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} (代理拦截或端口非 Ollama)"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, str(e)


def find_ollama() -> str | None:
    """按 PATH 与常见安装位置查找 ollama 可执行文件."""
    p = shutil.which("ollama")
    if p:
        return p
    for h in DEFAULT_OLLAMA_HINTS:
        if Path(h).exists():
            return h
    return None


def probe_gguf_arch(gguf_path: str | Path) -> dict:
    """读 GGUF 元数据里的架构与关键超参, 用于判断运行时是否可能支持."""
    try:
        from gguf import GGUFReader
    except ImportError as e:
        raise RuntimeError("需要 gguf 包: pip install sunnynet-cx[gguf]") from e
    r = GGUFReader(str(gguf_path))
    out = {}
    for k in ("general.architecture", "general.name", "general.license",
              "tokenizer.ggml.model"):
        f = r.fields.get(k)
        if f is None:
            continue
        try:
            v = f.contents()
            if hasattr(v, "tolist"):
                v = v.tolist()
        except Exception:
            v = f.parts[-1]
        out[k.split(".")[-1]] = v
    for k, f in r.fields.items():
        if k.startswith("rwkv7.") or k.startswith("llama."):
            try:
                out[k] = int(f.parts[-1])
            except Exception:
                pass
    out["_tensors"] = len(r.tensors)
    return out


# -------------------------------------------------------------- Ollama
MODELEFILE_TEMPLATE = """\
FROM {gguf}

TEMPLATE \"\"\"{template}\"\"\"

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER top_k {top_k}
PARAMETER num_ctx {num_ctx}
PARAMETER stop "User:"
PARAMETER stop "System:"
"""


def write_modelfile(gguf_path: str | Path, out_path: str | Path | None = None,
                    temperature: float = 1.0, top_p: float = 0.95,
                    top_k: int = 40, num_ctx: int = 4096,
                    template: str | None = None) -> Path:
    """生成 Ollama Modelfile. gguf 用绝对路径, 避免 create 时相对路径歧义.

    out_path 缺省时落在 gguf 同目录的 Modelfile; 不要落在含非 ASCII 的临时
    目录 (部分 Ollama 构建在解析该路径时会失败)。
    """
    g = Path(gguf_path).resolve()
    if not g.exists():
        raise FileNotFoundError(g)
    out = Path(out_path) if out_path else g.parent / "Modelfile"
    body = MODELEFILE_TEMPLATE.format(
        gguf=g.as_posix(), temperature=temperature, top_p=top_p,
        top_k=top_k, num_ctx=num_ctx,
        template=DEFAULT_TEMPLATE if template is None else template)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return out


def ollama_import(gguf_path: str | Path, model_name: str,
                  modelfile: str | Path | None = None,
                  ollama_bin: str | None = None,
                  host: str | None = None,
                  template: str | None = DEFAULT_TEMPLATE,
                  run: bool = False) -> tuple[Path, list[str]]:
    """生成 Modelfile 并给出 (或执行) `ollama create` 命令.

    run=False 时只返回命令, 不触碰系统状态。run=True 会先探活 server,
    未运行则直接报出可执行的修复提示, 而不是让 CLI 抛
    "could not locate ollama app"。
    """
    mf = write_modelfile(gguf_path, modelfile, template=template)
    exe = ollama_bin or find_ollama() or "ollama"
    h = host or OLLAMA_HOST
    cmd = [exe, "create", model_name, "-f", str(mf)]
    if run:
        ok, info = server_alive(h)
        if not ok:
            raise RuntimeError(
                f"Ollama server 未运行 ({_host_url(h)}): {info}\n"
                f"  ollama create 在 server 离线时会尝试拉起本地 app, "
                f"失败即报 'could not locate ollama app'。\n"
                f"  先启动:  set OLLAMA_HOST={h} && ollama serve")
        env = dict(os.environ, OLLAMA_HOST=h)
        r = subprocess.run(cmd, capture_output=True, env=env)
        if r.returncode != 0:
            err = (r.stderr or r.stdout or b"").decode("utf-8", "replace")
            raise RuntimeError(f"ollama create 失败: {err[-600:]}")
    return mf, cmd


# ----------------------------------------------------------- LM Studio
DEFAULT_LMSTUDIO_HINTS = [
    Path.home() / ".lmstudio" / "models",
    Path.home() / ".cache" / "lm-studio" / "models",
]


def find_lmstudio_models_dir() -> Path | None:
    """找 LM Studio 的模型目录 (存在才算)."""
    for h in DEFAULT_LMSTUDIO_HINTS:
        if h.is_dir():
            return h
    return None


def stage_lmstudio_gguf(gguf_path: str | Path,
                        models_dir: str | Path | None = None,
                        publisher: str = "sunny-cx",
                        repo: str = "rwkv7-191M-world",
                        copy: bool = False) -> Path:
    """把 GGUF 放进 LM Studio 模型目录, 返回目标路径.

    LM Studio 的模型目录是 `<models>/<publisher>/<repo>/<file>.gguf`,
    目录结构本身即索引, 不需要额外注册。
    copy=False 用硬链接 (同盘秒完成, 零额外占用), 跨盘自动回退为复制。
    """
    g = Path(gguf_path).resolve()
    if not g.exists():
        raise FileNotFoundError(g)
    root = Path(models_dir) if models_dir else find_lmstudio_models_dir()
    if root is None:
        raise RuntimeError(
            "未找到 LM Studio 模型目录; 用 --models-dir 指定 "
            "(默认查找 ~/.lmstudio/models 与 ~/.cache/lm-studio/models)")
    dst_dir = root / publisher / repo
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / g.name
    if dst.exists():
        return dst
    if copy:
        shutil.copy2(g, dst)
        return dst
    try:
        os.link(g, dst)
    except OSError:
        shutil.copy2(g, dst)
    return dst


def write_lmstudio_config(out_path: str | Path,
                          base_url: str = "http://127.0.0.1:8080/v1",
                          model_id: str = "sunnynet",
                          api_key: str = "sk-none",
                          gguf_placed: str | None = None) -> Path:
    """生成 LM Studio 接入说明 (两种路径: GGUF 直放 / 自定义 endpoint).

    LM Studio 0.3+ 支持在「开发者」页添加自定义 OpenAI 兼容服务;
    这里给出一份可直接抄填的参数清单 (JSON)。
    """
    cfg = {
        "_comment": "LM Studio 接入 SunnyNet / RWKV-7; 两条路径任选其一",
        "path_a_gguf": {
            "desc": "RWKV-7 GGUF 直接放进模型目录 (LM Studio 原生加载)",
            "note": "需 LM Studio 内置 llama.cpp 运行时支持 rwkv7 架构",
            "gguf_placed": gguf_placed or "<未放置; 用 sunnynet-integrate lmstudio --gguf ...>",
            "prompt_format": "User: {prompt}\\n\\nAssistant:",
            "stop": ["User:", "System:"],
        },
        "path_b_endpoint": {
            "desc": "把 SunnyNet 的 OpenAI 兼容服务配为自定义 endpoint",
            "server_start_command": "sunnynet-serve --hf-dir <模型目录> --hf-arch mamba2 --port 8080",
            "base_url": base_url,
            "api_key": api_key,
            "model_id": model_id,
            "notes": [
                "LM Studio: 打开 'Developer' 标签 -> 添加自定义 endpoint",
                "Base URL 填上面的 base_url",
                "API Key 可留空或填任意非空串 (本地服务不校验)",
                "模型列表会自动从 /v1/models 拉取",
            ],
        },
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


# ------------------------------------------------------------------ CLI
def _smoke(host: str, model: str, prompt: str = "The capital of France is",
           timeout: float = 180.0) -> tuple[bool, str]:
    """导入后发一条确定性 prompt, 确认模型可加载且能出连贯 token."""
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": 24, "temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        _host_url(host) + "/api/generate", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with _opener().open(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, str(e)
    text = (d.get("response") or "").strip()
    return bool(text), text


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="sunnynet-integrate",
        description="生成 Ollama / LM Studio 接入配置")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ollama", help="生成 Modelfile 并给出导入命令")
    a.add_argument("gguf", help="GGUF 权重路径")
    a.add_argument("--name", default="sunnynet-rwkv7", help="Ollama 模型名")
    a.add_argument("--modelfile", default=None)
    a.add_argument("--num-ctx", type=int, default=4096)
    a.add_argument("--temperature", type=float, default=1.0)
    a.add_argument("--top-p", type=float, default=0.95)
    a.add_argument("--top-k", type=int, default=40)
    a.add_argument("--template", default=None,
                   help='覆盖 TEMPLATE; 传 "" 表示不写 (用 Ollama 内置模板)')
    a.add_argument("--host", default=None, help="Ollama server, 默认读 OLLAMA_HOST")
    a.add_argument("--run", action="store_true",
                   help="实际执行 ollama create (默认只打印命令)")
    a.add_argument("--smoke", action="store_true",
                   help="导入后发一条 prompt 验证生成 (需 --run)")

    b = sub.add_parser("lmstudio", help="放置 GGUF 并生成 LM Studio 接入配置")
    b.add_argument("--gguf", default=None, help="要放入模型目录的 GGUF")
    b.add_argument("--models-dir", default=None, help="LM Studio 模型目录")
    b.add_argument("--publisher", default="sunny-cx")
    b.add_argument("--repo", default="rwkv7-191M-world")
    b.add_argument("--copy", action="store_true",
                   help="复制而非硬链接 (跨盘或需要独立副本时用)")
    b.add_argument("--out", default="lmstudio_sunnynet.json")
    b.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    b.add_argument("--model-id", default="sunnynet")

    c = sub.add_parser("probe", help="读取 GGUF 架构与超参")
    c.add_argument("gguf")

    args = ap.parse_args(argv)

    if args.cmd == "ollama":
        info = None
        try:
            info = probe_gguf_arch(args.gguf)
        except Exception as e:
            print(f"[warn] 读取 GGUF 失败: {e}", file=sys.stderr)
        if info:
            print(f"[gguf] architecture={info.get('architecture')} "
                  f"name={info.get('name')} tensors={info.get('_tensors')}")
            if info.get("architecture") not in ("rwkv7", "rwkv", "llama"):
                print("[warn] 该架构未必被 Ollama 内置运行时支持; "
                      "RWKV-7 需较新的 llama.cpp (Ollama >= 0.6)", file=sys.stderr)
            if info.get("architecture") == "rwkv7":
                print("[note] rwkv7 由 Ollama 内置 rwkv-world 模板覆盖: "
                      "本工具写入显式 TEMPLATE 以避免 prompt 双重包裹")
        h = args.host or OLLAMA_HOST
        if args.run:
            ok, ver = server_alive(h)
            print(f"[server] {_host_url(h)} -> "
                  + (f"online (v{ver})" if ok else f"OFFLINE ({ver})"))
        mf, cmd = ollama_import(args.gguf, args.name, args.modelfile,
                                host=h, run=args.run,
                                template=args.template)
        print(f"[modelfile] {mf}")
        print("[cmd] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if not args.run:
            print("[note] 未执行; 加 --run 才会真正导入")
        elif args.smoke:
            ok, out = _smoke(h, args.name)
            print(f"[smoke] {'OK' if ok else 'FAIL'}: {out!r}")
        return 0

    if args.cmd == "lmstudio":
        placed = None
        if args.gguf:
            placed = stage_lmstudio_gguf(args.gguf, args.models_dir,
                                         args.publisher, args.repo, args.copy)
            print(f"[gguf] -> {placed}")
        else:
            d = Path(args.models_dir) if args.models_dir else find_lmstudio_models_dir()
            print(f"[models-dir] {d if d else '未找到 (用 --models-dir 指定)'}")
        out = write_lmstudio_config(args.out, args.base_url, args.model_id,
                                    gguf_placed=str(placed) if placed else None)
        print(f"[config] {out}")
        return 0

    if args.cmd == "probe":
        print(json.dumps(probe_gguf_arch(args.gguf), ensure_ascii=False,
                         indent=2, default=str))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())