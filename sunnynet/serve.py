"""OpenAI 兼容的本地推理 API 服务.

用标准库 http.server, 不引入 web 框架。进程内加载一份权重, 权重只读;
每个请求各自持有递推状态, 故多线程并发安全。

端点:
    GET  /health
    GET  /v1/models
    POST /v1/completions
    POST /v1/chat/completions   (stream=true 时走 SSE)
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = REPO / "data" / "proc" / "tokenizer.json"

# --hf-arch 的合法取值, CLI / serve / ollama 三处共用, 免得再漂移
HF_ARCHES = ("mamba2",)

# 架构 -> (绑定模块, 句柄类); 整模型前向全在 C
_HF_C = {
    "mamba2": ("runtime.mamba2_c", "SnMamba2"),
}

__all__ = ["build_engine", "Engine", "HFEngine", "HFCEngine", "HF_ARCHES",
           "serve", "main"]


# --------------------------------------------------------------------- 采样
def sample_logits(logits, temperature=1.0, top_p=1.0, top_k=0, rng=None,
                  repetition_penalty=1.0, seen=None):
    if repetition_penalty and repetition_penalty != 1.0 and seen:
        z = np.asarray(logits, dtype=np.float64).copy()
        pen = float(repetition_penalty)
        for tid in set(int(x) for x in seen):
            if 0 <= tid < z.size:
                z[tid] = z[tid] / pen if z[tid] > 0 else z[tid] * pen
        if temperature is None or temperature <= 0:
            return int(np.argmax(z))
        logits = z
    if temperature is None or temperature <= 0:
        return int(np.argmax(logits))
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    if top_k and top_k < z.size:
        thr = np.partition(z, -int(top_k))[-int(top_k)]
        z = np.where(z < thr, -np.inf, z)
    if 0.0 < top_p < 1.0:
        order = np.argsort(-z)
        srt = z[order]
        p = np.exp(srt - srt.max())
        p /= p.sum()
        cum = np.cumsum(p)
        cut = cum > top_p
        cut[1:] = cut[:-1]
        cut[0] = False
        srt = np.where(cut, -np.inf, srt)
        z = np.full_like(z, -np.inf)
        z[order] = srt
    z -= np.nanmax(z)
    p = np.exp(z)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(logits))
    p /= total
    rng = rng or np.random.default_rng()
    return int(rng.choice(z.size, p=p))


# ----------------------------------------------------------------- 引擎抽象
class Engine:
    """统一成 encode / decode / init_state / step 四件事."""

    vocab: int = 0
    eos: int | None = None
    name: str = "sunnynet"

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def init_state(self):
        raise NotImplementedError

    def step(self, token: int, state):
        """-> (logits (V,), new_state); numpy 后端就地改状态, 返回同一对象."""
        raise NotImplementedError


class _SPTokenizer:
    """sentencepiece 包装; add_bos 手工 prepend."""

    def __init__(self, model_path: str | Path, bos_id: int = 1, add_bos: bool = True):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))
        self.bos_id = bos_id
        self.add_bos = add_bos

    def encode(self, text: str) -> list[int]:
        ids = list(self.sp.encode(text, out_type=int))
        return [self.bos_id] + ids if self.add_bos else ids

    def decode(self, ids) -> str:
        return self.sp.decode([int(i) for i in ids])


class _Wrap:
    def __init__(self, tk):
        self.tk = tk

    def encode(self, text):
        return list(self.tk.encode(text, add_special_tokens=True))

    def decode(self, ids):
        return self.tk.decode([int(i) for i in ids])


def _hf_tokenizer(src: str | Path):
    """HF 权重目录的分词器: sentencepiece 优先, 否则 transformers."""
    src = Path(src)
    sp_model = src / "tokenizer.model"
    if sp_model.exists():
        return _SPTokenizer(sp_model, bos_id=1, add_bos=True)
    from transformers import AutoTokenizer
    return _Wrap(AutoTokenizer.from_pretrained(str(src)))


def _hf_eos(model_dir: str | Path, tokenizer_dir: str | None = None):
    src = Path(tokenizer_dir or model_dir)
    cp = src / "config.json"
    if not cp.exists():
        return None
    try:
        return json.loads(cp.read_text(encoding="utf-8")).get("eos_token_id")
    except Exception:
        return None


class HFEngine(Engine):
    """Mamba-2: HF safetensors 权重, numpy 单步."""

    def __init__(self, arch: str, model_dir: str, tokenizer_dir: str | None = None):
        self.name = arch
        if arch == "mamba2":
            from .runtime.mamba2 import Mamba2 as Model
        else:
            raise ValueError(f"未支持的 --hf-arch: {arch}")
        self.model = Model(model_dir)
        self.vocab = int(getattr(self.model, "vocab", 0))
        self.eos = _hf_eos(model_dir, tokenizer_dir)
        self.tk = _hf_tokenizer(tokenizer_dir or model_dir)

    def encode(self, text):
        return self.tk.encode(text)

    def decode(self, ids):
        return self.tk.decode(ids)

    def init_state(self):
        return self.model.init_state()

    def step(self, token, state):
        logits = self.model.step(token, state)
        return np.asarray(logits, dtype=np.float32), state


class HFCEngine(Engine):
    """HF 架构走整模型 C 内核单步; 权重由 --blob 指定的 .c.bin / .q4.bin.

    与 HFEngine 的区别只在 step: 前向全在 C, 状态由 C 侧持有, 故 init_state
    只做 reset。句柄非线程安全, 用锁把单步串行化 (顺带也挡了并发下状态互踩)。
    """

    def __init__(self, arch: str, blob: str | Path, model_dir: str | None = None,
                 tokenizer_dir: str | None = None, threads: int = 0):
        if arch not in _HF_C:
            raise ValueError(f"未支持的 --hf-arch: {arch}")
        bp = Path(blob)
        if not bp.exists():
            raise SystemExit(f"缺少 C 内核 blob: {bp}")
        mod_name, cls_name = _HF_C[arch]
        import importlib
        cls = getattr(importlib.import_module("." + mod_name, __package__), cls_name)
        from .runtime.blob import read_blob_vocab
        vocab = read_blob_vocab(bp)
        self.model = cls(bp, vocab, threads=threads or None)
        self.name = f"{arch}-c"
        self.vocab = int(vocab)
        self._lock = threading.Lock()
        src = tokenizer_dir or model_dir
        self.eos = _hf_eos(src) if src else None
        self.tk = _hf_tokenizer(src) if src else None

    def encode(self, text):
        if self.tk is None:
            raise RuntimeError("HFCEngine 未配分词器, 请给 --hf-dir 或 --tokenizer-dir")
        return self.tk.encode(text)

    def decode(self, ids):
        if self.tk is None:
            return ""
        return self.tk.decode(ids)

    def init_state(self):
        with self._lock:
            self.model.reset()
        return None

    def step(self, token, state):
        with self._lock:
            return np.asarray(self.model.step_np(int(token)),
                              dtype=np.float32), state


class GGUFEngine(Engine):
    """RWKV-7 GGUF 权重, numpy 单步."""

    def __init__(self, path: str):
        from .runtime.blob import require_rwkv7_gguf
        from .runtime.rwkv7 import RWKV7
        require_rwkv7_gguf(path)
        self.model = RWKV7(path)
        self.name = "rwkv7"
        self.vocab = int(self.model.vocab)
        self.eos = None

    def encode(self, text):
        return list(self.model.tok.encode(text))

    def decode(self, ids):
        return self.model.tok.decode([int(i) for i in ids])

    def init_state(self):
        return self.model.init_state()

    def step(self, token, state):
        logits = self.model.step(token, state)
        return np.asarray(logits, dtype=np.float32), state


class GGUFCEngine(Engine):
    """RWKV-7 走 C 内核整模型单步; 词表/分词复用 numpy 侧的 RWKV7."""

    def __init__(self, path: str, blob: str | None = None, quant: str = "fp32"):
        from .runtime.blob import require_rwkv7_gguf
        from .runtime.rwkv7 import RWKV7
        from .runtime.rwkv7_c import SnRWKV7
        require_rwkv7_gguf(path)
        tok = RWKV7(path)
        if blob:
            bp = Path(blob)
            if not bp.exists():
                raise SystemExit(f"缺少权重 blob: {bp}")
        else:
            from .runtime.blob import ensure_rwkv7_blob
            bp = ensure_rwkv7_blob(path, quant=quant)
        self.model = SnRWKV7(bp, tok.vocab)
        self.tok = tok.tok
        self.name = "rwkv7-c"
        self.vocab = int(tok.vocab)
        self.eos = None

    def encode(self, text):
        return list(self.tok.encode(text))

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids])

    def init_state(self):
        self.model.reset()
        return {}

    def step(self, token, state):
        self.model.step_raw(int(token))
        return np.asarray(self.model.logits(), dtype=np.float32), state


class TorchEngine(Engine):
    """SunnyNet 原生 checkpoint, torch 前向."""

    def __init__(self, ckpt: str, tokenizer: str):
        import torch
        from .cli import load_model
        from .tokenizer import BPETokenizer
        self.torch = torch
        model, cfg, _ = load_model(ckpt)
        self.model = model
        self.cfg = cfg
        self.name = "sunnynet"
        self.vocab = int(cfg.vocab_size)
        self.tk = BPETokenizer.load(tokenizer)
        self.eos = getattr(self.tk, "eos_id", None)

    def encode(self, text):
        return list(self.tk.encode(text))

    def decode(self, ids):
        return self.tk.decode([int(i) for i in ids])

    def init_state(self):
        return self.model.init_state(1)

    def step(self, token, state):
        with self.torch.no_grad():
            logits, st = self.model.forward_step(
                self.torch.tensor([int(token)]), state)
        return logits[0].numpy().astype(np.float32), st


def _resolve_model_flags(args) -> None:
    """把 -m/--model 映射到既有 flag, 按后缀识别 .gguf / .pt / HF 目录。

    CLI / serve / ollama 三处共用, 免得 -m 的识别规则各写一份。
    """
    m = getattr(args, "model", None)
    if not m:
        return
    if getattr(args, "gguf", None) or getattr(args, "ckpt", None) \
            or getattr(args, "hf_dir", None):
        return
    p = Path(m)
    if p.suffix == ".gguf" or p.name.endswith(".gguf"):
        args.gguf = m
    elif p.suffix == ".pt":
        args.ckpt = m
    elif p.is_dir():
        args.hf_dir = m
    else:
        args.gguf = m


def build_engine(args) -> Engine:
    if args.hf_dir:
        blob = getattr(args, "blob", None)
        if blob:
            return HFCEngine(args.hf_arch, blob, args.hf_dir, args.tokenizer_dir,
                             threads=getattr(args, "threads", 0) or 0)
        return HFEngine(args.hf_arch, args.hf_dir, args.tokenizer_dir)
    if args.gguf:
        if args.gguf_backend == "c":
            return GGUFCEngine(args.gguf, args.blob, args.quant)
        return GGUFEngine(args.gguf)
    if args.ckpt:
        return TorchEngine(args.ckpt, args.tokenizer)
    raise SystemExit("需要 --ckpt / --gguf / --hf-dir 之一")


# ------------------------------------------------------------------- 生成
def generate_stream(engine, prompt_ids, max_tokens, temperature, top_p,
                    top_k, seed, stops, repetition_penalty=1.0):
    """逐个 yield (token_id, 新增文本); 命中 eos/stop 即止."""
    state = engine.init_state()
    logits = None
    for t in prompt_ids:
        logits, state = engine.step(int(t), state)
    rng = np.random.default_rng(seed)
    out_ids: list[int] = []
    prev_text = ""
    for _ in range(max_tokens):
        nxt = sample_logits(logits, temperature, top_p, top_k, rng,
                            repetition_penalty, out_ids)
        if engine.eos is not None and nxt == int(engine.eos):
            break
        out_ids.append(nxt)
        text = engine.decode(out_ids)
        delta = text[len(prev_text):]
        prev_text = text
        if any(s and s in text for s in stops):
            for s in stops:
                if s and s in text:
                    delta = text[:text.index(s)][len(prev_text) - len(delta):]
                    break
            if delta:
                yield nxt, delta
            return
        yield nxt, delta
        logits, state = engine.step(nxt, state)


def _messages_to_prompt(messages) -> str:
    """无 chat template 的基座模型: 单条 user 直取, 多条按 role 拼接."""
    if not messages:
        return ""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0].get("content", ""))
    parts = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


# ------------------------------------------------------------------- HTTP
class _Handler(BaseHTTPRequestHandler):
    engine: Engine = None
    server_version = "sunnynet/0.1.0"

    def log_message(self, fmt, *a):  # 静音默认 stderr 日志
        pass

    def _send(self, code: int, obj: dict, ctype="application/json"):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code: int, msg: str, etype="invalid_request_error"):
        self._send(code, {"error": {"message": msg, "type": etype, "code": code}})

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"请求体不是合法 JSON: {e}") from e

    # ------------------------------------------------------------- GET
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/health"):
            return self._send(200, {"status": "ok", "model": self.engine.name})
        if path == "/v1/models":
            return self._send(200, {
                "object": "list",
                "data": [{"id": self.engine.name, "object": "model",
                          "created": int(time.time()), "owned_by": "sunny-cx"}],
            })
        self._err(404, f"未知路径 {self.path}", "not_found")

    # ------------------------------------------------------------ POST
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            body = self._read_json()
        except ValueError as e:
            return self._err(400, str(e))
        try:
            if path == "/v1/completions":
                return self._completions(body, chat=False)
            if path == "/v1/chat/completions":
                return self._completions(body, chat=True)
            self._err(404, f"未知路径 {self.path}", "not_found")
        except Exception as e:  # 参数非法等一律回 400, 不能让连接直接断
            return self._err(400, f"{type(e).__name__}: {e}",
                             "invalid_request_error")

    def _completions(self, body, chat: bool):
        if chat:
            prompt = _messages_to_prompt(body.get("messages") or [])
        else:
            prompt = body.get("prompt") or ""
        if not prompt:
            return self._err(400, "prompt/messages 为空")

        try:
            max_tokens = int(body.get("max_tokens") or 64)
            _t = body.get("temperature")
            temperature = 1.0 if _t is None else float(_t)
            top_p = float(body.get("top_p") or 1.0)
            top_k = int(body.get("top_k") or 0)
            seed = body.get("seed")
            stream = bool(body.get("stream", False))
            stops = body.get("stop") or []
            if isinstance(stops, str):
                stops = [stops]
            _r = body.get("repetition_penalty")
            rep = 1.0 if _r is None else float(_r)
        except (TypeError, ValueError) as e:
            return self._err(400, f"参数类型错误: {e}", "invalid_request_error")
        if max_tokens < 0 or max_tokens > 8192:
            return self._err(400, "max_tokens 需在 0..8192", "invalid_request_error")
        if rep <= 0:
            return self._err(400, "repetition_penalty 需 > 0",
                             "invalid_request_error")

        try:
            ids = self.engine.encode(prompt)
        except Exception as e:
            return self._err(500, f"分词失败: {e}", "server_error")
        model_id = self.engine.name
        cid = f"cmpl-{int(time.time()*1000)}"

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            # 无 Content-Length 的流只能靠关闭连接收尾; HTTP/1.0 下声明
            # keep-alive 会让读到 EOF 的客户端一直挂住
            self.close_connection = True

            def sse(obj):
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
                                 .encode("utf-8"))
                self.wfile.flush()

            if chat:
                sse({"id": cid, "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": model_id,
                     "choices": [{"index": 0, "delta": {"role": "assistant"},
                                  "finish_reason": None}]})
            try:
                for _tok, delta in generate_stream(
                        self.engine, ids, max_tokens, temperature, top_p,
                        top_k, seed, stops, rep):
                    if chat:
                        sse({"id": cid, "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": model_id,
                             "choices": [{"index": 0,
                                          "delta": {"content": delta},
                                          "finish_reason": None}]})
                    else:
                        sse({"id": cid, "object": "text_completion",
                             "created": int(time.time()), "model": model_id,
                             "choices": [{"index": 0, "text": delta,
                                          "finish_reason": None}]})
            except Exception as e:
                sse({"error": {"message": str(e), "type": "server_error"}})
            done = ({"id": cid, "object": "chat.completion.chunk" if chat
                     else "text_completion", "created": int(time.time()),
                     "model": model_id,
                     "choices": [{"index": 0, "delta": {} if chat else None,
                                  "text": None if chat else "",
                                  "finish_reason": "stop"}]})
            sse(done)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        out_ids, text = [], ""
        for tok, _delta in generate_stream(self.engine, ids, max_tokens,
                                           temperature, top_p, top_k, seed,
                                           stops, rep):
            out_ids.append(tok)
            text = self.engine.decode(out_ids)
        for s in stops:
            if s and s in text:
                text = text[:text.index(s)]
        n = len(out_ids)
        usage = {"prompt_tokens": len(ids), "completion_tokens": n,
                 "total_tokens": len(ids) + n}

        if chat:
            self._send(200, {
                "id": cid, "object": "chat.completion",
                "created": int(time.time()), "model": model_id,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": usage,
            })
        else:
            self._send(200, {
                "id": cid, "object": "text_completion",
                "created": int(time.time()), "model": model_id,
                "choices": [{"index": 0, "finish_reason": "stop", "text": text}],
                "usage": usage,
            })


def serve(engine: Engine, host="127.0.0.1", port=8080):
    handler = type("BoundHandler", (_Handler,), {"engine": engine})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sunnynet-serve",
                                 description="OpenAI 兼容的 SunnyNet 本地服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("-m", "--model", default=None,
                    help=".gguf / .pt / HF 目录, 按后缀自动识别")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--gguf-backend", choices=["numpy", "c"], default="numpy")
    ap.add_argument("--quant", choices=["fp32", "int4"], default="fp32")
    ap.add_argument("--blob", default=None)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--hf-arch", choices=list(HF_ARCHES), default="mamba2")
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--threads", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--model-name", default=None,
                    help="对外暴露的模型名 (默认按权重名)")
    args = ap.parse_args(argv)

    _resolve_model_flags(args)
    t0 = time.perf_counter()
    engine = build_engine(args)
    if args.model_name:
        engine.name = args.model_name
    print(f"[sunnynet] 模型 {engine.name} vocab={engine.vocab} "
          f"加载 {time.perf_counter()-t0:.2f}s", flush=True)
    httpd = serve(engine, args.host, args.port)
    print(f"[sunnynet] 监听 http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())