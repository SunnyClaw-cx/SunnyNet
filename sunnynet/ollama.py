"""Ollama 原生 API 兼容层.

Ollama 的客户端与 SDK 打的是 /api/tags、/api/generate、/api/chat, 不是 OpenAI 的
/v1/*。本模块把同一份 Engine 暴露成 Ollama 的形状, 使 `ollama` 生态里的工具
(OLLAMA_HOST 指过来即可) 能直接用到 SunnyNet 的递推内核。

不改 Ollama 本体: Ollama 拉模型时仍走它自己的 GGUF 加载; 这里提供的是"把
SunnyNet 当 Ollama 服务端"的路径, 适合已经用 Ollama API 的应用零改动切换。

端点:
    GET  /api/tags        模型列表
    POST /api/generate    单轮补全 (stream 支持 NDJSON 逐行)
    POST /api/chat        多轮对话
    GET  /api/version     版本
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .serve import (Engine, HF_ARCHES, build_engine, generate_stream,
                    _messages_to_prompt, _resolve_model_flags)

__all__ = ["ollama_handler", "serve_ollama", "main"]

VERSION = "0.1.0"


class _OllamaHandler(BaseHTTPRequestHandler):
    engine: Engine = None
    server_version = f"sunnynet-ollama/{VERSION}"

    def log_message(self, fmt, *a):
        pass

    # ------------------------------------------------------------- 工具
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _ndjson_line(self, obj: dict):
        self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()

    def _err(self, code: int, msg: str):
        self._send(code, {"error": msg})

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            raise ValueError(f"请求体不是合法 JSON: {e}") from e

    def _model_tag(self) -> str:
        return getattr(self.engine, "name", "sunnynet")

    # ------------------------------------------------------------- GET
    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/api/version":
            return self._send(200, {"version": VERSION})
        if path == "/api/tags":
            name = self._model_tag()
            return self._send(200, {"models": [{
                "name": name, "model": name,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
                "size": 0,
                "details": {"family": "rwkv", "parameter_size": "",
                            "quantization_level": "unknown"},
            }]})
        if path in ("/", "/api"):
            return self._send(200, {"status": "sunnynet ollama-compatible",
                                    "version": VERSION})
        self._err(404, f"未知路径 {self.path}")

    # ------------------------------------------------------------- POST
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            body = self._read_json()
        except ValueError as e:
            return self._err(400, str(e))
        try:
            if path == "/api/generate":
                return self._generate(body)
            if path == "/api/chat":
                return self._chat(body)
            self._err(404, f"未知路径 {self.path}")
        except Exception as e:
            return self._err(400, f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------- 生成
    def _parse_opts(self, body) -> dict:
        o = body.get("options") or {}
        try:
            return {
                "max_tokens": int(o.get("num_predict", body.get("num_predict", 64))),
                "temperature": float(o.get("temperature", 0.8)),
                "top_p": float(o.get("top_p", 0.95)),
                "top_k": int(o.get("top_k", 40)),
                "repeat_penalty": float(o.get("repeat_penalty", 1.0)),
                "seed": o.get("seed", body.get("seed")),
                "stops": body.get("stop") or [],
            }
        except (TypeError, ValueError) as e:
            raise ValueError(f"options 类型错误: {e}") from e

    def _generate(self, body):
        prompt = body.get("prompt") or ""
        if not prompt:
            return self._err(400, "prompt 为空")
        opts = self._parse_opts(body)
        model = body.get("model") or self._model_tag()
        stream = bool(body.get("stream", True))
        ids = self.engine.encode(prompt)
        t0 = time.perf_counter()
        if not stream:
            text = ""
            for _tok, delta in generate_stream(
                    self.engine, ids, opts["max_tokens"], opts["temperature"],
                    opts["top_p"], opts["top_k"], opts["seed"], opts["stops"],
                    opts["repeat_penalty"]):
                text += delta
            return self._send(200, {
                "model": model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": text, "done": True,
                "done_reason": "stop",
                "total_duration": int((time.perf_counter() - t0) * 1e9),
                "eval_count": len(self.engine.encode(text)),
            })
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True
        n = 0
        for _tok, delta in generate_stream(
                self.engine, ids, opts["max_tokens"], opts["temperature"],
                opts["top_p"], opts["top_k"], opts["seed"], opts["stops"],
                opts["repeat_penalty"]):
            n += 1
            self._ndjson_line({
                "model": model,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": delta, "done": False,
            })
        self._ndjson_line({
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": "", "done": True, "done_reason": "stop",
            "eval_count": n,
            "total_duration": int((time.perf_counter() - t0) * 1e9),
        })

    def _chat(self, body):
        msgs = body.get("messages") or []
        if not msgs:
            return self._err(400, "messages 为空")
        opts = self._parse_opts(body)
        model = body.get("model") or self._model_tag()
        stream = bool(body.get("stream", True))
        prompt = _messages_to_prompt(msgs)
        ids = self.engine.encode(prompt)
        t0 = time.perf_counter()
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        def chunk(delta, done, reason=None):
            obj = {"model": model, "created_at": created,
                   "message": {"role": "assistant", "content": delta},
                   "done": done}
            if done:
                obj["done_reason"] = reason or "stop"
                obj["total_duration"] = int((time.perf_counter() - t0) * 1e9)
            return obj

        if not stream:
            text = ""
            for _tok, delta in generate_stream(
                    self.engine, ids, opts["max_tokens"], opts["temperature"],
                    opts["top_p"], opts["top_k"], opts["seed"], opts["stops"],
                    opts["repeat_penalty"]):
                text += delta
            return self._send(200, chunk(text, True))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True
        for _tok, delta in generate_stream(
                self.engine, ids, opts["max_tokens"], opts["temperature"],
                opts["top_p"], opts["top_k"], opts["seed"], opts["stops"],
                opts["repeat_penalty"]):
            self._ndjson_line(chunk(delta, False))
        self._ndjson_line(chunk("", True))


def serve_ollama(engine, host="127.0.0.1", port=11434):
    handler = type("BoundOllama", (_OllamaHandler,), {"engine": engine})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sunnynet-ollama",
        description="把 SunnyNet 暴露成 Ollama 原生 API (默认 11434)")
    ap.add_argument("-m", "--model", default=None, help=".gguf / .pt / HF 目录")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--gguf-backend", choices=["numpy", "c"], default="c")
    ap.add_argument("--quant", choices=["fp32", "int4"], default="int4")
    ap.add_argument("--blob", default=None)
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--hf-arch", choices=list(HF_ARCHES), default="mamba2")
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--threads", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11434)
    args = ap.parse_args(argv)

    _resolve_model_flags(args)
    eng = build_engine(args)
    httpd = serve_ollama(eng, args.host, args.port)
    print(f"[sunnynet-ollama] {eng.name} 监听 http://{args.host}:{args.port}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())