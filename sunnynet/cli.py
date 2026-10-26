"""sunnynet-cli: 加载 SunnyNet 权重, 在 CPU 上自回归生成文本.

    python -m sunnynet.cli --ckpt runs/small.pt --prompt "子曰：" --max-new 80

挂点: 权重加载 + 分词器 + 自回归解码循环。递推状态走 SunnyNet 的
forward_step (与整段前向逐位一致), --runtime 可把单步递推交给独立 C 内核。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from .model import SunnyConfig, SunnyNet
from .serve import HF_ARCHES
from .tokenizer import BPETokenizer

__all__ = ["main", "load_model", "decode_stream"]

REPO = Path(__file__).resolve().parent.parent
TOKENIZER = REPO / "data" / "proc" / "tokenizer.json"


def load_model(ckpt: str, arch: str | None = None):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = arch or d.get("arch", "sunnynet")
    if a != "sunnynet":
        raise SystemExit(f"本 CLI 只跑 sunnynet 权重, 收到 arch={a}")
    cfg = SunnyConfig(**d["config"])
    m = SunnyNet(cfg)
    m.load_state_dict(d["model"])
    m.eval()
    return m, cfg, d.get("summary", {})


def _sample(last: torch.Tensor, temperature: float, top_k: int, top_p: float):
    if temperature <= 0:
        return int(last.argmax())
    last = last / temperature
    if top_k > 0:
        v, _ = torch.topk(last, min(top_k, last.numel()))
        last = last.masked_fill(last < v[-1], float("-inf"))
    if 0.0 < top_p < 1.0:
        srt, idx = torch.sort(last, descending=True)
        cum = torch.softmax(srt, dim=-1).cumsum(-1)
        cut = cum > top_p
        cut[1:] = cut[:-1].clone()
        cut[0] = False
        srt = srt.masked_fill(cut, float("-inf"))
        last = torch.full_like(last, float("-inf")).scatter(0, idx, srt)
    return int(torch.multinomial(torch.softmax(last, dim=-1), 1))


@torch.no_grad()
def decode_stream(model, tk, prompt: str, max_new: int = 80,
                  temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95,
                  seed: int | None = None, verbose: bool = False,
                  runtime=None):
    """逐 token 生成, yield 新 token 的文本片段; 返回 (全文, 统计).

    runtime 给定时, 解码走独立 C 内核 (整模型单步), 采样与分词仍在 torch。
    """
    if seed is not None:
        torch.manual_seed(seed)
    ids = tk.encode(prompt) or [tk.bos_id]
    t0 = time.perf_counter()
    if runtime is not None:
        logits = torch.tensor(runtime.prefill(ids))
    else:
        s = model.init_state(1)
        logits, s = model(torch.tensor([ids], dtype=torch.long), return_state=True)
        logits = logits[0, -1]
    prefill_s = time.perf_counter() - t0
    out = list(ids)
    last = logits
    step_times = []
    for _ in range(max_new):
        nxt = _sample(last, temperature, top_k, top_p)
        out.append(nxt)
        piece = tk.decode([nxt])
        yield piece
        if nxt == tk.eos_id:
            break
        t0 = time.perf_counter()
        if runtime is not None:
            last = torch.tensor(runtime.step(nxt))
        else:
            last, s = model.forward_step(torch.tensor([nxt]), s)
            last = last[0]
        step_times.append(time.perf_counter() - t0)
    stats = {
        "prefill_tokens": len(ids),
        "prefill_ms": prefill_s * 1e3,
        "backend": "c" if runtime is not None else "torch",
        "gen_tokens": len(step_times),
        "decode_ms_median": (sorted(step_times)[len(step_times) // 2] * 1e3
                             if step_times else 0.0),
        "decode_ms_per_token": (sum(step_times) / len(step_times) * 1e3
                                if step_times else 0.0),
    }
    decode_stream.stats = stats
    decode_stream.text = tk.decode(out)
    decode_stream.state = None


def _main_gguf(args) -> int:
    """外部 GGUF 权重路径 (RWKV-7), 不经 PyTorch."""
    from .runtime.rwkv7 import RWKV7
    t0 = time.perf_counter()
    m = RWKV7(args.gguf)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    text, ids = m.generate(args.prompt, max_new=args.max_new,
                           temperature=args.temperature, top_k=args.top_k,
                           seed=args.seed)
    gen_s = time.perf_counter() - t0
    st = {
        "backend": "gguf/rwkv7",
        "load_ms": load_s * 1e3,
        "gen_tokens": len(ids),
        "decode_ms_per_token": gen_s / max(len(ids), 1) * 1e3,
    }
    if args.json:
        print(json.dumps({"prompt": args.prompt, "text": text,
                          "arch": "rwkv7", "dim": m.n_embd, "layers": m.n_layer,
                          "stats": st}, ensure_ascii=False, indent=2))
    else:
        sys.stdout.write(text)
        print()
        if args.stats:
            print(f"[rwkv7 {m.n_layer}L d{m.n_embd} vocab {m.vocab} "
                  f"backend gguf/numpy]")
            print(f"[load {st['load_ms']:.1f} ms; {st['gen_tokens']} tok "
                  f"{st['decode_ms_per_token']:.2f} ms/tok (numpy 路径)]")
    return 0


def _main_gguf_c(args) -> int:
    """GGUF 权重走 C 内核整模型单步 (RWKV-7)."""
    import numpy as np
    from .runtime.rwkv7 import RWKV7
    from .runtime.rwkv7_c import SnRWKV7

    # 未显式给 blob 时按需导出并缓存: 首次会花一次反量化的时间, 之后直接命中.
    if args.blob:
        blob = Path(args.blob)
        if not blob.exists():
            raise SystemExit(f"缺少权重 blob: {blob}")
    else:
        from .runtime.blob import ensure_rwkv7_blob
        blob = ensure_rwkv7_blob(args.gguf, quant=args.quant)
    t0 = time.perf_counter()
    tk = RWKV7(args.gguf)
    m = SnRWKV7(blob, tk.vocab)
    load_s = time.perf_counter() - t0

    ids = tk.tok.encode(args.prompt)
    t0 = time.perf_counter()
    logits = np.array(m.prefill(ids), dtype=np.float32)
    prefill_s = time.perf_counter() - t0
    out = list(ids)
    step_times = []
    for _ in range(args.max_new):
        nxt = int(np.argmax(logits)) if args.temperature <= 0 else \
            _sample_np(logits, args.temperature, args.top_k)
        out.append(nxt)
        if len(out) > args.max_new + len(ids):
            break
        t0 = time.perf_counter()
        logits = np.array(m.step(nxt), dtype=np.float32)
        step_times.append(time.perf_counter() - t0)
    text = tk.tok.decode(out)
    st = {
        "backend": "gguf/rwkv7/c",
        "load_ms": load_s * 1e3,
        "prefill_tokens": len(ids),
        "prefill_ms": prefill_s * 1e3,
        "gen_tokens": len(step_times),
        "decode_ms_per_token": (sum(step_times) / len(step_times) * 1e3
                                if step_times else 0.0),
    }
    if args.json:
        print(json.dumps({"prompt": args.prompt, "text": text,
                          "arch": "rwkv7", "stats": st}, ensure_ascii=False, indent=2))
    else:
        sys.stdout.write(text)
        print()
        if args.stats:
            print(f"[rwkv7 backend gguf/c]")
            print(f"[load {st['load_ms']:.1f} ms; prefill {st['prefill_tokens']} tok "
                  f"{st['prefill_ms']:.1f} ms; decode {st['gen_tokens']} tok "
                  f"{st['decode_ms_per_token']:.2f} ms/tok (C 内核)]")
    return 0


def _main_rwkv7_blob(args) -> int:
    """RWKV-7 blob 直接走 C 内核 (权重来自 GGUF 或 FLA safetensors 导出)。

    分词用 RWKV 官方词表文件 (rwkv_vocab_v20230424.txt), 不依赖 GGUF。
    """
    import numpy as np
    from .runtime.rwkv7_c import SnRWKV7
    from .runtime.vocab import RWKVVocab

    blob = Path(args.blob)
    if not blob.exists():
        raise SystemExit(f"缺少权重 blob: {blob}")
    vocab_file = Path(args.tokenizer_dir) / "rwkv_vocab_v20230424.txt" \
        if args.tokenizer_dir else None
    if vocab_file is None or not vocab_file.exists():
        alt = Path(args.blob).parent / "rwkv_vocab_v20230424.txt"
        vocab_file = alt if alt.exists() else None

    t0 = time.perf_counter()
    tk = RWKVVocab(str(vocab_file)) if vocab_file else None
    m = SnRWKV7(str(blob), args.vocab or 65536)
    load_s = time.perf_counter() - t0

    ids = tk.encode(args.prompt) if tk is not None else \
        [1] + [b + 2 for b in args.prompt.encode("utf-8")]
    m.reset()
    logits = None
    for t in ids:
        logits = m.step_np(int(t))
    out = list(ids)
    step_times = []
    for _ in range(args.max_new):
        nxt = (int(np.argmax(logits)) if args.temperature <= 0
               else _sample_np(logits, args.temperature, args.top_k))
        out.append(nxt)
        t0 = time.perf_counter()
        logits = m.step_np(nxt)
        step_times.append(time.perf_counter() - t0)
    text = tk.decode(out) if tk is not None else ""
    st = {
        "backend": "rwkv7/c",
        "load_ms": load_s * 1e3,
        "prefill_tokens": len(ids),
        "gen_tokens": len(step_times),
        "decode_ms_per_token": (sum(step_times) / len(step_times) * 1e3
                                if step_times else 0.0),
    }
    if args.json:
        print(json.dumps({"prompt": args.prompt, "text": text,
                          "arch": "rwkv7", "stats": st},
                         ensure_ascii=False, indent=2))
    else:
        sys.stdout.write(text)
        print()
        if args.stats:
            print(f"[rwkv7 blob backend rwkv7/c]")
            print(f"[load {st['load_ms']:.1f} ms; prefill {st['prefill_tokens']} tok; "
                  f"decode {st['gen_tokens']} tok "
                  f"{st['decode_ms_per_token']:.2f} ms/tok]")
    return 0


def _dn_vocab(blob: str, hf_dir: str) -> int:
    """从 blob 头部读 vocab (避免为拿一个数加载整个 HF 权重)。"""
    from .runtime.blob import read_blob_vocab
    return read_blob_vocab(blob)


class _CAdapter:
    """把整模型 C 句柄包成 CLI 期望的 step(token, state) 接口 (state 由 C 侧持有)。"""

    def __init__(self, handle, hf_dir: str):
        self.h = handle
        self.hf_dir = hf_dir
        self.n_layers = _dn_layers(hf_dir)

    def init_state(self):
        return None

    def reset(self, state=None):
        self.h.reset()
        return None

    def step(self, token, state=None):
        import numpy as np
        sn = getattr(self.h, "step_np", None)
        if sn is not None:
            return sn(int(token))
        return np.asarray(self.h.step(int(token)), dtype=np.float32)


def _use_c_backend(args) -> bool:
    """--blob 配合 --hf-arch 时走整模型 C 内核 (与 serve.HFCEngine 同一判据)."""
    return bool(args.blob) and args.hf_arch in HF_ARCHES


def _dn_layers(hf_dir: str) -> int:
    import json
    p = os.path.join(hf_dir, "config.json")
    try:
        with open(p, encoding="utf-8") as f:
            c = json.load(f)
    except Exception:
        return 0
    for k in ("num_hidden_layers", "n_layer"):
        if k in c:
            return int(c[k])
    return 0


def _main_hf(args) -> int:
    """HF 权重目录路径 (Mamba-2), 不经 PyTorch 前向."""
    import numpy as np

    if args.hf_arch == "mamba2":
        from .runtime.mamba2 import Mamba2 as Model
    else:
        raise SystemExit(f"未支持的 --hf-arch: {args.hf_arch}")

    t0 = time.perf_counter()
    c_arch = args.hf_arch if _use_c_backend(args) else None
    if c_arch == "mamba2":
        from .runtime.mamba2_c import SnMamba2
        m = _CAdapter(SnMamba2(args.blob, _dn_vocab(args.blob, args.hf_dir),
                               threads=args.threads), args.hf_dir)
    else:
        m = Model(args.hf_dir)
    load_s = time.perf_counter() - t0

    tk = None
    if args.prompt:
        try:
            from transformers import AutoTokenizer
            src = args.tokenizer_dir or args.hf_dir
            tk = AutoTokenizer.from_pretrained(src)
            ids = tk.encode(args.prompt, add_special_tokens=True)
        except Exception as e:
            # 部分 checkpoint (如 state-spaces/mamba2) 不含 tokenizer 文件
            print(f"[warn] tokenizer 加载失败, 改按字节编码: {type(e).__name__}",
                  file=sys.stderr)
            tk = None
            ids = [1] + [b + 2 for b in args.prompt.encode("utf-8")]
    else:
        ids = [1]

    state = m.reset(m.init_state())
    t0 = time.perf_counter()
    logits = None
    for t in ids:
        logits = m.step(int(t), state)
    prefill_s = time.perf_counter() - t0

    out = list(ids)
    step_times = []
    for _ in range(args.max_new):
        nxt = (int(np.argmax(logits)) if args.temperature <= 0
               else _sample_np(logits, args.temperature, args.top_k))
        out.append(nxt)
        t0 = time.perf_counter()
        logits = m.step(nxt, state)
        step_times.append(time.perf_counter() - t0)
    text = tk.decode(out) if tk is not None else ""
    st = {
        "backend": f"hf/{args.hf_arch}/c" if c_arch else f"hf/{args.hf_arch}",
        "load_ms": load_s * 1e3,
        "prefill_tokens": len(ids),
        "prefill_ms": prefill_s * 1e3,
        "gen_tokens": len(step_times),
        "decode_ms_per_token": (sum(step_times) / len(step_times) * 1e3
                                if step_times else 0.0),
    }
    if args.json:
        print(json.dumps({"prompt": args.prompt, "text": text,
                          "arch": args.hf_arch, "stats": st},
                         ensure_ascii=False, indent=2))
    else:
        sys.stdout.write(text)
        print()
        if args.stats:
            print(f"[{args.hf_arch} L{m.n_layers} backend {st['backend']}]")
            print(f"[load {st['load_ms']:.1f} ms; prefill {st['prefill_tokens']} tok "
                  f"{st['prefill_ms']:.1f} ms; decode {st['gen_tokens']} tok "
                  f"{st['decode_ms_per_token']:.2f} ms/tok]")
    return 0


def _sample_np(logits, temperature: float, top_k: int) -> int:
    import numpy as np
    z = logits.astype(np.float64) / temperature
    if top_k and top_k < z.size:
        thr = np.partition(z, -top_k)[-top_k]
        z = np.where(z < thr, -np.inf, z)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return int(np.random.default_rng().choice(z.size, p=p))


def _main_sub(sub: str, argv: list) -> int:
    if sub == "chat":
        return _cmd_chat(argv)
    if sub == "serve":
        return _cmd_serve(argv)
    if sub == "convert":
        return _cmd_convert(argv)
    if sub == "quantize":
        return _cmd_quantize(argv)
    if sub == "info":
        return _cmd_info(argv)
    raise SystemExit(f"未知子命令 {sub!r}")


def _cmd_chat(argv) -> int:
    """交互式聊天; 管道输入或 --prompt 时退化为单轮."""
    ap = argparse.ArgumentParser(prog="sunnynet chat",
                                 description="交互式聊天 (流式输出)")
    _add_infer_args(ap, gguf_backend_default="c")
    ap.add_argument("--system", default=None, help="system prompt")
    ap.add_argument("--ctx", type=int, default=8,
                    help="保留的历史轮数 (每轮含 user+assistant)")
    args = ap.parse_args(argv)
    _resolve_model_flags(args)
    if not (args.gguf or args.ckpt or args.hf_dir):
        raise SystemExit("需要 -m/--model 或 --gguf/--ckpt/--hf-dir")

    from .serve import build_engine, generate_stream, _messages_to_prompt
    eng = build_engine(args)

    def one_round(messages):
        prompt = _messages_to_prompt(messages)
        ids = eng.encode(prompt)
        pieces = []
        for _tid, piece in generate_stream(eng, ids, args.max_new,
                                           args.temperature, args.top_p,
                                           args.top_k, args.seed, [],
                                           args.repetition_penalty):
            pieces.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
        return "".join(pieces)

    history: list = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    # 一次性模式: --prompt 或 stdin 非 tty
    if args.prompt:
        print(one_round(history + [{"role": "user", "content": args.prompt}]))
        return 0
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            print(one_round(history + [{"role": "user", "content": text}]))
            return 0
        return 0

    print("SunnyNet chat; 空行退出, /reset 清空上下文")
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            break
        if line == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print("[context cleared]")
            continue
        history.append({"role": "user", "content": line})
        reply = one_round(history)
        print()
        history.append({"role": "assistant", "content": reply})
        if args.ctx > 0:
            sys_msgs = [h for h in history if h["role"] == "system"]
            rest = [h for h in history if h["role"] != "system"]
            history = sys_msgs + rest[-2 * args.ctx:]
    return 0


def _cmd_serve(argv) -> int:
    ap = argparse.ArgumentParser(prog="sunnynet serve",
                                 description="OpenAI 兼容 HTTP 服务")
    _add_infer_args(ap, gguf_backend_default="c")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model-name", default=None,
                    help="对外暴露的模型名 (默认按权重名)")
    args = ap.parse_args(argv)
    _resolve_model_flags(args)
    if not (args.gguf or args.ckpt or args.hf_dir):
        raise SystemExit("需要 -m/--model 或 --gguf/--ckpt/--hf-dir")
    from .serve import build_engine, serve
    eng = build_engine(args)
    if args.model_name:
        eng.name = args.model_name
    print(f"[sunnynet] 模型 {eng.name} vocab={eng.vocab}", flush=True)
    httpd = serve(eng, args.host, args.port)
    print(f"[sunnynet] 监听 http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def _cmd_convert(argv) -> int:
    """把 GGUF 权重转成 SunnyNet 优化过的 GGUF (加自定义标记 + 预量化)."""
    from .runtime.convert import convert_main
    return convert_main(argv)


def _cmd_quantize(argv) -> int:
    """GGUF -> GGUF 量化 (Q4_K_M / Q5_K_M / Q8_0 ...), 附困惑度评测."""
    from .runtime.quantize import quantize_main
    return quantize_main(argv)


def _cmd_info(argv) -> int:
    """打印权重信息; 对 GGUF 显示架构/层数/词表/张量数."""
    ap = argparse.ArgumentParser(prog="sunnynet info")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    p = Path(args.model)
    info: dict = {"file": str(p), "bytes": p.stat().st_size,
                  "mb": round(p.stat().st_size / 2**20, 1)}
    if p.suffix == ".gguf":
        from .runtime.gguf import GGUFModel
        g = GGUFModel(p)
        arch = g.meta.get("general.architecture") or g.meta.get("architecture")
        info.update({
            "format": "gguf",
            "architecture": arch,
            "tensors": len(g.names()),
            "vocab": len(g.tokens()),
            "block_count": g.get_int("rwkv7.block_count", 0),
            "embedding_length": g.get_int("rwkv7.embedding_length", 0),
        })
        ftype = g.meta.get("general.file_type")
        if ftype is not None:
            info["file_type"] = ftype
    else:
        info["format"] = p.suffix.lstrip(".") or "unknown"
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        for k, v in info.items():
            print(f"{k:18s} {v}")
    return 0


def _add_infer_args(ap, *, gguf_backend_default: str = "numpy"):
    """推理相关参数, chat / serve / 顶层旧接口共用."""
    ap.add_argument("-m", "--model", default=None,
                    help="权重路径: .gguf / .pt / HF 目录, 按后缀自动识别")
    ap.add_argument("--ckpt", default=None, help="SunnyNet checkpoint 路径")
    ap.add_argument("--gguf", default=None,
                    help="外部 GGUF 权重路径 (支持 RWKV-7), 与 --ckpt 二选一")
    ap.add_argument("--prompt", default="", help="提示词")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="对已出现 token 的 logits 惩罚 (>1 抑制重复)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tokenizer", default=str(TOKENIZER))
    ap.add_argument("--tokenizer-dir", default=None,
                    help="--hf-dir 时另指定 tokenizer 来源目录")
    ap.add_argument("--threads", type=int, default=0, help="0 = 不限制")
    ap.add_argument("--backend", choices=["torch", "c"], default="torch",
                    help="c = 纯 C 整模型单步 (需先导出 .runtime.bin)")
    ap.add_argument("--gguf-backend", choices=["numpy", "c"],
                    default=gguf_backend_default,
                    help="--gguf 时的执行后端; c = C 内核整模型单步")
    ap.add_argument("--blob", default=None,
                    help="C 内核权重 blob: RWKV-7 的 --gguf-backend c 用它, "
                         "或 --hf-arch mamba2 时指向对应的 .c.bin "
                         "(整模型前向在 C); 单独给出时按 RWKV-7 blob 直接推理")
    ap.add_argument("--vocab", type=int, default=0,
                    help="单独用 --blob 时的词表大小 (默认 65536)")
    ap.add_argument("--quant", choices=["fp32", "int4"], default="int4",
                    help="--gguf-backend c 的权重精度; int4 需对应的 .q4.bin")
    ap.add_argument("--half", action="store_true",
                    help="--backend c 时把 GEMV 权重与 head 载入为 fp16; "
                         "读字节减半, 单步约快 1.3x, logits 相对偏差约 7e-4")
    ap.add_argument("--hf-dir", default=None,
                    help="HF 权重目录 (Mamba-2), 与 --ckpt/--gguf 三选一")
    ap.add_argument("--hf-arch", choices=list(HF_ARCHES),
                    default="mamba2",
                    help="--hf-dir 的架构")
    ap.add_argument("--stats", action="store_true", help="输出解码耗时统计")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")


def _resolve_model_flags(args) -> None:
    from .serve import _resolve_model_flags as impl
    impl(args)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # 无子命令时保持旧行为 (顶层 flag 直接推理)
    subcmds = {"chat", "serve", "convert", "quantize", "info"}
    if argv and argv[0] in subcmds:
        return _main_sub(argv[0], argv[1:])

    ap = argparse.ArgumentParser(prog="sunnynet-cli",
                                 description="SunnyNet CPU 推理")
    _add_infer_args(ap)
    args = ap.parse_args(argv)
    _resolve_model_flags(args)

    if args.hf_dir:
        return _main_hf(args)
    if args.gguf:
        if args.gguf_backend == "c":
            return _main_gguf_c(args)
        return _main_gguf(args)
    if args.blob and not args.ckpt:
        return _main_rwkv7_blob(args)
    if not args.ckpt:
        raise SystemExit("需要 --ckpt / --gguf / --hf-dir / --blob / -m 之一")

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    tkp = Path(args.tokenizer)
    if not tkp.exists():
        raise SystemExit(f"分词器不存在: {tkp}")
    tk = BPETokenizer.load(str(tkp))
    model, cfg, summary = load_model(args.ckpt)
    n = model.num_params()

    runtime = None
    if args.backend == "c":
        from .runtime.model import SnModel, threads_for_blob
        blob = Path(args.blob) if args.blob else Path(args.ckpt).with_suffix(".runtime.bin")
        if not blob.exists():
            raise SystemExit(f"缺少权重 blob: {blob}\n"
                             f"先运行: python -X utf8 experiments/_export_runtime.py "
                             f"{args.ckpt} {blob}")
        # 小模型单线程最快, 大模型用物理核; --threads 显式指定时以用户为准.
        thr = args.threads if args.threads > 0 else threads_for_blob(blob)
        runtime = SnModel(blob, cfg.vocab_size, cfg.dim, cfg.inner, cfg.rank,
                          cfg.taps, cfg.layers, 8, int(cfg.tie_embeddings),
                          half=args.half, threads=thr)

    pieces = []
    for p in decode_stream(model, tk, args.prompt, args.max_new,
                           args.temperature, args.top_k, args.top_p,
                           args.seed, verbose=True, runtime=runtime):
        pieces.append(p)
        sys.stdout.write(p)
        sys.stdout.flush()
    st = getattr(decode_stream, "stats", {})

    if args.json:
        print()
        print(json.dumps({"prompt": args.prompt, "text": "".join(pieces),
                          "params": n, "dim": cfg.dim, "layers": cfg.layers,
                          "stats": st}, ensure_ascii=False, indent=2))
    elif args.stats:
        print()
        print(f"[params {n/1e6:.2f}M dim {cfg.dim} layers {cfg.layers} "
              f"backend {st.get('backend', 'torch')}]")
        print(f"[prefill {st['prefill_tokens']} tok {st['prefill_ms']:.1f} ms; "
              f"decode {st['gen_tokens']} tok "
              f"{st['decode_ms_per_token']:.2f} ms/tok]")
    else:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())