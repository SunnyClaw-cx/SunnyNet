"""把 karpathy/tinyllamas 的 nanoGPT checkpoint 转成 HF Llama 格式目录.

源: stories15M/42M/110M.pt (dim 288/384/768, LLaMA 架构, llama2 32K BPE).
只为让这些模型能进 _ext_model_bench.py 的统一评测, 不发布转换产物。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def permute_rope(w: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """llama2.c 的 RoPE 按相邻元素对旋转, HF Llama 按前后两半旋转。
    同一组权重在两种约定下不等价, 须对 q/k 投影做交错置换, 否则 logits 全错。"""
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert(src: Path, dst: Path, tok_src: Path):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    a = ck["model_args"]
    sd = ck["model"]
    dim, n_layers = a["dim"], a["n_layers"]
    n_heads, n_kv = a["n_heads"], a["n_kv_heads"]
    inter = sd["layers.0.feed_forward.w1.weight"].shape[0]

    out = {}
    out["model.embed_tokens.weight"] = sd["tok_embeddings.weight"]
    for i in range(n_layers):
        p = f"layers.{i}."
        q = f"model.layers.{i}."
        out[q + "self_attn.q_proj.weight"] = permute_rope(sd[p + "attention.wq.weight"], n_heads, dim, dim)
        out[q + "self_attn.k_proj.weight"] = permute_rope(sd[p + "attention.wk.weight"], n_kv, dim, dim)
        out[q + "self_attn.v_proj.weight"] = sd[p + "attention.wv.weight"]
        out[q + "self_attn.o_proj.weight"] = sd[p + "attention.wo.weight"]
        out[q + "mlp.gate_proj.weight"] = sd[p + "feed_forward.w1.weight"]
        out[q + "mlp.down_proj.weight"] = sd[p + "feed_forward.w2.weight"]
        out[q + "mlp.up_proj.weight"] = sd[p + "feed_forward.w3.weight"]
        out[q + "input_layernorm.weight"] = sd[p + "attention_norm.weight"]
        out[q + "post_attention_layernorm.weight"] = sd[p + "ffn_norm.weight"]
    out["model.norm.weight"] = sd["norm.weight"]
    # nanoGPT 把 output 和 tok_embeddings 绑同一张量, safetensors 拒绝共享
    out["lm_head.weight"] = sd["output.weight"].clone()

    dst.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.contiguous().to(torch.float32) for k, v in out.items()},
              str(dst / "model.safetensors"))

    cfg = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": dim, "intermediate_size": inter,
        "num_hidden_layers": n_layers, "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv, "head_dim": dim // n_heads,
        "max_position_embeddings": a["max_seq_len"], "vocab_size": a["vocab_size"],
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "hidden_act": "silu", "tie_word_embeddings": False,
        "torch_dtype": "float32", "bos_token_id": 1, "eos_token_id": 2,
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for f in ["tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(tok_src / f, dst / f)

    n = sum(v.numel() for v in out.values())
    print(f"{src.name} -> {dst}  params={n:,} (h={dim} L={n_layers} inter={inter})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", required=True, help="tinyllamas snapshot 目录")
    ap.add_argument("--tok", required=True, help="含 tokenizer.model 的目录")
    ap.add_argument("--models", default="stories15M,stories42M,stories110M")
    ap.add_argument("--out-root", default="")
    args = ap.parse_args()

    hub = Path(args.hub)
    out_root = Path(args.out_root) if args.out_root else ROOT / "experiments" / "ext_models"
    for m in args.models.split(","):
        src = hub / f"{m}.pt"
        if not src.exists():
            print("missing", src)
            continue
        convert(src, out_root / m, Path(args.tok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())