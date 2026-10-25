"""验证 karpathy 转换: 手写 LLaMA 前向 vs HF, 逐 token 比 logits."""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def manual_forward(sd, ids, n_layers, n_heads, dim, eps=1e-5, theta=10000.0):
    T = len(ids)
    x = sd["tok_embeddings.weight"][torch.tensor(ids)]
    hd = dim // n_heads
    freqs = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(T).float()
    ang = torch.outer(t, freqs)
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]

    def rope(v):
        v = v.view(1, T, n_heads, hd)
        v1, v2 = v[..., ::2], v[..., 1::2]
        r1 = v1 * cos - v2 * sin
        r2 = v1 * sin + v2 * cos
        return torch.stack([r1, r2], dim=-1).flatten(-2)

    mask = torch.triu(torch.full((T, T), float("-inf")), 1)
    for i in range(n_layers):
        p = f"layers.{i}."
        h = F.rms_norm(x, (dim,), sd[p + "attention_norm.weight"], eps)
        q = rope(h @ sd[p + "attention.wq.weight"].T)
        k = rope(h @ sd[p + "attention.wk.weight"].T)
        v = h @ sd[p + "attention.wv.weight"].T
        q = q.view(1, T, n_heads, hd).transpose(1, 2)
        k = k.view(1, T, n_heads, hd).transpose(1, 2)
        v = v.view(1, T, n_heads, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd) + mask
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(1, T, dim)
        x = x + o @ sd[p + "attention.wo.weight"].T
        h = F.rms_norm(x, (dim,), sd[p + "ffn_norm.weight"], eps)
        g = h @ sd[p + "feed_forward.w1.weight"].T
        u = h @ sd[p + "feed_forward.w3.weight"].T
        x = x + (F.silu(g) * u) @ sd[p + "feed_forward.w2.weight"].T
    x = F.rms_norm(x, (dim,), sd["norm.weight"], eps)
    return x @ sd["output.weight"].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="karpathy stories15M.pt 路径")
    ap.add_argument("--hf", required=True, help="转换后的 HF 目录")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ck["model"]
    a = ck["model_args"]
    from transformers import AutoModelForCausalLM
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(Path(args.hf) / "tokenizer.model"))
    m = AutoModelForCausalLM.from_pretrained(args.hf,
                                             dtype=torch.float32, local_files_only=True).eval()
    text = "Once upon a time, there was a little girl named Lily. She loved to play in the garden with her friends."
    ids = sp.encode(text)
    x = torch.tensor([ids])
    with torch.no_grad():
        hf = m(x).logits[0]
        man = manual_forward(sd, ids, a["n_layers"], a["n_heads"], a["dim"])[0]
    d = (hf - man).abs()
    print(f"tokens={len(ids)} hf/manual max|diff|={d.max():.2e} mean={d.mean():.2e}")
    print("hf argmax:", hf.argmax(-1).tolist()[:10])
    print("ma argmax:", man.argmax(-1).tolist()[:10])


if __name__ == "__main__":
    sys.exit(main())