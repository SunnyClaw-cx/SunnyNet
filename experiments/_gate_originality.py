"""通道门控判决实验: 成对竞争 vs softmax vs 标量 vs 直通.

三项证据, 逐项独立子进程:
  1) 算子级 Jacobian: softmax 是否含 1/d 因子, 成对竞争是否 O(1) (直接测 ||J||).
  2) 模型级梯度: 各门控下嵌入梯度范数随层数的衰减.
  3) 质量: 同一 harness 短训练 val ppl.

用法: python -X utf8 experiments/_gate_originality.py <stage> [args]
  stage: jac | grad | train
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

VOCAB = 8192


def gate_pairwise(g, group):
    B, T, D = g.shape
    gg = g.view(B, T, D // group, group)
    w = torch.sigmoid(gg - torch.roll(gg, -1, dims=-1)) * \
        torch.sigmoid(gg - torch.roll(gg, -2, dims=-1))
    return w.reshape(B, T, D)


def gate_softmax(g, group):
    B, T, D = g.shape
    gg = g.view(B, T, D // group, group)
    return torch.softmax(gg, dim=-1).reshape(B, T, D)


def gate_scalar(g, group):
    return torch.sigmoid(g)


def gate_none(g, group):
    return g * 0.0 + 1.0


GATES = {"pairwise": gate_pairwise, "softmax": gate_softmax,
         "scalar": gate_scalar, "none": gate_none}


def stage_jac():
    """算子级: sum_i ||dw_i/dg||_2 对 d 的标度. softmax 若含 1/d, 该值随 d 下降;
    成对竞争若 O(1), 应基本不随 d 变."""
    out = {}
    for d in (16, 64, 128, 256, 512):
        g = torch.randn(1, 1, d, dtype=torch.float64)
        row = {}
        for name, fn in GATES.items():
            gp = g.detach().clone().requires_grad_(True)
            Jm = torch.autograd.functional.jacobian(
                lambda x: GATES[name](x.view(1, 1, -1), 8)[0, 0], gp[0, 0])
            row[name] = {
                "spec_norm": float(torch.linalg.matrix_norm(Jm, 2)),
                "mean_row_sum": float(Jm.abs().sum(dim=1).mean()),
            }
        out[d] = row
        print(f"d={d:4d}: " + "  ".join(
            f"{k:8s}(||J||2={v['spec_norm']:.3g})" for k, v in row.items()))
    return out


def _cfg(dim, layers):
    from sunnynet.model import SunnyConfig
    return SunnyConfig(vocab_size=VOCAB, dim=dim, inner=dim * 3 // 2, rank=dim // 2,
                       layers=layers, h_max=2048.0, scan_chunk=64)


def stage_grad():
    """模型级: 嵌入梯度范数 vs 层数, 每个门控各测一次."""
    from sunnynet import layers as L
    orig = L.SunnyBlock._compete
    out = {}
    try:
        for name, fn in GATES.items():
            def patched(self, gate, _fn=fn):
                return _fn(gate, self.group)
            L.SunnyBlock._compete = patched
            curve = {}
            for nlay in (2, 4, 8, 16):
                torch.manual_seed(7)
                from sunnynet.model import SunnyNet
                m = SunnyNet(_cfg(128, nlay))
                idx = torch.randint(0, VOCAB, (2, 128))
                loss = torch.nn.functional.cross_entropy(
                    m(idx).reshape(-1, VOCAB), idx.reshape(-1))
                m.zero_grad(set_to_none=True)
                loss.backward()
                g = m.embed.weight.grad
                curve[nlay] = float(g.norm()) if g is not None else 0.0
            out[name] = curve
            vals = "  ".join(f"L{k}:{v:.3g}" for k, v in curve.items())
            print(f"{name:9s} {vals}")
    finally:
        L.SunnyBlock._compete = orig
    return out


def stage_train(steps=600, dim=128, layers=4, B=8, T=256, seed=1234, lr=3e-3):
    """质量: 同一 harness, 换门控, 短训练 val ppl."""
    import numpy as np
    from sunnynet import layers as L
    tr = np.memmap(ROOT / "data/proc/tokens_zh_ancient.bin", dtype=np.uint16, mode="r")
    va = np.memmap(ROOT / "data/proc/val_zh_ancient.bin", dtype=np.uint16, mode="r")
    tr = torch.from_numpy(np.asarray(tr, dtype=np.int64))
    va = torch.from_numpy(np.asarray(va, dtype=np.int64))

    torch.set_num_threads(6)
    from train.train import lr_at
    from sunnynet.model import SunnyNet

    orig = L.SunnyBlock._compete
    out = {}
    try:
        for name, fn in GATES.items():
            def patched(self, gate, _fn=fn):
                return _fn(gate, self.group)
            L.SunnyBlock._compete = patched
            torch.manual_seed(seed)
            m = SunnyNet(_cfg(dim, layers))
            opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
            for step in range(steps):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, steps, lr, max(1, steps // 20))
                i = int(torch.randint(0, len(tr) - B * T - 2, (1,)))
                x = tr[i:i + B * T].reshape(B, T)
                y = tr[i + 1:i + 1 + B * T].reshape(B, T)
                loss = torch.nn.functional.cross_entropy(
                    m(x).reshape(-1, VOCAB), y.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            m.eval()
            with torch.no_grad():
                tot, n = 0.0, 0
                for s in range(0, len(va) - T - 1, T * 4):
                    x = va[s:s + T].reshape(1, T)
                    y = va[s + 1:s + 1 + T].reshape(1, T)
                    tot += float(torch.nn.functional.cross_entropy(
                        m(x).reshape(-1, VOCAB), y.reshape(-1), reduction="sum"))
                    n += T
                ppl = float(torch.exp(torch.tensor(tot / n)))
            out[name] = {"val_ppl": ppl}
            print(f"{name:9s} val_ppl={ppl:.3f}")
    finally:
        L.SunnyBlock._compete = orig
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "jac"
    resdir = ROOT / "experiments/results"
    resdir.mkdir(parents=True, exist_ok=True)
    if stage == "jac":
        res = stage_jac()
        (resdir / "_gate_originality_jac.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    elif stage == "grad":
        res = stage_grad()
        (resdir / "_gate_originality_grad.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 600
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1234
        res = stage_train(steps=steps, seed=seed)
        out = resdir / f"_gate_originality_train_s{seed}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out}")
    print("RESULT " + json.dumps(res, ensure_ascii=False))