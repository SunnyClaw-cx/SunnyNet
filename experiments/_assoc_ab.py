"""关联记忆写规则 A/B: 软凸组合 vs delta 覆盖.

同一 harness、同一 token 预算, 只换 AssocMemory 的写规则, 测三件事:
  1) ppl   — 真实语料 (zh_ancient) 上是否损害语言建模.
  2) mqar  — 多查询联想召回 (标准 MQAR), 测基本关联能力.
  3) overwrite — 同一 key 被改写后, 读回的是新值还是混合值 (delta 的立论点).

写 key 取自上一位置 (layers.assoc_shift): 不延迟的话识别到 key 时还没见过 value,
关联根本写不进去, 这是此前 recall 全部停在随机水平的根因。

用法: python -X utf8 experiments/_assoc_ab.py <stage> [steps] [seed]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

VOCAB = 8192


def _cfg(dim, layers, rule, shift=True, decay="reduce"):
    from sunnynet.model import SunnyConfig
    ab = {"assoc": True, "assoc_slots": 32, "assoc_m": 16, "assoc_every": 1,
          "assoc_write": rule, "assoc_shift": shift}
    if rule == "delta":
        ab["assoc_decay"] = decay
    return SunnyConfig(vocab_size=VOCAB, dim=dim, inner=dim * 3 // 2, rank=dim // 2,
                       layers=layers, h_max=2048.0, scan_chunk=64, ablation=ab)


def _split(name):
    """变体名 -> (写规则, 衰减模式, 是否 shift).

    第二位是 shift 指示 (shift/noshift) 时衰减取 reduce; 是衰减模式时 shift 取 True.
    """
    parts = name.split("_")
    rule = parts[0]
    if rule == "off":
        return "off", "reduce", False
    tail = parts[1] if len(parts) > 1 else "shift"
    if tail in ("shift", "noshift"):
        return rule, "reduce", tail == "shift"
    return rule, tail, True


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def stage_ppl(steps=600, dim=128, layers=4, B=8, T=256, seed=1234, lr=3e-3):
    import numpy as np
    from train.train import lr_at
    from sunnynet.model import SunnyNet
    tr = np.memmap(ROOT / "data/proc/tokens_zh_ancient.bin", dtype=np.uint16, mode="r")
    va = np.memmap(ROOT / "data/proc/val_zh_ancient.bin", dtype=np.uint16, mode="r")
    tr = torch.from_numpy(np.asarray(tr, dtype=np.int64))
    va = torch.from_numpy(np.asarray(va, dtype=np.int64))
    torch.set_num_threads(6)
    out = {}
    for name in ("soft", "delta_reduce", "delta_scalar"):
        write, decay, _ = _split(name)
        torch.manual_seed(seed)
        m = SunnyNet(_cfg(dim, layers, write, decay=decay))
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps, lr, max(1, steps // 20))
            i = int(torch.randint(0, len(tr) - B * T - 2, (1,)))
            x = tr[i:i + B * T].reshape(B, T)
            y = tr[i + 1:i + 1 + B * T].reshape(B, T)
            loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
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
        out[name] = ppl
        print(f"  rule={name:13s} val_ppl={ppl:.3f}")
    return out


def _mqar_batch(B, T, n_pair, n_kv, gen):
    """MQAR: 序列放 n_pair 个 (key, value) 相邻对, 末尾查询 key, 预测其 value.

    相邻放置使写 key 取上一位置即可命中: 位置 p 的 key 在 p+1 被写入.
    干扰对之间用随机噪声 token 填充至长度 T-1.
    """
    x = torch.randint(2 * n_kv, VOCAB, (B, T), generator=gen)
    y = x.clone()
    for b in range(B):
        slots = list(range(0, T - 3, 2))
        perm = torch.randperm(len(slots), generator=gen)[:n_pair].tolist()
        pairs = []
        for s in perm:
            p = slots[s]
            kk = int(torch.randint(0, n_kv, (1,), generator=gen))
            vv = int(torch.randint(n_kv, 2 * n_kv, (1,), generator=gen))
            x[b, p] = kk
            y[b, p] = vv              # 让模型在 key 位就预测 value (相邻)
            x[b, p + 1] = vv          # value token 作为下一输入, 供写入
            y[b, p + 1] = vv
            pairs.append((kk, vv))
        qk, qv = pairs[int(torch.randint(0, n_pair, (1,), generator=gen))]
        x[b, T - 1] = qk
        y[b, T - 1] = qv
    return x, y


def stage_mqar(steps=800, dim=64, layers=2, B=32, T=64, n_pair=4, n_kv=32,
               seed=1234, lr=3e-3):
    torch.set_num_threads(6)
    gen = [torch.Generator().manual_seed(seed)]

    def reset():
        gen[0] = torch.Generator().manual_seed(seed)

    def batch():
        return _mqar_batch(B, T, n_pair, n_kv, gen[0])

    def ev(m):
        accs = []
        with torch.no_grad():
            for _ in range(20):
                x, y = batch()
                pred = m(x)[:, -1].argmax(-1)
                accs.append(float((pred == y[:, -1]).float().mean()))
        return sum(accs) / len(accs)

    out = {}
    for rule in ("off", "soft_shift", "delta_shift", "delta_scalar"):
        acc = _run_variant(rule, dim, layers, steps, lr, seed, batch, ev, reset)
        out[rule] = acc
        print(f"  rule={rule:13s} mqar_acc={acc:.4f}  (chance={1.0/n_kv:.4f})")
    return out


def _cfg_off(dim, layers):
    from sunnynet.model import SunnyConfig
    return SunnyConfig(vocab_size=VOCAB, dim=dim, inner=dim * 3 // 2, rank=dim // 2,
                       layers=layers, h_max=2048.0, scan_chunk=64, ablation={})


def _run_variant(name, dim, layers, steps, lr, seed, batch_fn, eval_fn, reset_fn=None):
    from train.train import lr_at
    from sunnynet.model import SunnyNet
    torch.manual_seed(seed)
    if reset_fn is not None:
        reset_fn()                       # 每个变体看同一串数据, 否则比较不公
    if name == "off":
        m = SunnyNet(_cfg_off(dim, layers))
    else:
        rule, decay, shift = _split(name)
        m = SunnyNet(_cfg(dim, layers, rule, shift=shift, decay=decay))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    for step in range(steps):
        for gg in opt.param_groups:
            gg["lr"] = lr_at(step, steps, lr, max(1, steps // 20))
        x, y = batch_fn()
        loss = torch.nn.functional.cross_entropy(
            m(x)[:, -1].reshape(-1, VOCAB), y[:, -1].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        return eval_fn(m)


def stage_overwrite(steps=800, dim=64, layers=2, B=32, T=64, n_pair=4, n_dup=2,
                    n_kv=32, seed=1234, lr=3e-3):
    """覆盖召回: 同一 key 先写旧值再写新值, 查询处应读回新值.

    soft 累加会读成新旧混合; delta 先减去旧关联再写, 应读回新值.
    用 2*n_pair 个槽位: 前 n_pair 写旧值, 后 n_pair 写新值, 其中前 n_dup 个 key
    与前者重复 (构成覆盖); 其余 key 只写一次作为对照.
    "off" 变体关闭关联记忆, 用于排除"主路径就能解、任务太简单".
    """
    torch.set_num_threads(6)
    gen = [torch.Generator().manual_seed(seed)]

    def reset():
        gen[0] = torch.Generator().manual_seed(seed)

    def put(x, y, b, p, tok, val):
        x[b, p] = tok
        y[b, p] = val
        x[b, p + 1] = val
        y[b, p + 1] = val

    def batch():
        x = torch.randint(2 * n_kv, VOCAB, (B, T), generator=gen[0])
        y = x.clone()
        n_slot = 2 * n_pair
        for b in range(B):
            pos = sorted(torch.randperm(T // 2 - 1, generator=gen[0])[:n_slot].tolist())
            pos = [2 * p for p in pos]
            keys = [int(torch.randint(0, n_kv, (1,), generator=gen[0])) for _ in range(n_pair)]
            old = [int(torch.randint(n_kv, 2 * n_kv, (1,), generator=gen[0]))
                   for _ in range(n_pair)]
            new = [int(torch.randint(n_kv, 2 * n_kv, (1,), generator=gen[0]))
                   for _ in range(n_pair)]
            for j in range(n_pair):                      # 第一轮: 写旧值
                put(x, y, b, pos[j], keys[j], old[j])
            for j in range(n_pair):                      # 第二轮: 前 n_dup 个覆盖
                key = keys[j] if j < n_dup else \
                    int(torch.randint(0, n_kv, (1,), generator=gen[0]))
                put(x, y, b, pos[n_pair + j], key, new[j])
            qj = int(torch.randint(0, n_dup, (1,), generator=gen[0]))
            x[b, T - 1] = keys[qj]
            y[b, T - 1] = new[qj]                        # 期望新值
        return x, y

    def ev(m):
        accs = []
        with torch.no_grad():
            for _ in range(20):
                x, y = batch()
                pred = m(x)[:, -1].argmax(-1)
                accs.append(float((pred == y[:, -1]).float().mean()))
        return sum(accs) / len(accs)

    out = {}
    for rule in ("off", "soft_shift", "soft_noshift", "delta_shift", "delta_noshift",
                 "delta_scalar"):
        acc = _run_variant(rule, dim, layers, steps, lr, seed, batch, ev, reset)
        out[rule] = acc
        print(f"  rule={rule:13s} overwrite_acc={acc:.4f}  (chance={1.0/n_kv:.4f})")
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "mqar"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1234
    if stage == "ppl":
        res = stage_ppl(steps or 600, seed=seed)
    elif stage.startswith("overwrite"):
        res = stage_overwrite(steps or 800, seed=seed)
    else:
        res = stage_mqar(steps or 800, seed=seed)
    p = ROOT / f"experiments/results/_assoc_ab_{stage}_s{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("RESULT " + json.dumps(res))
    print("wrote", p)