"""统一训练器: SunnyNet 与 Transformer 基线共用同一训练/验证/计时循环, 保证对照公平.

全部指标真实采集 (见 sunnynet/profiling.py), 不产生任何估计值.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_corpus import DOMAINS, load_tokenizer, load_tokens  # noqa: E402
from sunnynet.baseline import BaselineConfig, TransformerLM, solve_baseline_for_params  # noqa: E402
from sunnynet.linear_baselines import (BLOCKS, LINEAR_ARCHS, _LMBase,  # noqa: E402
                                       build_linear_baseline)
from sunnynet.model import SunnyConfig, SunnyNet  # noqa: E402
from sunnynet.profiling import CacheCounter, PerfTracker, peak_rss_mb  # noqa: E402

DEFAULT_MIX = {"zh_ancient": 0.40, "en_literature": 0.35, "code_python": 0.25}

ARCH_CHOICES = ["sunnynet", "transformer", *LINEAR_ARCHS]
DEFAULT_LR = {"sunnynet": 3e-3, "transformer": 6e-3,
              "linear_attn": 3e-3, "rwkv": 3e-3, "mamba": 3e-3,
              "hyena": 3e-3, "h3": 3e-3}


class DomainMixer:
    """按域权重随机取窗口的流式数据装载器 (不整表载入, 不打乱全量数据)."""

    def __init__(self, weights: dict, seq_len: int, batch_size: int,
                 split: str = "train", seed: int = 0):
        self.arrs = {}
        for d, w in weights.items():
            if w <= 0:
                continue
            a = load_tokens(d, split)
            if a.size > seq_len + 2:
                self.arrs[d] = a
        if not self.arrs:
            raise SystemExit("没有可用 token 流, 请先运行 build_corpus.py")
        self.doms = list(self.arrs)
        p = np.array([weights[d] for d in self.doms], dtype=np.float64)
        self.p = p / p.sum()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.cursor = {d: 0 for d in self.doms}

    def batch(self):
        B, L = self.batch_size, self.seq_len
        off = self.rng.choice(len(self.doms), size=B, p=self.p)
        xs = np.empty((B, L), dtype=np.int64)
        ys = np.empty((B, L), dtype=np.int64)
        for b in range(B):
            d = self.doms[int(off[b])]
            a = self.arrs[d]
            s = int(self.rng.integers(0, a.size - L - 1))
            w = a[s:s + L + 1].astype(np.int64)
            xs[b] = w[:-1]
            ys[b] = w[1:]
        return torch.from_numpy(xs), torch.from_numpy(ys)

    def val_batches(self, n_batches: int):
        """验证集取固定窗口, 保证跨轮次可比较."""
        B, L = self.batch_size, self.seq_len
        rng = np.random.default_rng(12345)
        out = []
        for _ in range(n_batches):
            xs = np.empty((B, L), dtype=np.int64)
            ys = np.empty((B, L), dtype=np.int64)
            for b in range(B):
                d = self.doms[b % len(self.doms)]
                a = self.arrs[d]
                s = int(rng.integers(0, a.size - L - 1))
                w = a[s:s + L + 1].astype(np.int64)
                xs[b] = w[:-1]
                ys[b] = w[1:]
            out.append((torch.from_numpy(xs), torch.from_numpy(ys)))
        return out


def _preset_cfg(preset: str, vocab: int) -> SunnyConfig:
    if preset == "proto20m":
        return SunnyConfig.proto20m(vocab)
    if preset == "proto20mw":
        return SunnyConfig(vocab_size=vocab, dim=640, inner=1280, rank=384,
                           layers=6, h_max=8192.0)
    if preset == "smalln":
        # 等栈参数下的窄残差流配置: 逐元素算子按宽度付费, dim 128->96 提速约 17%
        return SunnyConfig(vocab_size=vocab, dim=96, inner=192, rank=152,
                           layers=4, h_max=2048.0)
    if preset == "smalln88":
        return SunnyConfig(vocab_size=vocab, dim=88, inner=176, rank=196,
                           layers=4, h_max=2048.0)
    if preset == "smalln80":
        return SunnyConfig(vocab_size=vocab, dim=80, inner=160, rank=244,
                           layers=4, h_max=2048.0)
    if preset == "p100m":
        return SunnyConfig(vocab_size=vocab, dim=1024, inner=1792, rank=640,
                           layers=14, h_max=16384.0)
    if preset == "p100mw":
        return SunnyConfig(vocab_size=vocab, dim=1408, inner=1920, rank=1024,
                           layers=8, h_max=16384.0)
    if preset == "p100msw":
        # 同栈参数 (93.83M, +0.68%) 下减层加宽: L8->L6, d1408->d1536.
        # 每层一笔固定算子派发税 (~1100 次调用/步), 深度在速度与质量两条轴上都不划算.
        return SunnyConfig(vocab_size=vocab, dim=1536, inner=2624, rank=1152,
                           layers=6, h_max=16384.0)
    if preset == "deepl6":
        return SunnyConfig.deepl6(vocab)
    if preset == "deepl8":
        return SunnyConfig.deepl8(vocab)
    return SunnyConfig.small(vocab)


def build_models(arch: str, vocab: int, preset: str, ablation: dict | None,
                 seq_len: int, seed: int, h_max: float = 0.0,
                 base_layers: int = 0):
    torch.manual_seed(seed)
    sunny_cfg = _preset_cfg(preset, vocab)
    if h_max > 0:
        sunny_cfg.h_max = h_max
    if ablation:
        sunny_cfg.ablation = dict(ablation)
    sunny_cfg.scan_chunk = 64
    sunny = SunnyNet(sunny_cfg)

    if arch == "sunnynet":
        return sunny, None, sunny_cfg

    target = sunny.num_params(non_embedding=True)
    if arch == "transformer":
        tl = base_layers or 6
        bcfg, bn, rel = solve_baseline_for_params(target, vocab, n_layers=tl, n_heads=8,
                                                  max_len=max(seq_len, 2048))
        base = TransformerLM(bcfg)
    else:
        nl = {"proto20m": 6, "proto20mw": 5, "p100m": 8, "p100mw": 8}.get(preset, 4)
        bcfg, bn, rel = build_linear_baseline(arch, target, vocab,
                                              max_len=max(seq_len, 2048), n_layers=nl)
        base = _LMBase(bcfg)
        for _ in range(bcfg.n_layers):
            base.blocks.append(BLOCKS[arch](bcfg))
        base._finish_init()
    print(f"[baseline:{arch}] 以 SunnyNet 计算栈参数 {target:,} 为目标, "
          f"选中 d_model={bcfg.d_model} -> 计算栈参数 {bn:,} (偏差 {rel*100:.2f}%)")
    return sunny, base, bcfg


def lr_at(step: int, total: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


@torch.no_grad()
def evaluate(model, batches, device="cpu") -> float:
    model.eval()
    dev = torch.device(device)
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
            tot += float(loss)
            n += y.numel()
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCH_CHOICES, default="sunnynet")
    ap.add_argument("--preset", choices=["small", "smalln", "smalln88", "smalln80", "deepl6", "deepl8", "proto20m", "proto20mw", "p100m", "p100mw", "p100msw"],
                    default="small")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--h-max", type=float, default=0.0, help="覆盖 SunnyConfig.h_max (0=用预设)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--kernel", type=int, default=-1,
                    help="1=用 AVX2 融合内核, 0=强制 torch 路径, -1=读 SUNNY_KERNEL")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--patience", type=int, default=4, help="val loss 连续上升多少轮触发早停")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--ablation", type=str, default="", help="如 decay_gate=0,circulant=0")
    ap.add_argument("--mix", type=str, default="")
    ap.add_argument("--save-ckpt", type=str, default="")
    ap.add_argument("--save-every", type=int, default=0,
                    help="里程碑 checkpoint 间隔步数 (0=关闭)")
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu 或 cuda (GPU 仅用于规模化训练, 结论不混 CPU 速度口径)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    th = torch.get_num_threads()

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda 但 CUDA 不可用, 回退 cpu")
        dev = torch.device("cpu")

    if args.kernel >= 0:
        from sunnynet.kernels.core_auto import set_kernel_active
        set_kernel_active(bool(args.kernel), threads=th)
    from sunnynet.kernels.core_auto import kernel_active
    k_active = kernel_active()

    weights = dict(DEFAULT_MIX)
    if args.mix:
        weights = {}
        for kv in args.mix.split(","):
            k, v = kv.split("=")
            weights[k] = float(v)

    ablation = {}
    if args.ablation:
        for kv in args.ablation.split(","):
            k, v = kv.split("=")
            try:
                iv = int(v)
            except ValueError:
                ablation[k] = v
            else:
                ablation[k] = bool(iv) if iv in (0, 1) else iv

    tk = load_tokenizer()
    vocab = tk.vocab_size

    sunny, base, cfg = build_models(args.arch, vocab, args.preset, ablation, args.seq_len, args.seed, args.h_max)
    model = sunny if args.arch == "sunnynet" else base
    model = model.to(dev)
    n_all = model.num_params()
    n_ne = model.num_params(non_embedding=True)

    mixer = DomainMixer(weights, args.seq_len, args.batch_size, "train", seed=args.seed)
    val_mixer = DomainMixer(weights, args.seq_len, args.batch_size, "val", seed=1)
    val_batches = val_mixer.val_batches(args.val_batches)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    cache = CacheCounter()
    perf = PerfTracker()
    if cache.ok:
        print(f"    访存计数器: {cache.kind}"
              + ("" if cache.kind == "cache_read_hit_rate"
                 else "  (本机无 Cache 性能对象, 缓存命中率不可测, 以页错误率作代理, 不冒充实测)"))

    print(f"=== {args.arch} | preset={args.preset} | threads={th} "
          f"| params={n_all:,} (non-embed {n_ne:,}) | kernel={k_active}")
    print(f"    steps={args.steps} bs={args.batch_size} seq={args.seq_len} "
          f"lr={args.lr} tokens={args.steps * args.batch_size * args.seq_len:,}")
    print(f"    cache counter available={cache.ok} | mix={weights} | ablation={ablation or 'none'}")

    log = {"arch": args.arch, "preset": args.preset, "params": n_all,
           "params_non_embed": n_ne, "threads": th, "kernel": k_active, "args": vars(args),
           "mix": weights, "ablation": ablation, "history": []}
    best_val, bad, stopped = float("inf"), 0, False
    diverged = False
    t_start = time.perf_counter()

    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = mixer.batch()
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        t0 = time.perf_counter()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        ntok = x.numel()
        perf.update(ntok, dt, gn, cache)

        if not math.isfinite(gn) or not loss.isfinite():
            print(f"  [发散] step {step+1}: loss={float(loss):.4f} grad_norm={gn}")
            diverged = True
            break

        rec = None
        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_batches, device=str(dev))
            t_eval = time.perf_counter() - t_start
            rec = {"step": step + 1, "train_loss": round(float(loss.detach()), 6),
                   "val_loss": round(vl, 6), "ppl": round(math.exp(min(vl, 20)), 4),
                   "ms_per_token": round(perf.ms_per_token, 6),
                   "tokens_per_sec": round(perf.tokens_per_sec, 2),
                   "grad_norm": round(gn, 6), "lr": round(lr, 8),
                   "peak_rss_mb": round(perf.peak_rss, 1),
                   "cache_hit": (round(perf.mean_cache_hit, 4)
                                 if perf.mean_cache_hit is not None else None),
                   "page_faults_per_sec": (round(perf.mean_page_faults, 2)
                                           if perf.mean_page_faults is not None else None),
                   "elapsed_s": round(t_eval, 2)}
            log["history"].append(rec)
            if args.save_every > 0 and args.save_ckpt and (step + 1) % args.save_every == 0:
                sp = Path(args.save_ckpt).with_name(
                    Path(args.save_ckpt).stem + f"_{step+1:06d}" + Path(args.save_ckpt).suffix)
                sp.parent.mkdir(parents=True, exist_ok=True)
                sd2 = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save({"arch": args.arch, "model": sd2,
                            "config": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
                            "summary": {"step": step + 1, "val_loss": rec["val_loss"]}}, sp)
                print(f"    里程碑 checkpoint -> {sp}")
            print(f"  step {step+1:5d} loss={rec['train_loss']:.4f} val={rec['val_loss']:.4f} "
                  f"ppl={rec['ppl']:.2f} {rec['ms_per_token']:.3f} ms/tok "
                  f"gn={gn:.3f} rss={rec['peak_rss_mb']:.0f}MB "
                  f"cache={rec['cache_hit']} pf/s={rec['page_faults_per_sec']} "
                  f"{rec['elapsed_s']:.1f}s")
            if vl < best_val - 1e-4:
                best_val, bad = vl, 0
            else:
                bad += 1
                if not args.no_early_stop and bad >= args.patience:
                    print(f"  [早停] val loss 连续 {bad} 轮未改善 (best={best_val:.4f})")
                    stopped = True
                    break

    wall = time.perf_counter() - t_start
    log["summary"] = {
        "wall_seconds": round(wall, 2),
        "steps_done": step + 1,
        "stopped_early": stopped,
        "diverged": diverged,
        "best_val_loss": round(best_val, 6) if best_val < float("inf") else None,
        "best_ppl": round(math.exp(min(best_val, 20)), 4) if best_val < float("inf") else None,
        "ms_per_token_mean": round(perf.ms_per_token, 6),
        "tokens_per_sec_mean": round(perf.tokens_per_sec, 2),
        "peak_rss_mb": round(perf.peak_rss, 1),
        "mean_grad_norm": round(perf.mean_grad_norm, 6) if perf.mean_grad_norm else None,
        "cache_hit_mean": (round(perf.mean_cache_hit, 4)
                           if perf.mean_cache_hit is not None else None),
        "cache_counter_kind": perf.cache_kind,
        "page_faults_per_sec_mean": (round(perf.mean_page_faults, 2)
                                     if perf.mean_page_faults is not None else None),
        "final_train_loss": round(float(loss.detach()), 6) if loss.isfinite() else None,
    }
    print(f"=== 完成 wall={wall:.1f}s  {log['summary']['ms_per_token_mean']:.3f} ms/token  "
          f"peak_rss={log['summary']['peak_rss_mb']:.0f}MB  "
          f"pf/s={log['summary']['page_faults_per_sec_mean']}  "
          f"best_val={log['summary']['best_val_loss']}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    日志 -> {p}")

    if args.save_ckpt:
        cp = Path(args.save_ckpt)
        cp.parent.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"arch": args.arch, "model": sd,
                    "config": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
                    "summary": log["summary"]}, cp)
        print(f"    checkpoint -> {cp} ({cp.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())