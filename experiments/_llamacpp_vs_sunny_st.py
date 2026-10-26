"""llama.cpp vs SunnyNet 单线程收口对照 (多轮, 同权重).

本机 (i7-8750H) 在持续负载下多线程数字抖动极大, 单线程受调度干扰最小。这里只测
1 线程, 把轮数加到 7, 逐轮配对取比值 —— 绝对值受降频影响, 比值不受。

四格: llama.cpp F32 (723 MB) / SunnyNet fp32 (764 MB)  —— 同权重同精度
      llama.cpp Q4_0 (127 MB) / SunnyNet INT4 (282 MB) —— 同量化档
"""
from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GGUF = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.gguf"
SUNNY_FP32 = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.runtime.bin"
SUNNY_INT4 = ROOT / "experiments/gguf/rwkv7-191M-world-bf16.q4.bin"
OUT = ROOT / f"experiments/results/_llamacpp_vs_sunny_t{os.environ.get('THREADS', '1')}.json"

LC_DIR = Path(os.environ.get("LLAMA_DIR", r"F:/sunnyclaw-data/tools/llama.cpp"))
BENCH = LC_DIR / "llama-bench.exe"
LC_BF16 = LC_DIR / "rwkv7-191M-world-bf16.gguf"
LC_F32 = LC_DIR / "rwkv7-191M-world-f32.gguf"
LC_Q4 = LC_DIR / "rwkv7-191M-world-q4_0.gguf"
NGEN = 64
SUNNY_STEPS = 60
THREADS = os.environ.get("THREADS", "1")

WORKER = r'''
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(r"{root}")))
from sunnynet.runtime.rwkv7 import RWKV7
from sunnynet.runtime.rwkv7_c import SnRWKV7
ref = RWKV7(Path(r"{gguf}"))
m = SnRWKV7(Path(sys.argv[1]), ref.vocab)
m.reset()
ids = ref.tok.encode("Hello world, this is a warmup sequence for timing.")
for t in ids[:-1]:
    m.step_raw(t)
tok = ids[-1]
ts = []
for _ in range({steps}):
    t0 = time.perf_counter(); m.step_raw(tok); ts.append(time.perf_counter() - t0)
ts.sort()
print(json.dumps({{"ms_tok": ts[len(ts)//2]*1e3}}))
'''


def sunny(blob: Path) -> float:
    code = WORKER.format(root=str(ROOT), gguf=str(GGUF), steps=SUNNY_STEPS)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = THREADS
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(blob)],
                       capture_output=True, text=True, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1200:])
    return json.loads(r.stdout.strip().splitlines()[-1])["ms_tok"]


def llama(gguf: Path) -> float:
    r = subprocess.run([str(BENCH), "-m", str(gguf), "-p", "0", "-n", str(NGEN),
                        "-t", THREADS, "-r", "5", "-o", "json"],
                       capture_output=True, text=True, cwd=str(LC_DIR))
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1200:])
    d = json.loads(r.stdout)[0]
    return d["avg_ns"] / d["n_gen"] / 1e6


CELLS = [
    ("llama.cpp bf16", "llama", LC_BF16, 384),
    ("llama.cpp F32", "llama", LC_F32, 723),
    ("SunnyNet fp32", "sunny", SUNNY_FP32, 764),
    ("llama.cpp Q4_0", "llama", LC_Q4, 127),
    ("SunnyNet INT4", "sunny", SUNNY_INT4, 282),
]


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    acc: dict[str, list[float]] = {}
    per_round = []
    for i in range(rounds):
        row = {}
        for label, side, ident, _ in CELLS:
            row[label] = llama(ident) if side == "llama" else sunny(ident)
            acc.setdefault(label, []).append(row[label])
        per_round.append(row)
        print(f"round {i}: " + "  ".join(f"{k} {v:.1f}" for k, v in row.items()), flush=True)

    med = {k: st.median(v) for k, v in acc.items()}
    pairs = [("SunnyNet fp32", "llama.cpp F32"),
             ("SunnyNet INT4", "llama.cpp Q4_0")]
    paired = {}
    for s, l in pairs:
        ratios = [row[l] / row[s] for row in per_round]
        paired[f"{s} vs {l}"] = {
            "ratios": [round(r, 3) for r in ratios],
            "median": round(st.median(ratios), 3),
            "sunny_faster": f"{sum(1 for r in ratios if r > 1)}/{rounds}",
        }
    # 跨精度: SunnyNet INT4 对 llama.cpp 的原生 bf16 (同一份源 gguf)
    ratios = [row["llama.cpp bf16"] / row["SunnyNet INT4"] for row in per_round]
    paired["SunnyNet INT4 vs llama.cpp bf16"] = {
        "ratios": [round(r, 3) for r in ratios],
        "median": round(st.median(ratios), 3),
        "sunny_faster": f"{sum(1 for r in ratios if r > 1)}/{rounds}",
    }

    # 有效带宽: 权重字节 / 步耗时
    bw = {label: round(mb / med[label], 2) for label, _, _, mb in CELLS}

    summary = {
        "impl": f"llama.cpp b11889 CPU vs SunnyNet rwkv7.c, {THREADS} 线程",
        "weights": "同一份 BlinkDL/rwkv-7-world 191M",
        "note": "本机滑动降频, 只看逐轮配对比值; 权重 MB 为该侧实际读取字节",
        "threads": int(THREADS),
        "rounds": rounds,
        "median_ms_tok": {k: round(v, 2) for k, v in med.items()},
        "effective_bandwidth_GBps": bw,
        "paired_ratio": paired,
        "all_ms": acc,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{THREADS} 线程 median ms/token 与有效带宽")
    for label, _, _, mb in CELLS:
        print(f"  {label:16s} {med[label]:7.1f} ms   {mb} MB   {bw[label]:5.2f} GB/s")
    print("\n逐轮配对倍数 (>1 表示 SunnyNet 更快)")
    for k, v in paired.items():
        print(f"  {k:28s} {v['median']:.2f}x  ({v['sunny_faster']})  {v['ratios']}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())