"""HF 架构走 C 内核的单架构验收: numpy 与 C 两条路径逐 token 对拍.

一个架构一个进程, 免得 1.3GB 级 fp32 blob 的 malloc 被前一个模型的内存挤掉。
    python experiments/_verify_hf_c.py <arch> <hf_dir> <blob> [<tokenizer_dir>]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sunnynet.cli import _add_infer_args, _resolve_model_flags  # noqa: E402
from sunnynet.serve import build_engine  # noqa: E402

PROMPT = "The capital of France is"
N = 6


def _args(arch, hf_dir, blob, tokdir):
    ap = argparse.ArgumentParser()
    _add_infer_args(ap, gguf_backend_default="c")
    argv = ["--hf-dir", hf_dir, "--hf-arch", arch]
    if blob:
        argv += ["--blob", blob]
    if tokdir:
        argv += ["--tokenizer-dir", tokdir]
    a = ap.parse_args(argv)
    _resolve_model_flags(a)
    return a


def _run(eng, prompt=PROMPT, n=N):
    ids = eng.encode(prompt)
    state = eng.init_state()
    lg = None
    for t in ids:
        lg, state = eng.step(int(t), state)
    first = lg.astype(np.float64).copy()
    out = []
    for _ in range(n):
        nxt = int(np.argmax(lg))
        out.append(nxt)
        lg, state = eng.step(nxt, state)
    return first, out


def main():
    arch, hf_dir, blob = sys.argv[1], sys.argv[2], sys.argv[3]
    tokdir = sys.argv[4] if len(sys.argv) > 4 else None

    ref = build_engine(_args(arch, hf_dir, None, tokdir))
    rf, ro = _run(ref)
    ref_text = ref.decode(ro)

    t0 = time.perf_counter()
    ce = build_engine(_args(arch, hf_dir, blob, tokdir))
    load_s = time.perf_counter() - t0
    cf, co = _run(ce)
    c_text = ce.decode(co)

    md = float(np.abs(cf - rf).max())
    rel = md / max(float(np.abs(rf).max()), 1e-9)
    out = {
        "arch": arch, "blob": Path(blob).name,
        "blob_mb": round(Path(blob).stat().st_size / 2 ** 20, 1),
        "numpy_engine": type(ref).__name__, "c_engine": type(ce).__name__,
        "c_engine_name": ce.name,
        "numpy_text": ref_text, "c_text": c_text,
        "text_same": c_text == ref_text,
        "first_step_max_abs_diff": md, "first_step_rel_diff": rel,
        "argmax_same": int(np.argmax(cf)) == int(np.argmax(rf)),
        "c_engine_load_s": round(load_s, 2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()