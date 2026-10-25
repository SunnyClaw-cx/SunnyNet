# Roadmap 与验收记录

*[English](../en/ROADMAP.md)*

本文件记录 CPU 推理路线的四条边界、各项交付状态，并区分**本机实测**与**待真机复现**。
凡本机无验收环境的性能数字，一律标为待复现，不作为对外宣称。

## TL;DR

| 项 | 状态 | 依据 |
|---|---|---|
| 死磕 CPU（AVX2/AVX-512/NEON），不做 CUDA | 守住 | 仓库无 CUDA 源码、无 `torch.cuda` |
| 死磕线性架构，不碰标准 Transformer | 守住 | 无 `MultiheadAttention` 实现 |
| 先做跑别人模型的引擎，再做自家模型 | 守住 | GGUF/外部权重通路先于自有权重发布 |
| 不做 GUI / 训练框架 / 多模态 | 守住 | 无 tkinter/PyQt/gradio/streamlit 依赖 |
| GGUF 兼容层、CLI、OpenAI server、量化、Ollama wrapper | 已本机验收 | 见 §2，验收 JSON 在 `experiments/roadmap_verify/` |
| AVX-512 / NEON 内核 | 代码 + 语义等价已验，**性能待真机** | 见 §3 |
| 树莓派 / Apple Silicon / RK3588 | **无验收环境** | 见 §4 |

## 1. 四条边界

1. 只做 CPU：AVX2 / FMA 是已验证基线；AVX-512 与 NEON 提供源码与加载层，不做 CUDA 后端。
   理由：通用 GPU 推理栈（vLLM / TensorRT-LLM）是别人的主场。
2. 只做线性复杂度架构：RWKV-7 / Mamba-2 / GLA / DeltaNet / HGRN2。标准 Transformer 不在范围内。
3. 顺序：先把外部权重的加载与执行做快，再谈自有模型。当前所有已交付的速度数字都来自
   外部权重（RWKV-7 191M、Mamba-2 130M、DeltaNet 340M、GLA 340M）。
4. 不做 GUI、不做训练框架、不做多模态。`serve.py` / `ollama.py` 只提供 HTTP 接口，不含前端。

仓库内自查（可复现）：

```bash
grep -rn "MultiheadAttention\|torch.cuda\|import tkinter\|PyQt\|gradio\|streamlit" sunnynet/
```

## 2. 已本机验收的条目

验收脚本落盘于 `experiments/roadmap_verify/`，数字来自脚本输出，非手抄。

### 2.1 GGUF 一行命令走 C 内核

```
sunnynet chat -m experiments/gguf/rwkv7-191M-world-bf16.gguf --prompt "Once upon a time"
```

`ensure_rwkv7_blob()` 按 mtime 缓存 blob，首次调用自动导出，之后直接加载。

测试件：`rwkv7-191M-world-bf16.gguf`，368.47 MiB，342 张量，12 层，d_model 768，词表 65536。

| 后端 | ms/token | 口径 |
|---|---:|---|
| numpy fp32 | 530.58 | 8 token，warm 2 |
| C fp32 | 86.55 | 32 token，warm 8 |
| C INT4 | 28.21 | 64 token，warm 16 |

INT4 / fp32 = 3.07×；INT4 / numpy = 18.81×。引擎构建（含 blob 导出）3.3–3.5 s。

INT4 是 B=1 单序列路径，本机 12 逻辑核。与既有结论一致：倍数来自「权重字节数 ÷ 可用带宽」，
不来自实现语言。

### 2.2 OpenAI 兼容 server

官方 `openai` SDK 3.36.0 直连，`base_url` 指向本地端口。

| 调用 | 结果 |
|---|---|
| `models.list()` | 200，`rwkv7-c` |
| `completions.create()` | 200 |
| `chat.completions.create()` | 200 |
| `chat.completions.create(stream=True)` | 200，SSE 逐块 |
| `temperature="abc"` | 400 `参数类型错误: could not convert string to float` |

修复记录：流式响应此前带 `Connection: keep-alive` 且无 `Content-Length`。HTTP/1.0 下
客户端读到 EOF 会一直挂住（实测 `TimeoutError`）。改为 `Connection: close` 并置
`close_connection = True` 后正常收尾。`ollama.py` 的 NDJSON 流有同样问题，一并修掉。

### 2.3 Ollama 原生 API

| 端点 | 结果 |
|---|---|
| `GET /api/version` | 200 `{"version": "0.1.0"}` |
| `GET /api/tags` | 200，`rwkv7-c` |
| `POST /api/generate`（非流） | 200，`eval_count` 与 `num_predict` 一致，`done=true` |
| `POST /api/generate`（流） | 200，NDJSON 逐行，末行 `done=true` |
| `POST /api/chat`（非流 / 流） | 200 / 200 |

### 2.4 量化工具链

```
sunnynet quantize <src.gguf> <dst.gguf> --type q4_0 --ppl --gguf-backend c
```

| 类型 | 目标大小 | 相对源 | 张量 |
|---|---:|---:|---|
| Q8_0 | 335.17 MiB | 1.099× | 量化 169 / 保留 173 |
| Q4_0 | 268.12 MiB | 1.374× | 量化 169 / 保留 173 |

Q4_0 困惑度（512 token，C 后端）：67.89 → 73.82，+8.73%。

K-quant（Q4_K_M / Q5_K_M / Q6_K）在 gguf-py 0.19.0 里只读不写，命令显式报错并列出可写类型，
不静默降级：

```
类型 q4_k_m 在 gguf-py 里只能读不能写 (quantize 未实现 K-quant)。
可写类型: q4_0, q4_1, q5_0, q5_1, q8_0
```

### 2.5 convert 标记

```
sunnynet convert <src.gguf> <dst.gguf> --arch rwkv7
```

写回后 GGUF 元数据含 `sunnynet.arch` / `sunnynet.version` / `sunnynet.kernel_c` /
`sunnynet.source_sha256`，用于校验 blob 缓存是否对应当前权重。

## 3. AVX-512 与 NEON：代码已验，性能待真机

### 3.1 本机事实

| 事实 | 值 |
|---|---|
| CPU | Intel i7-8750H（Coffee Lake），AVX512F=0 |
| AVX-512 源码编译 | 通过（`core_avx512.c` + `step_avx512.c` 合编为 `sunnynet_core512.dll`） |
| AVX-512 加载 | 运行期 CPUID 判定不支持即拒绝，不执行非法指令 |
| NEON 交叉编译 AArch64 | `sunnynet_neon.aarch64-linux-gnu.so`，22264 B，ELF machine 0xb7 |
| NEON 交叉编译 ARMv7 | `sunnynet_neon.arm-linux-gnueabihf.so`，881104 B，ELF machine 0x28 |
| 两目标符号 | `step_core_neon_f32` / `core_neon_fwd_f32` / `scan_neon_f32` / `sn_neon_isa` 齐全 |
| qemu / docker / WSL | 全部不可用（WSL 被本机安全策略拦截） |

### 3.2 借标量 shim 在 x86 上执行真源码

本机没有 AVX-512 CPU 也没有 ARM 设备，但内核源码的**索引与组内语义**是可以在 x86 上跑的：
把 `immintrin.h` / `arm_neon.h` 换成同名 shim，`__m512` / `float32x4_t` 退化为 16 元 / 4 元数组，
逐元素运算语义与真实 SIMD 一致；组内置换仍走源码里的真索引向量。

```bash
python experiments/_avx512_shim/_run.py /f/devtools/w64devkit/bin/gcc
python experiments/_neon_shim/_run.py  /f/devtools/w64devkit/bin/gcc
```

| 内核 | 与 torch 参考的 max\|diff\| | 覆盖 |
|---|---:|---|
| AVX-512 core fwd（D=32，全 16 宽主循环） | 1.19e-07 | 双 group 同寄存器 |
| AVX-512 core fwd（D=24，补 8 宽尾部） | 5.96e-08 | 奇数 NG 的尾部路径 |
| AVX-512 step core（D=32 / D=24） | 1.19e-07 / 5.96e-08 | 同上 |
| AVX-512 dw step | 0.0 | 16 lane 逐元素 + 掩码尾部 |
| NEON scan | 0.0（逐位相同） | 4 lane 扫描 |
| NEON core fwd | 1.19e-07 | 组内 8 通道切两个 4-lane 块 |
| NEON step core | 1.19e-07 | 推理态单步 |

这条路径只能排除「索引写错但仍在界内」这类缺陷——例如 AVX-512 的
`idx[half*8+i] = half*8 + ((i+step)&7)` 若不分区，高 8 lane 会读到别组的通道而数值仍有限。
它**不能**证明真机上的浮点行为与吞吐。

已固化为测试：`tests/test_simd.py::test_avx512_indexing_matches_reference`、
`test_neon_indexing_matches_reference`。

踩坑：shim 初版把 `_mm512_fmadd_ps(a,b,c)` 写成 `a + b*c`（正确为 `a*b + c`），
症状是 `u_out` 与 `p_out` 已经对上、只有 `s_new` 错，容易误判成扫描支路的 bug。

### 3.3 ARMv7 的 FMA 回退

`vfmaq_f32` 需要 NEONv2。用预处理实测两目标下的宏：

| 目标 | `__ARM_FEATURE_FMA` | `SN_FMA` 走哪条 |
|---|---|---|
| `aarch64-linux-gnu` | 定义 | `vfmaq_f32` |
| `arm-linux-gnueabihf`（+`-mfpu=neon-vfpv4`） | 未定义 | `vaddq_f32 + vmulq_f32` |

即 ARMv7 32 位走非融合乘加路径，数值少一次舍入，编码为 `vmla.f32` 而非 `vfma.f32`。

### 3.4 待真机复现（不得对外宣称）

| 目标 | 原定验收标准 | 缺什么 |
|---|---|---|
| AVX-512 | 同频再快 1.5–1.8×，单 token 延迟比 AVX2 再降 30%+ | 无 AVX-512 CPU |
| Apple Silicon | M2/M3 跑 1.6B Q4 达 30+ tok/s | 无 Mac |
| 树莓派 5 / RK3588 | RWKV-7 0.4B 离线语音助手延迟 < 200ms | 无 ARM 设备，无 qemu |
| Ollama 对比 | 比旧版快 30%+ | 本机未安装 ollama |
| 与 llama.cpp 对比 | 1.6B Q4 比 llama.cpp 快 30%+ | 无 llama.cpp 源码；且手上测试件是 191M 而非 1.6B |

这几项在拿到对应硬件前只保留源码与编译证据。

## 4. 未做

- 端侧代码补全小模型（路线原条目 6）：与「不再训练新模型」的既定口径冲突，未启动。
- Ollama 兼容 wrapper 只做到 API 层（`sunnynet/ollama.py`）。原条目里的
  `ollama run rwkv7:1.6b` 自动改走 SunnyNet 需要改 Ollama 本体或走自定义 model 定义，未做。
- `sunnynet quantize` 只支持 gguf-py 能写的类型。K-quant 需要自己实现量化器，未做。

## 5. 验收复现命令

```bash
export PATH="/f/devtools/w64devkit/bin:$PATH"

# GGUF -> C 内核
python -m sunnynet.cli info   -m experiments/gguf/rwkv7-191M-world-bf16.gguf
python -m sunnynet.cli chat   -m experiments/gguf/rwkv7-191M-world-bf16.gguf --prompt "Once upon a time"

# OpenAI 兼容 server
python -m sunnynet.cli serve  -m experiments/gguf/rwkv7-191M-world-bf16.gguf --port 8080

# 量化 + 困惑度
python -m sunnynet.cli quantize <src.gguf> /tmp/q4.gguf --type q4_0 --ppl --gguf-backend c

# 交叉编译
PYTHON=<venv python> bash scripts/build_arm.sh aarch64-linux-gnu
PYTHON=<venv python> bash scripts/build_arm.sh arm-linux-gnueabihf

# 内核语义等价（x86 上跑真源码）
python experiments/_avx512_shim/_run.py $(command -v gcc)
python experiments/_neon_shim/_run.py  $(command -v gcc)
```

回归：`python -m pytest -q` → 166 passed, 42 skipped。