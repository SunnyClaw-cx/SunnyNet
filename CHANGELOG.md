# Changelog

*[English](CHANGELOG.en.md)*

## 未发布

### 训练里程碑 checkpoint + 语料/代码扩充
- `train/train.py` 新增 `--save-every N`（默认 0 关闭）：每 N 步在
  `--save-ckpt` 旁另存 `<stem>_<step>.pt` 里程碑 checkpoint，便于长跑采样
  与断点观察；保存前 `mkdir(parents=True)`，克隆环境无 `runs/` 也能跑。
- `fetch_books.py`（新增）：经 gutendex API 抓下载量最高的公开领域英文书
  （`download_count` 排序），按 `prepare_data` 的分域/清洗口径写入
  `en_literature`，输出字节兼容；替代原先 13 本固定经典书目，语料多样性大增。
- `fetch_code.py`（新增）：抓 numpy / flask / requests / urllib3 四个库的
  `.py` 源码（tarball + `clean_code`）补入 `code_python`，代码域不再只有
  TheAlgorithms 一个来源。
- `kaggle/`（新增/更新）：三个自包含 Kaggle 训练内核 ——
  `sunnynet_p100.py`（p100m 94M 英文散文）、`code_mix/`（94M 70/30 散文+代码）、
  `big_data/`（proto20m 21M 大语料 1000 本、seq 1024，冲 ~12 tokens/param）。
  各内核 base64 内嵌取数脚本、按 epoch 估算步数、VRAM 自探 batch、结果写
  `/kaggle/output`；`kaggle/README.md` 说明用法。
- `docs/en/MODEL_SHOWCASE.md`（新增）：2026-10-24 一次会话训练的 7 个模型逐一
  展示（参数/硬件/步数/token 数/最佳 val + 真实生成样例），并给出
  tokens/param 这一贯穿结论与「流畅≠连贯」的读数。

### TinyStories 语料 + 大域编码 OOM 修复
- `fetch_tiny_stories.py`（新增）：断点续传下载 TinyStories
  （HF `roneneldan/TinyStories` 的 `TinyStories-train.txt`，1.9 GB，~50k 短故事、
  ~1.16B token），按故事边界（空行）切成 ~4MB 分片写入 `data/raw/tiny_stories/`，
  作为插件式域（`--mix tiny_stories=1.0`，`train.py`/`DomainMixer` 无需改代码）。
- `kaggle/tiny_stories/`（新增）：自包含 P100 内核 —— 故事专用 BPE、流式编码、
  `proto20m` 21M @ seq 512 训 ~150M token（≈9 tok/非嵌入参数），里程碑采样到
  `/kaggle/output`。
- `build_corpus.encode_domain`：改逐文件 uint16 流式拼接，不再攒整域 Python
  int 列表（旧实现峰值内存 ∝ 整域 token 数，~1.2B token 语料会 OOM）。
  回归测试 `tests/test_build_corpus.py`（大域 token 计数 / 文件级 train-val 切分
  不泄漏 / 空域）。

## v0.3.0

### 层数扩展的代价机制与新预设
- 训练步随层数线性：固定 `d=128`、T=256/B8/6 线程、3 轮交错实测
  `step_ms ≈ 32.9·L + 102.4`。每层固定新增约 **1100 次 aten 调用**，其中 `aten::mm`
  只占 1.9%、逐元素与视图算子占 95.7%，每层 `mm` 仅 21 次
  ⇒ **每层是一笔与宽度无关的算子派发税**。
- 融合内核把每层斜率从 1099 降到 715 次调用。逐段归因（替换实验给收益上限）显示这笔税
  分散在 `scan`/`readout`/`compete`/`local`/`emit` 五段（归零后每层降至 0.65–0.74 倍），
  没有单段可独占 ⇒ 继续融合的收益有限。
- 同栈参数（365,844）下扫 6 个 `(L, dim)` 族，900 步 2 种子：`L3 wide` 最快
  （0.0995 ms/token），`L4 wide`（d121）质量最好（zh ppl 247.2），`L6`/`L8` 三族
  速度与质量**两条轴都更差** ⇒ 深度在同参数预算下不买质量。
- 新预设 `p100msw`（d1536/i2624/r1152/L6，栈 94,470,942，对齐 `p100mw` +0.68%）：
  隔离 A/B（B4×T256，3 轮交错）比 `p100mw`（d1408/L8）快 **4.2%**（3/3 轮占优），
  层数回到同行档。**不要把参数量堆到层数上——这是配置纪律。**

### 层间输入稀释的定位与修复
- 逐支公共分量归因（`experiments/_dilution_source.py`）：深模型逐层累积的公共分量里
  `W_c(y)` 支路占约 **96%**，是与输入无关、缓慢漂移的常量，经残差流自增强
  （状态 DC → 残差流 DC → `norm_in` → `u` → `y` → `W_c(y)`）。
  此前按原始成对余弦读出的"状态坍缩 0.50"是公共分量虚影，去均值后为 0。
- `state_norm` 改为同时归一化反馈源 `sfb`（原先只覆盖 `sn`），堵住 DC 从反馈支路漏回：
  `deepl8` 6000 步上 24 输入 argmax 多样度 **12 → 16**、val ppl 200.51 → **198.33**。
- 残差流归一 `h_norm` 实测证伪（ppl 263 → 625，24 输入塌成 1 个预测），保留作对照。

### DeltaNet 整模型 C 内核
- `runtime/deltanet.c`：整个前向在 C（三路深度短卷积 + 逐 head L2 + delta rule +
  RMSNorm + SwiGLU），权重由 `experiments/_export_deltanet.py` 导成 fp32 blob。
  Python 侧 `sunnynet.runtime.deltanet_c.SnDeltaNet`，CLI `--hf-arch deltanet --blob`。
- 对 numpy 参考路径 36 token：logits max|diff| **1.335e-05**、argmax **36/36**、
  重复运行与 reset 后重跑逐位 0.0、prefill 与逐步 0.0。
- 速度单步 78.69 → **71.24 ms/token（1.10×，5/5 逐轮占优）**。只有 1.10× 的原因是
  **两条路径都撞在内存带宽墙**：C 侧有效带宽 19.18 GB/s，已超单线程顺序读上限
  18.63 GB/s。对访存受限的整模型解码，C 化收益上限由带宽决定。
- 线程数取物理核（6 线程 64.33 ms / 21.24 GB/s，优于 12 逻辑核 68.55 ms）；
  默认自动取物理核。`tests/test_deltanet_c.py` 7 项。

### Mamba-2 整模型 C 内核
- `runtime/mamba2.c`：整个前向在 C（单路深度短卷积 + 元素级状态递推 + RMSNorm，
  无 L2/双 norm/SwiGLU MLP，衰减沿 state 维广播故无需矩阵乘），权重由
  `experiments/_export_mamba2.py` 导成 516 MB fp32 blob。
  Python 侧 `sunnynet.runtime.mamba2_c.SnMamba2`，CLI `--hf-arch mamba2 --blob`。
- 对 numpy 参考路径 36 token：logits max|diff| **1.678e-04**、相对 **6.307e-06**、
  argmax **36/36**。绝对误差比 DeltaNet 大一个量级，逐层对表
  （`experiments/_mamba2_layer_probe.py`）确认 24 层相对误差全在 4e-08 ~ 1.1e-06
  且不随层数放大 ⇒ 单步归约顺序噪声，非累积、非公式错误。
- 速度单步 47.36 → **34.59 ms/token（1.37×，5/5 逐轮占优）**。倍数高于 DeltaNet 的
  1.10× 同样是带宽原因：权重 0.52 GB 仅为 DeltaNet 1.37 GB 的 38%，
  两条路径都未贴到 18.63 GB/s 上限。`tests/test_mamba2_c.py` 7 项。

### GLA 整模型 C 内核
- `runtime/gla.c`：整个前向在 C（低秩衰减投影 + 元素级状态递推 + 输出门 +
  SwiGLU MLP），权重由 `experiments/_export_gla.py` 导成 1.37 GB fp32 blob（源 bf16）。
  Python 侧 `sunnynet.runtime.gla_c.SnGLA`，CLI `--hf-arch gla --blob`。
- 对 numpy 参考路径 36 token：logits max|diff| **1.431e-05**、相对 **1.081e-06**、
  argmax **36/36**、重复与 reset 逐位 0.0、prefill 与逐步 0.0。
- 速度单步 74.92 → **74.15 ms/token（1.01×，4/5 逐轮占优）**。权重 1.37 GB 与 DeltaNet
  同量级，理论下限 `1.37 GB ÷ 18.63 GB/s ≈ 73.5 ms`，两条路径的中位读数都已贴到这条线
  ⇒ C 化无余量可拿。三个架构合并印证：**收益由「权重字节数 / 可用带宽」决定**。
- 新增层内阶段探针 `sn_gla_probe`（10 个 stage），定位到 `gla_scan` 里所有 head 都读
  `v` 的第 0 段而非各自 value 段（单步 readout 差 96%，各投影却全对）。
  `head_k_dim ≠ head_v_dim` 时该错**不越界**，只会静默算错。`tests/test_gla_c.py` 8 项。

### 矩阵值 rank-1 递推内核
- 新增 `sn_mat_scan`（`runtime/recur.c`，声明见 `recur.h`）：统一形式
  `p=S·u; S=S⊙w+v⊗k−p⊗b; y=S·r`，参数传 `NULL` 即跳过，覆盖 RWKV-7 与 DeltaNet
  的矩阵状态层级。逐 head 串行、沿 `C` 维 8 宽向量化 + 标量尾。
- Python 侧 `sunnynet.runtime.mat_step(S, w=, u=, v=, k=, b=, r=)`，`S` 为 `(NH,R,C)`
  连续 fp32，原地更新，返回 `y`。
- 数值契约 `tests/test_mat_scan.py` 11 用例全通过，相对误差 1e-7 量级。
- 算子级 8×128×128 为 numpy 的 **4.25×**；但实测只占 DeltaNet 单步 **2.2%**，
  不能据此声称端到端加速——剩余时间在逐层 GEMV。

### 预编译库与源码一致性
- `build_prebuilt()` 写出 `_prebuilt/<平台>/sources.sha256`（按库分组的源码内容哈希），
  `locate()` 载入前比对；源码改过但没重编时不再静默命中旧库，直接回退源码编译
  （开发中真实踩到：`sn_mat_scan` 已编入源码却报 `AttributeError`）。
- 比对内容哈希而非 mtime——pip 解包会把源码 mtime 刷成解包时刻，按时间判会恒判过期。
  隔离安装实测全库命中、`prebuilt_stale()` 为 False。库数由三个整模型库
  （DeltaNet / Mamba-2 / GLA）增至 9。
- 契约测试 `tests/test_native_prebuilt.py`（7 项）。

### 构建期的两个 Windows 缺陷
- `build_prebuilt()` 原先直接调各模块 `_compile`，写的是共享的 `<pkg>/_build/<dll>`；
  该文件一旦被本进程加载过，Windows 拒绝覆盖写，编译即失败（全量测试顺序依赖地报
  3 个 `RuntimeError`，单跑该文件却全过）。改为编译到临时目录再复制；产物 dll 被占用
  时按库单独捕获 `PermissionError`，不拖垮其余。
- 六处 `_compile` 用 `subprocess.run(..., text=True)` 按 UTF-8 解码，而 Windows 下
  gcc 诊断按系统代码页（GBK）输出，路径含非 ASCII 时抛 `UnicodeDecodeError` 掩盖真实
  错误（只有 `abi.py` 单独容错过）。统一走 `_native.run_cc()` 收 bytes 再容错解码。
- 各模块新增 `_link_cmd(gcc, dst)`，`_compile` 只负责写到目标路径；契约测试断言每个
  模块都提供该接口。
- `deltanet_c._recommended_threads()` 不再自行 `CDLL` 拼接 `_build/sunnynet_model.dll`
  （绕过预编译查找、且是上条的触发源），改为复用 `runtime.model.recommended_threads()`。
- `prebuilt_stale()` 的判定有进程内缓存，`build_prebuilt()` 重建后未清，导致「改源码
  → 重建 → 加载」一直读到过期结论（等于仍退回源码编译）。重建末尾清缓存。
- 契约测试原先硬编码产物名 `sunnynet_runtime.dll`，而清单键带平台后缀（Linux 为
  `.so`），在 `ubuntu-latest` 上 `KeyError`；Windows 本地恰好匹配故一直未暴露。
  改为从模块取 `DLL.name`。

### fp16 权重解码路径
- 新增 `sn_load_h()`：载入时把 5 个 GEMV 矩阵与 LM head 截断为 fp16，小参数仍 fp32；
  单步实读 467.6 → 约 222.3 MB。内核 `gemv_h` 一次读 16 个 half（32 B）后转 fp32 做
  FMA，保持与 fp32 相同的加载宽度。
- `p100mw` 单步 24.67 → **18.60 ms/token**（**1.33×**，11 轮交错 11/11 占优）。
  logits max|diff| 4.181e-03、相对偏差 7.42e-04、argmax 64/64。
- Python 侧 `SnModel(..., half=True)`，CLI `--backend c --half`。fp32 仍是默认，逐位不变。

## v0.2.0

### 推理 Runtime 访存重构
- 整模型 C 解码的权重在 `sn_load` 时翻成行主序 `(out,in)`，GEMV 改为按输出通道
  并行、每行连续读。导出格式不变，已发布的 blob 无需重导。
- `p100mw`（105.36M）单步解码 76.60 → **25.72 ms/token**（**2.98×**，5 轮交错
  5/5 占优），有效带宽 6.10 → 18.18 GB/s。对优化前 logits max|diff| 2.766e-05、
  argmax 32/32。
- 并行度取物理核而非逻辑核（解码访存受限）：6 核 26.69 ms vs 12 逻辑核 36.05 ms。
  新增 `sn_recommended_threads()` 与 `sunnynet.runtime.model.set_threads()`；
  CLI 在 `--backend c` 且未指定 `--threads` 时自动采用物理核数。

### 外部权重
- GLA 接入（`sunnynet/runtime/gla.py`）：门控只在 key 维，衰减可分解为按行缩放
  ⇒ 元素级递推，不需要矩阵值层级。对独立参考实现 max|diff| 2.05e-05、argmax 12/12、
  分段续算逐位 0.0。CLI `--hf-arch gla`。

### 服务与集成
- OpenAI 兼容 HTTP 服务（`sunnynet/serve.py`，标准库零依赖）：`/v1/models`、
  `/v1/chat/completions`（含 SSE 流式）、`/v1/completions`、`/health`。
  官方 `openai` SDK 四种调用方式验证通过。命令 `sunnynet-serve`。
- Ollama / LM Studio 接入（`sunnynet/integrate.py`）：`sunnynet-integrate` 生成
  Modelfile 与 LM Studio 配置。RWKV-7 GGUF 经 Ollama 真实导入并生成通过
  （`family=rwkv7`、`runner=llamacpp`）。

### C API 稳定 ABI
- `sunnynet/runtime/sn_abi.h` / `sn_abi.c`：`sn_abi_` 前缀隔离、不透明句柄、
  能力位、错误码、`SN_ABI_VERSION 1.0.0`。与内部句柄路径逐位一致（max|diff| 0.0）。
  契约测试 9 项。详见 `docs/cn/ABI.md`。

### SIMD 后端
- AVX-512F/BW/VL 内核（`sunnynet/kernels/scan_avx512.c`）：编译通过，
  反汇编确认 198 处 `zmm` / 16 处 `vfmadd132ps`。本机 CPU 不支持，只做产物验证。
- ARM NEON 内核（`sunnynet/kernels/scan_neon.c`）：`zig cc` 交叉编译出 AArch64 ELF64，
  capstone 确认 4-lane `fmla`。执行验证放在 CI 的 `ubuntu-24.04-arm` runner。

### 权重发布
- `sunnynet-release`（`sunnynet/release.py`）：导出 safetensors（config 与训练摘要
  写入 metadata）、runtime blob、模型卡与 sha256 清单。
- 训练器支持 `--device cuda`，checkpoint 统一以 CPU tensor 落盘。
- `p100mw` 在 GTX 1060 6GB 上训练 2000 步（1.024M token），val ppl 9983 → 120.09，
  0.503 ms/token。发布资产 `.pt` 467.7 MB / `.safetensors` 421.5 MB /
  `.runtime.bin` 467.6 MB，torch、C Runtime、OpenAI 服务三条路径加载生成均通过。

### 测试
- 121 passed / 10 skipped（新增 `test_abi.py`、`test_simd.py`、`test_integrate.py`、
  `test_release.py`）。

## v0.1.0

首个可用版本。

### 原创架构
- SunnyNet：CPU 上训练的通用序列架构（非 Transformer / RNN / Mamba / CNN）。
  CenteredScaleNorm、组内成对通道竞争、对数域并行扫描、多尺度衰减。
- 预设 `proto20mw`（d640/i1280/r384/L6）与 `p100mw`（d1408/i1920/r1024/L8）。
- 不变量自检（`tests/test_architecture.py`）：fp64 单步 vs 整段、分段续算、分块不变性、因果性。

### 手写内核
- AVX2/FMA 融合内核（`sunnynet/kernels/`），训练步 1.10–1.17×。
- 通用一阶线性递推 C Runtime（`sunnynet/runtime/recur.c`），对 torch 向量化基线中位 9.90×。

### 推理 Runtime
- 整模型 C 单步解码（`sunnynet/runtime/sunnynet.c`）：B=1 单线程 T=256/1024/2048 为 1.53×/2.05×/1.97×。
- GGUF 只读（`sunnynet/runtime/gguf.py`），外部权重反量化与官方 `.pth` 逐张量一致。
- RWKV-7 整模型 C 内核（`sunnynet/runtime/rwkv7.c`）：对 numpy 参考 5.39×，8 线程 7.49×；
  单步 logits max|diff| 7.25e-05，120 步 argmax 全一致。
- INT4 分组权重量化（`sunnynet/runtime/quant.c`）：端到端 2.83×，体积 729→269 MB。
- 外部权重接入：Mamba-2（`runtime/mamba2.py`）与 DeltaNet（`runtime/deltanet.py`）
  逐 token 单步递推，不经 PyTorch 前向。Mamba-2 对 `transformers` 官方实现
  argmax 24/24；DeltaNet 贪心续写连贯（"The capital of France is the city of Paris…"）。

### 命令行
- `sunnynet` / `python -m sunnynet.cli`：`--ckpt`（SunnyNet）、`--gguf`（外部权重）、
  `--hf-dir`（Mamba-2 / DeltaNet）、`--backend c`、`--gguf-backend c`、`--quant {fp32,int4}`。
- `sunnynet.runtime.load_model(arch, path)`：按架构名加载外部权重。

### 分发
- 发布名 `sunnynet-cx`，导入名 `sunnynet`，命令 `sunnynet`。
  PyPI 上 `sunnynet` 已属他人项目（网络抓包中间件），同名不可注册。
- 发布走 GitHub Actions 的 Trusted Publisher（OIDC，无需长期令牌）；
  tag `v*` 触发 build → twine check → publish。

### 已知边界
- 内核需 AVX2/FMA 与 C 编译器（首次调用自动编译）。
- INT4 质量：英文保持，中文与代码有退化，未配校准。
- Mamba-2 / GLA 目前只有 numpy 参考路径（单线程 45.7 / 85.1 ms/token），尚未 C 化。
  DeltaNet 整模型已进 C（`sn_dn_step`），但受内存带宽限制只有 1.10×。
- DeltaNet 的 C 路径需要预导出的 1.37 GB fp32 blob（`--blob`），不随包分发。
- 0.5B 预设在本机 6GB 显卡为边界态，未发布质量数字。