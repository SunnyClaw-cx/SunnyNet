# Stable C ABI

`sunnynet/runtime/sn_abi.h` 定义对外承诺的稳定接口。只经该头文件暴露的符号受版本约束；
内部实现（`sunnynet.h` / `rwkv7.h` / `recur.h` / `quant.h`）不属于稳定接口，可随时改动。

## 版本策略

三个整数 `MAJOR.MINOR.PATCH`，打包成 `(MAJOR << 16) | (MINOR << 8) | PATCH`。

| 变更类型 | 版本动作 | 调用方影响 |
|---|---|---|
| 破坏性（改签名、删符号、改语义） | MAJOR+1 | 必须重新编译 |
| 向后兼容新增（新函数、新能力位） | MINOR+1 | 旧调用方无需改动 |
| 修正实现、无接口变化 | PATCH+1 | 无 |

调用方在编译期拿 `SN_ABI_VERSION_MAJOR` 传入 `sn_abi_check()`；返回 `SN_E_UNSUPPORTED`
即主版本不匹配，应当拒绝加载。

符号一律带 `sn_abi_` 前缀，与内部符号分离，避免重名和误用。

## 使用

```c
#include "sn_abi.h"

if (sn_abi_check(SN_ABI_VERSION_MAJOR) != SN_OK) {
    /* 主版本不兼容 */
}

uint32_t caps = sn_abi_caps();
if (caps & SN_CAP_RWKV7) {
    SnAbiRwkv7* m = sn_abi_rwkv7_load("model.runtime.bin");
    int32_t V = sn_abi_rwkv7_vocab(m);
    float* logits = malloc(sizeof(float) * V);
    sn_abi_rwkv7_step(m, token, logits);
    sn_abi_rwkv7_free(m);
    free(logits);
}
```

句柄为不透明指针——调用方不依赖任何结构体布局，因此内部状态可以自由重构。

## 能力位

按位查询；**未知位必须忽略**，这是前向兼容的前提。

| 位 | 名称 | 对应函数 |
|---|---|---|
| 0 | `SN_CAP_RECUR_SCAN` | `sn_abi_recur_scan` |
| 1 | `SN_CAP_RECUR_STEP` | `sn_abi_recur_step` |
| 2 | `SN_CAP_QUANT_I8` | `sn_abi_quant_i8` / `sn_abi_dequant_i8` |
| 3 | `SN_CAP_RWKV7` | `sn_abi_rwkv7_*` |
| 4 | `SN_CAP_THREADS` | `sn_abi_set_threads` / `sn_abi_get_threads` |

## 错误码

每个返回 `sn_status` 的函数失败时返回负值。成功恒为 `SN_OK`（0）。

| 码 | 名字 | 含义 |
|---|---|---|
| 0 | `SN_OK` | 成功 |
| -1 | `SN_E_INVAL` | 空指针或非正维度 |
| -2 | `SN_E_NOMEM` | 分配失败 |
| -3 | `SN_E_IO` | 打开或读取失败 |
| -4 | `SN_E_FORMAT` | 魔数或格式版本不匹配 |
| -5 | `SN_E_UNSUPPORTED` | 当前构建不支持 |

`sn_status` 固定为 `int32_t`，避免 `int` 宽度差异；数组维度参数同样固定 `int32_t`。

## 编译与加载

库把 `sn_abi.c` 与内部实现 `recur.c`、`rwkv7.c`、`quant.c` 一起编译成单个共享库
（Windows `.dll` / Linux `.so`），不额外引入依赖。

```python
from sunnynet.runtime import abi
abi.load_abi()          # 首次调用自动编译, 之后复用
print(abi.version_string(), abi.cap_names())
```

## 测试覆盖

`tests/test_abi.py` 校验：版本号与头文件一致、能力位含核心项、主版本兼容性判定、
错误码路径（空指针/负维度/非正线程数）、递推内核逐位数值、INT8 量化往返误差，
以及 RWKV-7 经稳定 ABI 与经内部句柄的 logits **逐位一致**（max|diff| = 0.0）。
---

# Stable C ABI

`sunnynet/runtime/sn_abi.h` defines the stable interface we commit to externally. Only symbols
exposed through that header are bound to the version; the internal implementations
(`sunnynet.h` / `rwkv7.h` / `recur.h` / `quant.h`) are not part of the stable interface and may
change at any time.

## Versioning policy

Three integers `MAJOR.MINOR.PATCH`, packed into `(MAJOR << 16) | (MINOR << 8) | PATCH`.

| Change type | Version action | Impact on callers |
|---|---|---|
| Breaking (signature change, symbol removal, semantic change) | MAJOR+1 | must recompile |
| Backward-compatible addition (new functions, new capability bits) | MINOR+1 | existing callers need no changes |
| Implementation fix, no interface change | PATCH+1 | none |

Callers pass `SN_ABI_VERSION_MAJOR` (available at compile time) into `sn_abi_check()`; a return
of `SN_E_UNSUPPORTED` means the major version does not match, and the load should be rejected.

All symbols carry the `sn_abi_` prefix, separated from internal symbols, avoiding name collisions
and misuse.

## Usage

```c
#include "sn_abi.h"

if (sn_abi_check(SN_ABI_VERSION_MAJOR) != SN_OK) {
    /* major version incompatible */
}

uint32_t caps = sn_abi_caps();
if (caps & SN_CAP_RWKV7) {
    SnAbiRwkv7* m = sn_abi_rwkv7_load("model.runtime.bin");
    int32_t V = sn_abi_rwkv7_vocab(m);
    float* logits = malloc(sizeof(float) * V);
    sn_abi_rwkv7_step(m, token, logits);
    sn_abi_rwkv7_free(m);
    free(logits);
}
```

Handles are opaque pointers - callers do not depend on any struct layout, so internal state can
be refactored freely.

## Capability bits

Queried bit by bit; **unknown bits must be ignored** - that is the prerequisite for forward
compatibility.

| Bit | Name | Corresponding function |
|---|---|---|
| 0 | `SN_CAP_RECUR_SCAN` | `sn_abi_recur_scan` |
| 1 | `SN_CAP_RECUR_STEP` | `sn_abi_recur_step` |
| 2 | `SN_CAP_QUANT_I8` | `sn_abi_quant_i8` / `sn_abi_dequant_i8` |
| 3 | `SN_CAP_RWKV7` | `sn_abi_rwkv7_*` |
| 4 | `SN_CAP_THREADS` | `sn_abi_set_threads` / `sn_abi_get_threads` |

## Error codes

Every function returning `sn_status` returns a negative value on failure. Success is always
`SN_OK` (0).

| Code | Name | Meaning |
|---|---|---|
| 0 | `SN_OK` | success |
| -1 | `SN_E_INVAL` | null pointer or non-positive dimension |
| -2 | `SN_E_NOMEM` | allocation failed |
| -3 | `SN_E_IO` | open or read failed |
| -4 | `SN_E_FORMAT` | magic number or format version mismatch |
| -5 | `SN_E_UNSUPPORTED` | not supported by the current build |

`sn_status` is fixed at `int32_t` to avoid `int` width differences; array dimension parameters
are likewise fixed at `int32_t`.

## Compilation and loading

The library compiles `sn_abi.c` together with the internal implementations `recur.c`, `rwkv7.c`,
`quant.c` into a single shared library (Windows `.dll` / Linux `.so`), with no additional
dependencies.

```python
from sunnynet.runtime import abi
abi.load_abi()          # compiles automatically on first call, reused afterwards
print(abi.version_string(), abi.cap_names())
```

## Test coverage

`tests/test_abi.py` verifies: version number matches the header, capability bits include the core
items, major-version compatibility verdict, error-code paths (null pointer / negative dimension /
non-positive thread count), bit-exact numerics of the recurrence kernel, INT8 quantization
round-trip error, and that RWKV-7 logits through the stable ABI and through the internal handle
are **bit-identical** (max|diff| = 0.0).
