*[中文版](../cn/ABI.md)*
# Stable C ABI

`sunnynet/runtime/sn_abi.h` defines the stable interface promised to the outside. Only symbols exposed through that header are bound by
versioning; the internal implementation (`sunnynet.h` / `rwkv7.h` / `recur.h` / `quant.h`) is not part of the stable interface and may
change at any time.

## Version policy

Three integers `MAJOR.MINOR.PATCH`, packed as `(MAJOR << 16) | (MINOR << 8) | PATCH`.

| Change type | Version action | Impact on callers |
|---|---|---|
| Breaking (signature change, symbol removal, semantic change) | MAJOR+1 | Recompilation required |
| Backward-compatible addition (new functions, new capability bits) | MINOR+1 | Existing callers need no change |
| Implementation fix, no interface change | PATCH+1 | None |

Callers pass `SN_ABI_VERSION_MAJOR` at compile time to `sn_abi_check()`; a return of `SN_E_UNSUPPORTED`
means the major version does not match and loading should be refused.

All symbols carry the `sn_abi_` prefix, separate from internal symbols, to avoid name collisions and misuse.

## Usage

```c
#include "sn_abi.h"

if (sn_abi_check(SN_ABI_VERSION_MAJOR) != SN_OK) {
    /* incompatible major version */
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

Handles are opaque pointers — callers do not depend on any struct layout, so internal state can be refactored freely.

## Capability bits

Queried bitwise; **unknown bits must be ignored**, which is the precondition for forward compatibility.

| Bit | Name | Corresponding function |
|---|---|---|
| 0 | `SN_CAP_RECUR_SCAN` | `sn_abi_recur_scan` |
| 1 | `SN_CAP_RECUR_STEP` | `sn_abi_recur_step` |
| 2 | `SN_CAP_QUANT_I8` | `sn_abi_quant_i8` / `sn_abi_dequant_i8` |
| 3 | `SN_CAP_RWKV7` | `sn_abi_rwkv7_*` |
| 4 | `SN_CAP_THREADS` | `sn_abi_set_threads` / `sn_abi_get_threads` |

## Error codes

Every function returning `sn_status` returns a negative value on failure. Success is always `SN_OK` (0).

| Code | Name | Meaning |
|---|---|---|
| 0 | `SN_OK` | Success |
| -1 | `SN_E_INVAL` | Null pointer or non-positive dimension |
| -2 | `SN_E_NOMEM` | Allocation failure |
| -3 | `SN_E_IO` | Open or read failure |
| -4 | `SN_E_FORMAT` | Magic number or format version mismatch |
| -5 | `SN_E_UNSUPPORTED` | Not supported by this build |

`sn_status` is fixed to `int32_t` to avoid `int` width differences; array dimension parameters are likewise fixed to `int32_t`.

## Building and loading

The library compiles `sn_abi.c` together with the internal implementations `recur.c`, `rwkv7.c`, `quant.c` into a single shared library
(Windows `.dll` / Linux `.so`), with no extra dependencies.

```python
from sunnynet.runtime import abi
abi.load_abi()          # compiles automatically on first call, reused afterwards
print(abi.version_string(), abi.cap_names())
```

## Test coverage

`tests/test_abi.py` verifies: the version number matches the header, capability bits include the core entries, major-version compatibility
decisions, error-code paths (null pointer / negative dimension / non-positive thread count), bit-exact values of the recurrence kernels,
INT8 quantization round-trip error, and that RWKV-7 logits via the stable ABI and via internal handles are **bit-identical** (max|diff| = 0.0).