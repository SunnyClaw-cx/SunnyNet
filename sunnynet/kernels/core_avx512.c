/* SunnyNet 融合 core 内核的 AVX-512F/BW/VL 版本.
 *
 * 与 core_avx2.c 数值等价: group 仍是 8, 只是把两个 group 并进一个 512 位寄存器
 * (低 8 lane = 组 A, 高 8 lane = 组 B), 组内置换用 16 lane 的 lane-swap,
 * 两组互不干扰。这样元素宽度翻倍, 且不必改 group 语义。
 *
 * 组内相邻元素靠 _mm512_permutexvar_ps 完成; 掩码只用来处理 D 不是 16 倍数时的尾部。
 *
 * 编译: gcc -O3 -mavx512f -mavx512bw -mavx512vl -mfma -funroll-loops -fopenmp
 * 仅能在支持 AVX-512F 的 CPU 上执行。
 */
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <omp.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void sn512_set_threads(int n) { omp_set_num_threads(n); }
EXPORT int sn512_threads(void) { return omp_get_max_threads(); }
EXPORT const char* sn512_isa(void) { return "avx512f+bw+vl+fma"; }

/* 16 lane 的组内置换索引: 低 8 = 组 A 的 lane 偏移, 高 8 = 组 B 的同一偏移。
 * off 取 0..7 表示 v[(i+off)&7]; 符号模式用 8..15 表示 v[(i-（16-off))&7].
 * 这里直接按 out[i] = in[(i + step) mod 8] 生成, step 可为负。
 */
static inline __m512i idx_shift(int step) {
    int idx[16];
    for (int half = 0; half < 2; ++half) {
        for (int i = 0; i < 8; ++i) {
            int j = (i + step) & 7;   /* C 的 % 对负数向零取整, 故用 &7 */
            idx[half * 8 + i] = half * 8 + j;
        }
    }
    return _mm512_loadu_si512((const void*)idx);
}

static inline __m512 expv16(__m512 x) {
    float a[16];
    _mm512_storeu_ps(a, x);
    for (int i = 0; i < 16; ++i) a[i] = expf(a[i]);
    return _mm512_loadu_ps(a);
}

/* 忽略尾部掩码: 只对满 16 宽度的块调用, 收尾走 core512_fwd_tail_f32. */
EXPORT void core512_fwd_f32(const float* q, const float* delta, const float* s0,
                            float* states, float* uo, float* po,
                            int B, int T, int D) {
    const int NG = D >> 3;                 /* group 数, 每 group 8 元素 */
    const int NP = NG >> 1;                /* 两个 group 一并处理的次数 */
    const __m512i il1 = idx_shift(1);
    const __m512i il2 = idx_shift(2);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int gp = 0; gp < NP; ++gp) {
            const int d0 = gp << 4;        /* 16 个连续元素 */
            const size_t qb = (size_t)b * T * 2 * D;
            const size_t sb = (size_t)b * T * D;
            __m512 s = _mm512_loadu_ps(s0 + (size_t)b * D + d0);
            __m512 dl = _mm512_loadu_ps(delta + d0);
            __m512 base = _mm512_sub_ps(one, dl);
            for (int t = 0; t < T; ++t) {
                const size_t qoff = qb + (size_t)t * 2 * D + d0;
                const size_t off = sb + (size_t)t * D + d0;
                __m512 ur = _mm512_loadu_ps(q + qoff);
                __m512 gt = _mm512_loadu_ps(q + qoff + D);
                __m512 e = expv16(_mm512_sub_ps(zero, gt));
                __m512 e1 = _mm512_permutexvar_ps(il1, e);
                __m512 e2 = _mm512_permutexvar_ps(il2, e);
                __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, e));
                __m512 s1 = _mm512_div_ps(one, _mm512_add_ps(one, _mm512_div_ps(e, e1)));
                __m512 s2 = _mm512_div_ps(one, _mm512_add_ps(one, _mm512_div_ps(e, e2)));
                __m512 w = _mm512_mul_ps(s1, s2);
                __m512 u = _mm512_mul_ps(ur, w);
                __m512 p = _mm512_fmadd_ps(dl, sig, base);
                s = _mm512_fmadd_ps(p, s, _mm512_mul_ps(_mm512_sub_ps(one, p), u));
                _mm512_storeu_ps(states + off, s);
                _mm512_storeu_ps(uo + off, u);
                _mm512_storeu_ps(po + off, p);
            }
        }
    }
}

/* NG 为奇数时的单个 group 收尾 (8 元素), 保持与 AVX2 版同序. */
EXPORT void core512_fwd_tail_f32(const float* q, const float* delta,
                                 const float* s0, float* states,
                                 float* uo, float* po,
                                 int B, int T, int D, int gstart) {
    const __m256i il1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    const __m256i il2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (int b = 0; b < B; ++b) {
        for (int g = gstart; g < (D >> 3); ++g) {
            const int d0 = g << 3;
            const size_t qb = (size_t)b * T * 2 * D;
            const size_t sb = (size_t)b * T * D;
            __m256 s = _mm256_loadu_ps(s0 + (size_t)b * D + d0);
            __m256 dl = _mm256_loadu_ps(delta + d0);
            __m256 base = _mm256_sub_ps(one, dl);
            for (int t = 0; t < T; ++t) {
                const size_t qoff = qb + (size_t)t * 2 * D + d0;
                const size_t off = sb + (size_t)t * D + d0;
                __m256 ur = _mm256_loadu_ps(q + qoff);
                __m256 gt = _mm256_loadu_ps(q + qoff + D);
                float tmp[8];
                _mm256_storeu_ps(tmp, _mm256_sub_ps(zero, gt));
                for (int i = 0; i < 8; ++i) tmp[i] = expf(tmp[i]);
                __m256 e = _mm256_loadu_ps(tmp);
                __m256 e1 = _mm256_permutevar8x32_ps(e, il1);
                __m256 e2 = _mm256_permutevar8x32_ps(e, il2);
                __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
                __m256 s1 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e1)));
                __m256 s2 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e2)));
                __m256 w = _mm256_mul_ps(s1, s2);
                __m256 u = _mm256_mul_ps(ur, w);
                __m256 p = _mm256_fmadd_ps(dl, sig, base);
                s = _mm256_fmadd_ps(p, s, _mm256_mul_ps(_mm256_sub_ps(one, p), u));
                _mm256_storeu_ps(states + off, s);
                _mm256_storeu_ps(uo + off, u);
                _mm256_storeu_ps(po + off, p);
            }
        }
    }
}

/* 反向, 语义同 core_bwd_f32. 每 16 宽度块含两个 group. */
EXPORT void core512_bwd_f32(const float* q, const float* delta, const float* s0,
                            const float* states, const float* uo, const float* po,
                            const float* dstate, const float* du_in,
                            float* dq, float* ddelta, float* ds0,
                            int B, int T, int D) {
    const int NG = D >> 3;
    const int NP = NG >> 1;
    const __m512i il1 = idx_shift(1);
    const __m512i il2 = idx_shift(2);
    const __m512i ir1 = idx_shift(-1);
    const __m512i ir2 = idx_shift(-2);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int gp = 0; gp < NP; ++gp) {
            const int d0 = gp << 4;
            const size_t qb = (size_t)b * T * 2 * D;
            const size_t sb = (size_t)b * T * D;
            const float* s0b = s0 + (size_t)b * D + d0;
            __m512 dl = _mm512_loadu_ps(delta + d0);
            __m512 dacc = _mm512_setzero_ps();
            __m512 ds = _mm512_setzero_ps();
            for (int t = T - 1; t >= 0; --t) {
                const size_t qoff = qb + (size_t)t * 2 * D + d0;
                const size_t off = sb + (size_t)t * D + d0;
                ds = _mm512_add_ps(ds, _mm512_loadu_ps(dstate + off));
                __m512 u = _mm512_loadu_ps(uo + off);
                __m512 p = _mm512_loadu_ps(po + off);
                __m512 sprev = (t > 0) ? _mm512_loadu_ps(states + off - D)
                                       : _mm512_loadu_ps(s0b);
                __m512 dp = _mm512_mul_ps(ds, _mm512_sub_ps(sprev, u));
                __m512 du = _mm512_fmadd_ps(ds, _mm512_sub_ps(one, p),
                                            _mm512_loadu_ps(du_in + off));
                ds = _mm512_mul_ps(ds, p);

                __m512 ur = _mm512_loadu_ps(q + qoff);
                __m512 gt = _mm512_loadu_ps(q + qoff + D);
                __m512 e = expv16(_mm512_sub_ps(zero, gt));
                __m512 e1 = _mm512_permutexvar_ps(il1, e);
                __m512 e2 = _mm512_permutexvar_ps(il2, e);
                __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, e));
                __m512 s1 = _mm512_div_ps(one, _mm512_add_ps(one, _mm512_div_ps(e, e1)));
                __m512 s2 = _mm512_div_ps(one, _mm512_add_ps(one, _mm512_div_ps(e, e2)));
                __m512 w = _mm512_mul_ps(s1, s2);

                __m512 dur = _mm512_mul_ps(du, w);
                __m512 dw = _mm512_mul_ps(du, ur);
                __m512 ds1 = _mm512_mul_ps(dw, s2);
                __m512 ds2 = _mm512_mul_ps(dw, s1);
                __m512 a1 = _mm512_mul_ps(ds1, _mm512_mul_ps(s1, _mm512_sub_ps(one, s1)));
                __m512 a2 = _mm512_mul_ps(ds2, _mm512_mul_ps(s2, _mm512_sub_ps(one, s2)));
                __m512 dg = _mm512_add_ps(a1, a2);
                dg = _mm512_sub_ps(dg, _mm512_permutexvar_ps(ir1, a1));
                dg = _mm512_sub_ps(dg, _mm512_permutexvar_ps(ir2, a2));
                dg = _mm512_fmadd_ps(_mm512_mul_ps(dp, dl),
                                     _mm512_mul_ps(sig, _mm512_sub_ps(one, sig)), dg);
                dacc = _mm512_fmadd_ps(dp, _mm512_sub_ps(sig, one), dacc);

                _mm512_storeu_ps(dq + qoff, dur);
                _mm512_storeu_ps(dq + qoff + D, dg);
            }
            _mm512_storeu_ps(ds0 + (size_t)b * D + d0, ds);
            float tmp[16];
            _mm512_storeu_ps(tmp, dacc);
            for (int i = 0; i < 16; ++i) {
                #pragma omp atomic
                ddelta[d0 + i] += tmp[i];
            }
        }
    }
}

/* NG 为奇数时的单个 group 收尾 (8 元素), 语义同 core_bwd_f32. */
EXPORT void core512_bwd_tail_f32(const float* q, const float* delta,
                                 const float* s0, const float* states,
                                 const float* uo, const float* po,
                                 const float* dstate, const float* du_in,
                                 float* dq, float* ddelta, float* ds0,
                                 int B, int T, int D, int gstart) {
    const __m256i il1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    const __m256i il2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
    const __m256i ir1 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
    const __m256i ir2 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (int b = 0; b < B; ++b) {
        for (int g = gstart; g < (D >> 3); ++g) {
            const int d0 = g << 3;
            const size_t qb = (size_t)b * T * 2 * D;
            const size_t sb = (size_t)b * T * D;
            const float* s0b = s0 + (size_t)b * D + d0;
            __m256 dl = _mm256_loadu_ps(delta + d0);
            __m256 dacc = _mm256_setzero_ps();
            __m256 ds = _mm256_setzero_ps();
            for (int t = T - 1; t >= 0; --t) {
                const size_t qoff = qb + (size_t)t * 2 * D + d0;
                const size_t off = sb + (size_t)t * D + d0;
                ds = _mm256_add_ps(ds, _mm256_loadu_ps(dstate + off));
                __m256 u = _mm256_loadu_ps(uo + off);
                __m256 p = _mm256_loadu_ps(po + off);
                __m256 sprev = (t > 0) ? _mm256_loadu_ps(states + off - D)
                                       : _mm256_loadu_ps(s0b);
                __m256 dp = _mm256_mul_ps(ds, _mm256_sub_ps(sprev, u));
                __m256 du = _mm256_fmadd_ps(ds, _mm256_sub_ps(one, p),
                                            _mm256_loadu_ps(du_in + off));
                ds = _mm256_mul_ps(ds, p);

                __m256 ur = _mm256_loadu_ps(q + qoff);
                __m256 gt = _mm256_loadu_ps(q + qoff + D);
                float tmp[8];
                _mm256_storeu_ps(tmp, _mm256_sub_ps(zero, gt));
                for (int i = 0; i < 8; ++i) tmp[i] = expf(tmp[i]);
                __m256 e = _mm256_loadu_ps(tmp);
                __m256 e1 = _mm256_permutevar8x32_ps(e, il1);
                __m256 e2 = _mm256_permutevar8x32_ps(e, il2);
                __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
                __m256 s1 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e1)));
                __m256 s2 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e2)));
                __m256 w = _mm256_mul_ps(s1, s2);

                __m256 dur = _mm256_mul_ps(du, w);
                __m256 dw = _mm256_mul_ps(du, ur);
                __m256 ds1 = _mm256_mul_ps(dw, s2);
                __m256 ds2 = _mm256_mul_ps(dw, s1);
                __m256 a1 = _mm256_mul_ps(ds1, _mm256_mul_ps(s1, _mm256_sub_ps(one, s1)));
                __m256 a2 = _mm256_mul_ps(ds2, _mm256_mul_ps(s2, _mm256_sub_ps(one, s2)));
                __m256 dg = _mm256_add_ps(a1, a2);
                dg = _mm256_sub_ps(dg, _mm256_permutevar8x32_ps(a1, ir1));
                dg = _mm256_sub_ps(dg, _mm256_permutevar8x32_ps(a2, ir2));
                dg = _mm256_fmadd_ps(_mm256_mul_ps(dp, dl),
                                     _mm256_mul_ps(sig, _mm256_sub_ps(one, sig)), dg);
                dacc = _mm256_fmadd_ps(dp, _mm256_sub_ps(sig, one), dacc);

                _mm256_storeu_ps(dq + qoff, dur);
                _mm256_storeu_ps(dq + qoff + D, dg);
            }
            _mm256_storeu_ps(ds0 + (size_t)b * D + d0, ds);
            float t2[8];
            _mm256_storeu_ps(t2, dacc);
            for (int i = 0; i < 8; ++i) {
                #pragma omp atomic
                ddelta[d0 + i] += t2[i];
            }
        }
    }
}

/* 整段前向: 内部处理 NG 为奇数时的收尾, 调用方不必自己分两次调. */
EXPORT void core512_fwd_all_f32(const float* q, const float* delta, const float* s0,
                                float* states, float* uo, float* po,
                                int B, int T, int D) {
    const int NG = D >> 3;
    core512_fwd_f32(q, delta, s0, states, uo, po, B, T, D);
    if (NG & 1)
        core512_fwd_tail_f32(q, delta, s0, states, uo, po, B, T, D, NG - 1);
}

/* 整段反向: 内部处理 NG 为奇数时的收尾. */
EXPORT void core512_bwd_all_f32(const float* q, const float* delta, const float* s0,
                                const float* states, const float* uo, const float* po,
                                const float* dstate, const float* du_in,
                                float* dq, float* ddelta, float* ds0,
                                int B, int T, int D) {
    const int NG = D >> 3;
    core512_bwd_f32(q, delta, s0, states, uo, po, dstate, du_in,
                    dq, ddelta, ds0, B, T, D);
    if (NG & 1)
        core512_bwd_tail_f32(q, delta, s0, states, uo, po, dstate, du_in,
                             dq, ddelta, ds0, B, T, D, NG - 1);
}

/* 运行期探测: 本库需 CPU 支持 AVX512F, 否则调用会触发非法指令. */
EXPORT int sn512_cpu_supported(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx, maxleaf;
#if defined(_MSC_VER)
    int r0[4]; __cpuid(r0, 0); maxleaf = (unsigned)r0[0];
    int r7[4]; __cpuid(r7, 7);
    ebx = (unsigned)r7[1];
    (void)eax; (void)ecx; (void)edx;
#else
    __asm__ volatile("cpuid" : "=a"(maxleaf) : "a"(0) : "ebx", "ecx", "edx");
    __asm__ volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(7), "c"(0));
#endif
    if (maxleaf < 7) return 0;
    return (ebx >> 16) & 1;
#else
    return 0;
#endif
}