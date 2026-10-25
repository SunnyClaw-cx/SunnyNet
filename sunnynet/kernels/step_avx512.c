/* SunnyNet 推理态单步内核的 AVX-512F/BW/VL 版本.
 *
 * 与 step_avx2.c 数值等价:
 *   dw_step512 是纯逐元素, 16 lane 直用;
 *   step_core512 有 group=8 语义 (组内成对竞争), 故一个 512 寄存器装两个 group,
 *   组内置换走 16 lane 的 lane-swap, 两半区互不干扰。
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

/* 与 core_avx512.c 合编成一个 .so 时, 公共符号由对方提供 (SN_512_NO_COMMON). */
#ifndef SN_512_NO_COMMON
EXPORT void sn512_set_threads(int n) { omp_set_num_threads(n); }
EXPORT int sn512_threads(void) { return omp_get_max_threads(); }
EXPORT const char* sn512_isa(void) { return "avx512f+bw+vl+fma"; }
#endif

static inline __m512i idx_shift512(int step) {
    int idx[16];
    for (int half = 0; half < 2; ++half)
        for (int i = 0; i < 8; ++i)
            idx[half * 8 + i] = half * 8 + ((i + step) & 7);
    return _mm512_loadu_si512((const void*)idx);
}

static inline __m512 expv16(__m512 x) {
    float a[16];
    _mm512_storeu_ps(a, x);
    for (int i = 0; i < 16; ++i) a[i] = expf(a[i]);
    return _mm512_loadu_ps(a);
}

/* 深度卷积单步: w 布局 (taps, D); hist[:,0] 是上一时刻输入.
 * out[d] = w[taps-1][d]*x[d] + sum_j w[taps-2-j][d]*hist[j][d] */
EXPORT void dw_step512_f32(const float* x, const float* hist, const float* w,
                           float* acc, int B, int D, int taps) {
    const int k = taps - 1;
    #pragma omp parallel for schedule(static) if(B * (D >> 4) >= 256)
    for (int b = 0; b < B; ++b) {
        const float* xb = x + (size_t)b * D;
        const float* hb = hist + (size_t)b * k * D;
        float* ab = acc + (size_t)b * D;
        const float* wt = w + (size_t)(taps - 1) * D;
        int d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 a = _mm512_mul_ps(_mm512_loadu_ps(xb + d),
                                     _mm512_loadu_ps(wt + d));
            for (int j = 0; j < k; ++j) {
                __m512 hv = _mm512_loadu_ps(hb + (size_t)j * D + d);
                __m512 wv = _mm512_loadu_ps(w + (size_t)(taps - 2 - j) * D + d);
                a = _mm512_fmadd_ps(hv, wv, a);
            }
            _mm512_storeu_ps(ab + d, a);
        }
        if (d < D) {
            const __mmask16 m = (__mmask16)((1u << (D - d)) - 1);
            __m512 a = _mm512_mul_ps(_mm512_maskz_loadu_ps(m, xb + d),
                                     _mm512_maskz_loadu_ps(m, wt + d));
            for (int j = 0; j < k; ++j) {
                __m512 hv = _mm512_maskz_loadu_ps(m, hb + (size_t)j * D + d);
                __m512 wv = _mm512_maskz_loadu_ps(m, w + (size_t)(taps - 2 - j) * D + d);
                a = _mm512_fmadd_ps(hv, wv, a);
            }
            _mm512_mask_storeu_ps(ab + d, m, a);
        }
    }
}

/* q (B,2D) 的 [0,D)=u_raw, [D,2D)=gate; s (B,D) -> s_new, u_out; delta (D,). */
EXPORT void step_core512_f32(const float* q, const float* delta, const float* s,
                             float* s_new, float* u_out, int B, int D) {
    const int NG = D >> 3;
    const int NP = NG >> 1;
    const __m512i il1 = idx_shift512(1);
    const __m512i il2 = idx_shift512(2);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    const int work = B * NP;
    #pragma omp parallel for schedule(static) if(work >= 128)
    for (int b = 0; b < B; ++b) {
        const float* qb = q + (size_t)b * 2 * D;
        const float* sb = s + (size_t)b * D;
        float* sbn = s_new + (size_t)b * D;
        float* ub = u_out + (size_t)b * D;
        for (int gp = 0; gp < NP; ++gp) {
            const int d0 = gp << 4;
            __m512 ur = _mm512_loadu_ps(qb + d0);
            __m512 gt = _mm512_loadu_ps(qb + D + d0);
            __m512 dl = _mm512_loadu_ps(delta + d0);
            __m512 sv = _mm512_loadu_ps(sb + d0);
            __m512 e = expv16(_mm512_sub_ps(zero, gt));
            __m512 e1 = _mm512_permutexvar_ps(il1, e);
            __m512 e2 = _mm512_permutexvar_ps(il2, e);
            __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, e));
            __m512 s1 = _mm512_div_ps(one, _mm512_add_ps(one, _mm512_div_ps(e, e1)));
            __m512 s2 = _mm512_div_ps(one, _mm512_add_ps(one, _mm512_div_ps(e, e2)));
            __m512 u = _mm512_mul_ps(ur, _mm512_mul_ps(s1, s2));
            __m512 p = _mm512_fmadd_ps(dl, sig, _mm512_sub_ps(one, dl));
            __m512 sn = _mm512_fmadd_ps(p, sv, _mm512_mul_ps(_mm512_sub_ps(one, p), u));
            _mm512_storeu_ps(ub + d0, u);
            _mm512_storeu_ps(sbn + d0, sn);
        }
    }
}

/* NG 为奇数时的最后一个 group (8 元素), 语义同 step_core_f32. */
EXPORT void step_core512_tail_f32(const float* q, const float* delta,
                                  const float* s, float* s_new, float* u_out,
                                  int B, int D, int gstart) {
    const __m256i il1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    const __m256i il2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (int b = 0; b < B; ++b) {
        const float* qb = q + (size_t)b * 2 * D;
        const float* sb = s + (size_t)b * D;
        float* sbn = s_new + (size_t)b * D;
        float* ub = u_out + (size_t)b * D;
        for (int g = gstart; g < (D >> 3); ++g) {
            const int d0 = g << 3;
            __m256 ur = _mm256_loadu_ps(qb + d0);
            __m256 gt = _mm256_loadu_ps(qb + D + d0);
            __m256 dl = _mm256_loadu_ps(delta + d0);
            __m256 sv = _mm256_loadu_ps(sb + d0);
            float tmp[8];
            _mm256_storeu_ps(tmp, _mm256_sub_ps(zero, gt));
            for (int i = 0; i < 8; ++i) tmp[i] = expf(tmp[i]);
            __m256 e = _mm256_loadu_ps(tmp);
            __m256 e1 = _mm256_permutevar8x32_ps(e, il1);
            __m256 e2 = _mm256_permutevar8x32_ps(e, il2);
            __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
            __m256 s1 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e1)));
            __m256 s2 = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_div_ps(e, e2)));
            __m256 u = _mm256_mul_ps(ur, _mm256_mul_ps(s1, s2));
            __m256 p = _mm256_fmadd_ps(dl, sig, _mm256_sub_ps(one, dl));
            __m256 sn = _mm256_fmadd_ps(p, sv, _mm256_mul_ps(_mm256_sub_ps(one, p), u));
            _mm256_storeu_ps(ub + d0, u);
            _mm256_storeu_ps(sbn + d0, sn);
        }
    }
}

/* 单步整段入口: 内部处理 NG 为奇数时的收尾. */
EXPORT void step_core512_all_f32(const float* q, const float* delta, const float* s,
                                 float* s_new, float* u_out, int B, int D) {
    const int NG = D >> 3;
    step_core512_f32(q, delta, s, s_new, u_out, B, D);
    if (NG & 1)
        step_core512_tail_f32(q, delta, s, s_new, u_out, B, D, NG - 1);
}

#ifndef SN_512_NO_COMMON
EXPORT int sn512_cpu_supported(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx, maxleaf;
#if defined(_MSC_VER)
    int r0[4]; __cpuid(r0, 0); maxleaf = (unsigned)r0[0];
    int r7[4]; __cpuid(r7, 7); ebx = (unsigned)r7[1];
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
#endif