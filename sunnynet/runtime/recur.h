/* SunnyNet Runtime: 通用一阶线性递推内核.
 *
 *   s_t = a_t * s_{t-1} + b_t            逐通道独立
 *   gated 模式 b_t = (1 - a_t) * u_t     (u 由调用方给定, 不物化 b)
 *
 * CPU (AVX2/FMA + OpenMP), 不依赖 Python/PyTorch.
 */
#ifndef SN_RECUR_H
#define SN_RECUR_H

#ifdef __cplusplus
extern "C" {
#endif

#define SN_FMA   0   /* b 已给定 */
#define SN_GATED 1   /* b = (1-a)*u */

void sn_set_threads(int n);
int  sn_threads(void);

/* 前向扫描. a, b: (B,T,N) 连续; s0: (B,N); out: (B,T,N); send: (B,N).
 * gated=1 时 b 语义为 u. out 可与 b 复用吗? 不可, 反向依赖 out. */
void sn_recur_scan(const float* a, const float* b, const float* s0,
                   float* out, float* send, int B, int T, int N, int gated);

/* 反向. gated=1 时 da/db 语义为 da/du, 且 da = ds*(s_prev - u), db = ds*(1-a). */
void sn_recur_bwd(const float* a, const float* b, const float* s0,
                  const float* out, const float* dout,
                  float* da, float* db, float* ds0,
                  int B, int T, int N, int gated);

/* 单步解码. a, b: (B,N); s: (B,N) 原地更新. */
void sn_recur_step(const float* a, const float* b, float* s, int B, int N, int gated);

/* 矩阵值 rank-1 递推单步 (逐 head 同维). 元素级形式表达不了 S += v x k 这类
 * 外积累加, 故单列一层. 统一形式:
 *     p[i]      = sum_j S[i][j] * u[j]          (读更新前的 S)
 *     S[i][j]    = S[i][j] * w[j] + v[i]*k[j] - p[i]*b[j]
 *     y[i]      = sum_j S[i][j] * r[j]
 *
 * 参 NULL 即跳过该项: w=NULL 无衰减, u=NULL 跳过投影, b=NULL 无校正,
 * r=NULL 不计算 y。各架构映射:
 *     RWKV-7   w=衰减(列) u=kk  v=v k=k b=kk*a r=r
 *     Mamba-2  衰减可分解, 走元素级 sn_recur_step 即可
 *
 * S: (NH,R,C) 行主序原地更新; w/u/k/b/r: (NH,C); v/y: (NH,R);
 * pbuf: (NH,R) 暂存, 由调用方提供以免内核内分配。
 */
void sn_mat_scan(float* S, const float* w, const float* u, const float* v,
                 const float* k, const float* b, const float* r, float* y,
                 float* pbuf, int NH, int R, int C);

/* 分块对称 INT8 量化. block<=0 取整层一块.
 * q: (B,N) int8, scale: (B,ceil(N/block)) */
void sn_quant_i8(const float* x, signed char* q, float* scale,
                 int B, int N, int block);
void sn_dequant_i8(const signed char* q, const float* scale, float* x,
                   int B, int N, int block);

#ifdef __cplusplus
}
#endif
#endif