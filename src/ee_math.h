/*
 * ee_math.h -- arithmetic of the early-exit pipeline (pure C99)
 * ============================================================
 *
 * Implements exactly the "Exact arithmetic" and "C tail" sections of
 * EE_INTERFACE.md, so the firmware and the Python host reference
 * (export_ee_firmware.py) agree: standardisation, int8 / int16 quantisation
 * and requantisation, stats pooling of an integer tap, the linear exit head,
 * softmax + entropy, argmax, and the float32 tail (bidirectional GRU and
 * attention pooling) that follows stage 3 when EE_HAS_C_TAIL is 1.
 *
 * Integer stage tensors are int8 (EE_ACT_BITS == 8: TFLite full-integer INT8)
 * or int16 (EE_ACT_BITS == 16: TFLite 16x8).  The int16 functions (suffix
 * _i16) apply the same float formulas as their int8 counterparts and clamp to
 * [-32768, 32767] instead of [-128, 127].
 *
 * No Zephyr or TFLM includes and no dynamic allocation (every buffer is the
 * caller's): the same file builds on the host as a shared library for ctypes
 * tests, e.g.
 *     gcc -std=c99 -Wall -Wextra -Werror -O2 -fPIC -shared ee_math.c -o libee_math.so -lm
 * Signatures therefore use only pointers, int and float (ee_gru_weights_t is
 * a struct of four pointers).
 *
 * Buffers never alias unless stated (the requantisations allow q_out == q_in).
 */
#ifndef EE_MATH_H
#define EE_MATH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Added to p before logf in the entropy (EE_INTERFACE.md). */
#define EE_ENTROPY_EPS 1e-12f

/* round_half_away(v) = sign(v) * floor(|v| + 0.5), computed with C99 roundf. */
float ee_round_half_away(float v);

/* q = clamp(round_half_away(x / scale) + zero_point, -128, 127).
 * Defensive: NaN maps to -128. */
int8_t ee_quantise_value(float x, float scale, int zero_point);

/* Dequantise one int8 value: scale * (q - zero_point). */
float ee_dequantise_value(int8_t q, float scale, int zero_point);

/* x_std[t*C + c] = (raw[t*C + c] - mean[c]) / std[c] for t < n_samples,
 * c < n_channels (row-major [t][c]). */
void ee_standardise(const float *raw, const float *mean, const float *std,
                    int n_samples, int n_channels, float *x_std);

/* q[i] = ee_quantise_value(x[i], scale, zero_point) for i < n. */
void ee_quantise(const float *x, int n, float scale, int zero_point, int8_t *q);

/* int8 -> int8 requantisation of n values.
 * scale_in == scale_out && zp_in == zp_out: plain copy (q_out may equal q_in);
 * otherwise q_out = clamp(round_half_away((q_in - zp_in) * (scale_in / scale_out))
 *                         + zp_out, -128, 127).
 * Returns 1 when the copy path was taken, 0 otherwise. */
int ee_requantise(const int8_t *q_in, int n, float scale_in, int zp_in,
                  float scale_out, int zp_out, int8_t *q_out);

/* Stats pooling over time of an int8 tap q[t*P + p] (t < T, p < P) with the
 * given scale / zero point into feat[3P] = [mean; max; std] (population std,
 * two-pass, float accumulation in increasing t). */
void ee_stats_pool(const int8_t *q, int T, int P, float scale, int zero_point,
                   float *feat);

/* logits[j] = b[j] + sum_d feat[d] * W[d*K + j]   (W row-major [D][K]);
 * float accumulation starts at b[j] and adds the terms in increasing d.
 * Also the C tail's dense layer (D = 2H, W = EE_OUT_W, b = EE_OUT_B) and,
 * inside the GRU, the input and recurrent projections. */
void ee_linear_head(const float *feat, int D, const float *W, const float *b,
                    int K, float *logits);

/* probs = softmax(logits) (max-subtracted); returns the entropy in nats,
 * H = -sum_j p_j * logf(p_j + EE_ENTROPY_EPS).  probs may not alias logits. */
float ee_softmax_entropy(const float *logits, int K, float *probs);

/* Index of the first maximum (ties resolve to the lowest index). */
int ee_argmax_f32(const float *v, int n);
int ee_argmax_i8(const int8_t *v, int n);

/* ---- int16 stage tensors (EE_ACT_BITS == 16) ----------------------------
 * Same formulas as the int8 functions above; the clamping range is
 * [-32768, 32767] (NaN maps to -32768).  Pooling sums are int32, exact for
 * T * 32768 < 2^31, i.e. any T < 65536. */
int16_t ee_quantise_value_i16(float x, float scale, int zero_point);
float ee_dequantise_value_i16(int16_t q, float scale, int zero_point);
void ee_quantise_i16(const float *x, int n, float scale, int zero_point, int16_t *q);
/* Returns 1 when the copy path was taken (equal scale and zero point), else 0. */
int ee_requantise_i16(const int16_t *q_in, int n, float scale_in, int zp_in,
                      float scale_out, int zp_out, int16_t *q_out);
void ee_stats_pool_i16(const int16_t *q, int T, int P, float scale, int zero_point,
                       float *feat);
int ee_argmax_i16(const int16_t *v, int n);

/* ---- C tail (EE_HAS_C_TAIL == 1): float32 BiGRU + attention pooling ------
 * The tail reads stage 3's integer output q[t*D + d] ([T][D], row-major) and
 * dequantises it one time step at a time, x[d] = scale * (q - zero_point), so
 * there is no float copy of the whole sequence.  The dense softmax that ends
 * the tail is ee_linear_head + ee_softmax_entropy + ee_argmax_f32. */

/* Weights of one GRU direction (Keras GRU, reset_after=True; gate blocks
 * z | r | n of H columns each, in that order). */
typedef struct {
    const float *W;   /* [D][3H] row-major: input kernel        */
    const float *U;   /* [H][3H] row-major: recurrent kernel    */
    const float *bi;  /* [3H] input bias     (Keras bias row 0) */
    const float *br;  /* [3H] recurrent bias (Keras bias row 1) */
} ee_gru_weights_t;

/* Floats of the caller's scratch for ee_bigru_i8 / ee_bigru_i16:
 * x[D] (one dequantised step), h[H], xp[3H], hp[3H]. */
#define EE_BIGRU_SCRATCH_FLOATS(D, H) ((D) + 7 * (H))

/* Bidirectional GRU with zero initial state, H units per direction, over the
 * T x D integer sequence q.  For every step:
 *   xp[j] = bi[j] + sum_d x[d] * W[d*3H + j]   (j < 3H, increasing d)
 *   hp[j] = br[j] + sum_u h[u] * U[u*3H + j]   (h of the previous step)
 *   z = sig(xp[j] + hp[j]), r = sig(xp[H+j] + hp[H+j]),
 *   n = tanhf(xp[2H+j] + r * hp[2H+j]), h[j] = z * h[j] + (1 - z) * n   (j < H)
 * with sig(v) = 1 / (1 + expf(-v)).  The forward direction runs t = 0..T-1
 * and writes hseq[t*2H + j]; the backward direction runs t = T-1..0 and
 * writes hseq[t*2H + H + j]; hseq ([T][2H]) is thus in time order.
 * scratch: EE_BIGRU_SCRATCH_FLOATS(D, H) floats. */
void ee_bigru_i8(const int8_t *q, int T, int D, float scale, int zero_point, int H,
                 const ee_gru_weights_t *fwd, const ee_gru_weights_t *bwd,
                 float *hseq, float *scratch);
void ee_bigru_i16(const int16_t *q, int T, int D, float scale, int zero_point, int H,
                  const ee_gru_weights_t *fwd, const ee_gru_weights_t *bwd,
                  float *hseq, float *scratch);

/* Additive attention pooling over time of hseq[t*C + c] (t < T, c < C):
 *   score_t = b + sum_c hseq[t][c] * w[c]   (increasing c)
 *   e_t = expf(score_t - max_t score_t)
 *   feat[c] = sum_t (e_t / sum_t e_t) * hseq[t][c]   (increasing t)
 * scores: T floats of scratch (holds e_t on return). */
void ee_attention_pool(const float *hseq, int T, int C, const float *w, float b,
                       float *scores, float *feat);

#ifdef __cplusplus
}
#endif

#endif /* EE_MATH_H */
