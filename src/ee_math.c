/*
 * ee_math.c -- arithmetic of the early-exit pipeline (pure C99)
 *
 * See ee_math.h and the "Exact arithmetic" and "C tail" sections of
 * EE_INTERFACE.md.  The loop orders below are part of the contract where
 * float accumulation is involved (pooling variance, head sums, softmax sums,
 * GRU projections, attention sums): each accumulates in increasing index
 * order.  CMakeLists.txt builds this file with -ffp-contract=off so the
 * Cortex-M33 does not fuse multiply-adds and the rounding matches the host
 * reference.
 */
#include "ee_math.h"

#include <math.h>
#include <string.h>

/* Clamp an integral float to the int8 range; NaN goes to -128. */
static int8_t clamp_to_int8(float r)
{
    if (!(r > -128.0f)) {
        return (int8_t)-128;
    }
    if (r > 127.0f) {
        return (int8_t)127;
    }
    return (int8_t)r;
}

float ee_round_half_away(float v)
{
    return roundf(v);
}

int8_t ee_quantise_value(float x, float scale, int zero_point)
{
    /* roundf(x / scale) is integral; adding zero_point in float is exact
     * whenever the result can land in [-128, 127]. */
    return clamp_to_int8(roundf(x / scale) + (float)zero_point);
}

float ee_dequantise_value(int8_t q, float scale, int zero_point)
{
    return scale * (float)((int)q - zero_point);
}

void ee_standardise(const float *raw, const float *mean, const float *std,
                    int n_samples, int n_channels, float *x_std)
{
    for (int t = 0; t < n_samples; t++) {
        const float *in = raw + (long)t * n_channels;
        float *out = x_std + (long)t * n_channels;
        for (int c = 0; c < n_channels; c++) {
            out[c] = (in[c] - mean[c]) / std[c];
        }
    }
}

void ee_quantise(const float *x, int n, float scale, int zero_point, int8_t *q)
{
    for (int i = 0; i < n; i++) {
        q[i] = ee_quantise_value(x[i], scale, zero_point);
    }
}

int ee_requantise(const int8_t *q_in, int n, float scale_in, int zp_in,
                  float scale_out, int zp_out, int8_t *q_out)
{
    if (scale_in == scale_out && zp_in == zp_out) {
        if (q_out != q_in && n > 0) {
            memmove(q_out, q_in, (size_t)n);
        }
        return 1;
    }

    const float ratio = scale_in / scale_out;
    const float zp_o = (float)zp_out;
    for (int i = 0; i < n; i++) {
        const float v = (float)((int)q_in[i] - zp_in) * ratio;
        q_out[i] = clamp_to_int8(roundf(v) + zp_o);
    }
    return 0;
}

void ee_stats_pool(const int8_t *q, int T, int P, float scale, int zero_point,
                   float *feat)
{
    const float zp = (float)zero_point;
    const float t_f = (float)T;

    for (int p = 0; p < P; p++) {
        /* Pass 1: integer sum and maximum. */
        int32_t sum = 0;
        int maxq = -128;
        for (int t = 0; t < T; t++) {
            const int v = q[(long)t * P + p];
            sum += v;
            if (v > maxq) {
                maxq = v;
            }
        }
        const float mq = (float)sum / t_f;

        /* Pass 2: float sum of squared deviations, increasing t. */
        float acc = 0.0f;
        for (int t = 0; t < T; t++) {
            const float d = (float)q[(long)t * P + p] - mq;
            acc += d * d;
        }
        const float var = acc / t_f;

        feat[p]         = scale * (mq - zp);
        feat[P + p]     = scale * ((float)maxq - zp);
        feat[2 * P + p] = scale * sqrtf(var);
    }
}

void ee_linear_head(const float *feat, int D, const float *W, const float *b,
                    int K, float *logits)
{
    /* Accumulator starts at b[j], then adds feat[d] * W[d][j] in increasing d
     * (the order of export_ee_firmware.py's host reference).  d-outer /
     * j-inner keeps the W reads contiguous without changing that order. */
    for (int j = 0; j < K; j++) {
        logits[j] = b[j];
    }
    for (int d = 0; d < D; d++) {
        const float f = feat[d];
        const float *row = W + (long)d * K;
        for (int j = 0; j < K; j++) {
            logits[j] += f * row[j];
        }
    }
}

float ee_softmax_entropy(const float *logits, int K, float *probs)
{
    float m = logits[0];
    for (int j = 1; j < K; j++) {
        if (logits[j] > m) {
            m = logits[j];
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < K; j++) {
        const float e = expf(logits[j] - m);
        probs[j] = e;
        sum += e;
    }

    float acc = 0.0f;
    for (int j = 0; j < K; j++) {
        const float p = probs[j] / sum;
        probs[j] = p;
        acc += p * logf(p + EE_ENTROPY_EPS);
    }
    return -acc;
}

int ee_argmax_f32(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

int ee_argmax_i8(const int8_t *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

/* ======================================================================
 * int16 stage tensors (EE_ACT_BITS == 16, TFLite 16x8).  Same formulas and
 * loop orders as the int8 functions above; only the type and the clamping
 * range differ.
 * ====================================================================== */

/* Clamp an integral float to the int16 range; NaN goes to -32768. */
static int16_t clamp_to_int16(float r)
{
    if (!(r > -32768.0f)) {
        return (int16_t)-32768;
    }
    if (r > 32767.0f) {
        return (int16_t)32767;
    }
    return (int16_t)r;
}

int16_t ee_quantise_value_i16(float x, float scale, int zero_point)
{
    /* roundf(x / scale) is integral; adding zero_point in float is exact
     * whenever the result can land in [-32768, 32767]. */
    return clamp_to_int16(roundf(x / scale) + (float)zero_point);
}

float ee_dequantise_value_i16(int16_t q, float scale, int zero_point)
{
    return scale * (float)((int)q - zero_point);
}

void ee_quantise_i16(const float *x, int n, float scale, int zero_point, int16_t *q)
{
    for (int i = 0; i < n; i++) {
        q[i] = ee_quantise_value_i16(x[i], scale, zero_point);
    }
}

int ee_requantise_i16(const int16_t *q_in, int n, float scale_in, int zp_in,
                      float scale_out, int zp_out, int16_t *q_out)
{
    if (scale_in == scale_out && zp_in == zp_out) {
        if (q_out != q_in && n > 0) {
            memmove(q_out, q_in, (size_t)n * sizeof(int16_t));
        }
        return 1;
    }

    const float ratio = scale_in / scale_out;
    const float zp_o = (float)zp_out;
    for (int i = 0; i < n; i++) {
        const float v = (float)((int)q_in[i] - zp_in) * ratio;
        q_out[i] = clamp_to_int16(roundf(v) + zp_o);
    }
    return 0;
}

void ee_stats_pool_i16(const int16_t *q, int T, int P, float scale, int zero_point,
                       float *feat)
{
    const float zp = (float)zero_point;
    const float t_f = (float)T;

    for (int p = 0; p < P; p++) {
        /* Pass 1: integer sum (int32: |sum| <= T * 32768) and maximum. */
        int32_t sum = 0;
        int maxq = -32768;
        for (int t = 0; t < T; t++) {
            const int v = q[(long)t * P + p];
            sum += v;
            if (v > maxq) {
                maxq = v;
            }
        }
        const float mq = (float)sum / t_f;

        /* Pass 2: float sum of squared deviations, increasing t. */
        float acc = 0.0f;
        for (int t = 0; t < T; t++) {
            const float d = (float)q[(long)t * P + p] - mq;
            acc += d * d;
        }
        const float var = acc / t_f;

        feat[p]         = scale * (mq - zp);
        feat[P + p]     = scale * ((float)maxq - zp);
        feat[2 * P + p] = scale * sqrtf(var);
    }
}

int ee_argmax_i16(const int16_t *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

/* ======================================================================
 * C tail: bidirectional GRU and attention pooling in float32
 * (EE_INTERFACE.md, "C tail").
 * ====================================================================== */

static float sigmoid_f(float v)
{
    return 1.0f / (1.0f + expf(-v));
}

/* One GRU step, h <- GRU(x, h), Keras reset_after=True.  Both projections
 * use the previous h (hp is computed before h is updated); h[j] is updated
 * in place from its own old value only. */
static void gru_step(const float *x, int D, const ee_gru_weights_t *w, int H,
                     float *h, float *xp, float *hp)
{
    const int G = 3 * H;
    ee_linear_head(x, D, w->W, w->bi, G, xp); /* xp[j] = bi[j] + sum_d x[d] W[d][j] */
    ee_linear_head(h, H, w->U, w->br, G, hp); /* hp[j] = br[j] + sum_u h[u] U[u][j] */
    for (int j = 0; j < H; j++) {
        const float z = sigmoid_f(xp[j] + hp[j]);
        const float r = sigmoid_f(xp[H + j] + hp[H + j]);
        const float n = tanhf(xp[2 * H + j] + r * hp[2 * H + j]);
        h[j] = z * h[j] + (1.0f - z) * n;
    }
}

/* One GRU direction over q (exactly one of q8 / q16 is non-NULL).  reverse:
 * 0 = t from 0 to T-1, 1 = t from T-1 to 0.  The state after the step at time
 * t goes to out[t * stride + j] (j < H), so the output is in time order. */
static void gru_direction(const int8_t *q8, const int16_t *q16, int T, int D,
                          float scale, int zero_point, int H,
                          const ee_gru_weights_t *w, int reverse, float *out,
                          int stride, float *scratch)
{
    float *x = scratch;       /* [D]  dequantised input of the current step */
    float *h = x + D;         /* [H]  state                                  */
    float *xp = h + H;        /* [3H] input projection                       */
    float *hp = xp + 3 * H;   /* [3H] recurrent projection                   */

    for (int j = 0; j < H; j++) {
        h[j] = 0.0f;
    }
    for (int i = 0; i < T; i++) {
        const int t = reverse ? (T - 1 - i) : i;
        const long base = (long)t * D;
        if (q8 != NULL) {
            for (int d = 0; d < D; d++) {
                x[d] = scale * (float)((int)q8[base + d] - zero_point);
            }
        } else {
            for (int d = 0; d < D; d++) {
                x[d] = scale * (float)((int)q16[base + d] - zero_point);
            }
        }
        gru_step(x, D, w, H, h, xp, hp);
        float *dst = out + (long)t * stride;
        for (int j = 0; j < H; j++) {
            dst[j] = h[j];
        }
    }
}

void ee_bigru_i8(const int8_t *q, int T, int D, float scale, int zero_point, int H,
                 const ee_gru_weights_t *fwd, const ee_gru_weights_t *bwd,
                 float *hseq, float *scratch)
{
    gru_direction(q, NULL, T, D, scale, zero_point, H, fwd, 0, hseq, 2 * H, scratch);
    gru_direction(q, NULL, T, D, scale, zero_point, H, bwd, 1, hseq + H, 2 * H, scratch);
}

void ee_bigru_i16(const int16_t *q, int T, int D, float scale, int zero_point, int H,
                  const ee_gru_weights_t *fwd, const ee_gru_weights_t *bwd,
                  float *hseq, float *scratch)
{
    gru_direction(NULL, q, T, D, scale, zero_point, H, fwd, 0, hseq, 2 * H, scratch);
    gru_direction(NULL, q, T, D, scale, zero_point, H, bwd, 1, hseq + H, 2 * H, scratch);
}

void ee_attention_pool(const float *hseq, int T, int C, const float *w, float b,
                       float *scores, float *feat)
{
    /* score_t = b + sum_c hseq[t][c] * w[c], increasing c. */
    for (int t = 0; t < T; t++) {
        const float *row = hseq + (long)t * C;
        float s = b;
        for (int c = 0; c < C; c++) {
            s += row[c] * w[c];
        }
        scores[t] = s;
    }

    float m = scores[0];
    for (int t = 1; t < T; t++) {
        if (scores[t] > m) {
            m = scores[t];
        }
    }

    /* e_t and their sum, increasing t. */
    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        const float e = expf(scores[t] - m);
        scores[t] = e;
        sum += e;
    }

    /* feat[c] = sum_t (e_t / sum) * hseq[t][c]: t outer, c inner keeps each
     * feat[c] accumulating in increasing t with contiguous hseq reads. */
    for (int c = 0; c < C; c++) {
        feat[c] = 0.0f;
    }
    for (int t = 0; t < T; t++) {
        const float a = scores[t] / sum;
        const float *row = hseq + (long)t * C;
        for (int c = 0; c < C; c++) {
            feat[c] += a * row[c];
        }
    }
}
