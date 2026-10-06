# Early-exit firmware: interface between the exporter and the firmware

This file is the contract between `export_ee_firmware.py` (repository root, Python, generates headers) and the firmware
under `firmware/src/` (Zephyr / nRF Connect SDK, TensorFlow Lite Micro, nRF54L15 DK). Both sides implement exactly what
is written here.

## Pipeline being deployed

TORUS-9 (fs 100 Hz, window 250 samples x 7 channels: acc xyz, gyro xyz, detrended pressure; 9 classes), one LOSO fold
(default subject 540). Backbone = the rotation-trained `plain_rot` model of that fold; three early exits after
`EXIT_TAPS[backbone]` plus the final exit (the backbone's own softmax). Each early exit = stats pooling over time of
the tap tensor ([mean; max; std] per feature) followed by one linear softmax layer, fitted on four rotated copies of
the training windows (the paper's "rotation-fitted heads"). Exit thresholds = the paper's per-exit entropy
thresholds at lambda = 0.4, searched on rotated validation windows (A4). An exit k is *enabled* when its threshold
exists; a disabled exit's head is never evaluated (its cost is not paid).

Backbones (`--backbone`): `tcn_small` (taps tcn_b1_add, tcn_b2_add, tcn_b3_add), `tinierhar` (th_b1_add, th_b3_add,
th_b5_add), `deepconvlstm_0_25` (dcl_relu0_1, dcl_relu1_0, dcl_relu1_1).

## Partition into stage models

The backbone is cut at the three taps into four TFLite models with int8 weights and **integer input and output** of
`EE_ACT_BITS` bits: int8 (TFLite full-integer INT8; TCN, DeepConvLSTM) or int16 (TFLite 16x8: int16 activations, int8
weights; TinierHAR, whose first tap is too wide for int8: its exit-1 head then agrees with the float head on 84.6 % of
held-out windows instead of 99.8 %). Everything below that says "int8" for a stage tensor means "EE_ACT_BITS-bit
integer"; the clamping range is [-128, 127] or [-32768, 32767].

| stage s | input | output |
|---|---|---|
| 0 | standardised window (1, 250, 7) | tap 0 |
| 1 | tap 0 | tap 1 |
| 2 | tap 1 | tap 2 |
| 3 | tap 2 | class probabilities (1, K) (the backbone's softmax), or the C tail's input (1, T, D) when `EE_HAS_C_TAIL` |

Tensor memory layout is row-major (NHWC); a tap of shape (1, T, ...) is read as `T` time steps of `P` contiguous
values (`P` = product of the remaining dims). Between stages the int8 output of stage s is requantised to the int8
input of stage s+1. Because each stage is calibrated separately, the two scales can differ in the last digits while
the requantisation maps every int8 code to itself; the exporter then writes the previous stage's output scale into
the next stage's input tensor (flatbuffer patch), so the firmware takes the plain-copy path.

A stage may hold **variable tensors** (the fused INT8 LSTM of DeepConvLSTM keeps its hidden and cell state in them,
and TFLite/TFLM carry them from one invocation to the next). The trained model starts every window from zero state,
so such a stage (`EE_STAGE_STATEFUL[s] == 1`) is reset before every invoke: `MicroInterpreter::Reset()` on the device
(zeroes the variable tensors to their zero point; leaves the input untouched), `reset_all_variables()` on the host.

## Exact arithmetic (Python host reference and C must agree)

All in float32 unless stated.
- `round_half_away(v)` = C `roundf(v)`: nearest integer, halves away from zero (exact; note that
  `sign(v) * floor(|v| + 0.5)` evaluated in float32 differs at v = 0.49999997f).
- Input quantisation: `q = clamp(round_half_away(x / s) + zp, -128, 127)`.
- Requantisation: if `s_in == s_out && zp_in == zp_out` then copy; else `q_out = clamp(round_half_away((q_in - zp_in) * (s_in / s_out)) + zp_out, -128, 127)`.
- Stats pooling of an int8 tap `q[t][p]` (t < T, p < P) with scale `s`, zero point `zp`, into `feat[3P]`:
  `sum_p = sum_t q` (int32); `mq = (float)sum_p / T`; `maxq = max_t q`; `var = sum_t ((float)q - mq)^2 / T` (float
  accumulation, two-pass); `feat[p] = s * (mq - zp)`, `feat[P + p] = s * ((float)maxq - zp)`, `feat[2P + p] = s * sqrtf(var)`.
- Head: `logits[j] = b[j] + sum_d feat[d] * W[d * K + j]` (W row-major [D][K], D = 3P).
- Softmax: `m = max logits; e_j = expf(logits_j - m); p_j = e_j / sum e`.
- Entropy (nats): `H = -sum_j p_j * logf(p_j + 1e-12f)`. All sums are sequential in index order (the host mirrors
  this); `expf` / `logf` may differ from the host by about 1 ulp, so entropies agree to ~1e-7, not bit for bit.
- Exit k (k = 0, 1, 2) is taken when it is enabled and `H_k < threshold[k]`; prediction = argmax p (first max).
- Final exit (no C tail): prediction = argmax of the int8 output of stage 3 (first max); confidence = `s * (q - zp)`.
  With `EE_HAS_C_TAIL` the final exit is the C tail's (next section).
- Standardisation (firmware preprocessing): `x_std[t][c] = (raw[t][c] - mean[c]) / std[c]`.

## C tail (TinierHAR: `EE_HAS_C_TAIL == 1`)

TFLM (NCS 3.0.2) has no GRU kernel. An unrolled GRU does not fit the nRF54L15 (~1 MB flash, ~215 KB arena), and a
rolled one converts to a WHILE body with dynamic-shape SLICE / CONCATENATION, which TFLM aborts on. So stage 3's TFLite
model ends at the GRU input (`tinierhar_flatten`, shape (1, T, D)) and the firmware runs the rest in float32:

- Input: `x[t][d] = s * (float)(q[t][d] - zp)` from stage 3's integer output (dequantised step by step).
- Bidirectional GRU, H units per direction (Keras GRU, `reset_after=True`, gate blocks z | r | n in that order, zero
  initial state). Forward over t = 0 .. T-1, backward over t = T-1 .. 0; at each step, for j < 3H:
  `xp[j] = BI[j] + sum_d x[d] * W[d * 3H + j]`, `hp[j] = BR[j] + sum_u h[u] * U[u * 3H + j]` (h of the previous step);
  then for j < H: `z = sig(xp[j] + hp[j])`, `r = sig(xp[H + j] + hp[H + j])`,
  `n = tanhf(xp[2H + j] + r * hp[2H + j])`, `h[j] = z * h[j] + (1 - z) * n`, with `sig(v) = 1 / (1 + expf(-v))`.
  The step's h is stored at `hseq[t][j]` (forward) or `hseq[t][H + j]` (backward), so hseq is in time order.
- Additive attention pooling: `score_t = ATT_B[0] + sum_c hseq[t][c] * ATT_W[c]` (c < 2H); `m = max_t score_t`;
  `e_t = expf(score_t - m)`; `feat[c] = sum_t (e_t / sum e) * hseq[t][c]`.
- Dense softmax: `logits[k] = OUT_B[k] + sum_c feat[c] * OUT_W[c * K + k]`; softmax; prediction = argmax (first max);
  confidence = its probability.
- All sums sequential in index order. The host reference uses NumPy matrix products, so host and device agree to about
  1e-6 (not bit for bit); test vectors keep a class margin of 0.05 at the final exit.

## Generated headers (written by the exporter into `firmware/src/`)

`<bb>` is the backbone key (`tcn_small`, `tinierhar`, `deepconvlstm_0_25`). All arrays are `static const`. Generated
headers are never edited by hand.

### `ee_<bb>_config.h` (C; included by `main.c`)
```c
#define EE_BACKBONE_NAME   "tcn_small"
#define EE_DATASET_NAME    "torus9"
#define EE_SUBJECT         540
#define EE_SAMPLE_RATE_HZ  100
#define EE_WINDOW_SAMPLES  250
#define EE_NUM_CHANNELS    7
#define EE_NUM_CLASSES     9
#define EE_NUM_EXITS       4                 /* 3 early exits + final */
static const float EE_SCALER_MEAN[EE_NUM_CHANNELS];
static const float EE_SCALER_STD[EE_NUM_CHANNELS];
static const char *const EE_CLASS_NAMES[EE_NUM_CLASSES];
static const char *const EE_CHANNEL_NAMES[EE_NUM_CHANNELS];
```

### `ee_<bb>_model.h` (C++; included only by `ee_pipeline.cpp`)
```c
#define EE_LAMBDA_STR        "0.4"
#define EE_THRESHOLD_SET     "rot"           /* validation windows the thresholds were searched on */
#define EE_NUM_STAGES        4
#define EE_NUM_EARLY_EXITS   3
#define EE_ACT_BITS          8 | 16          /* stage I/O and activation precision; weights int8 */
static const unsigned char ee_stage0_tflite[] __attribute__((aligned(16))) = { 0x.., ... };   /* likewise 1, 2, 3 */
#define EE_STAGE0_TFLITE_LEN  <bytes>                                                       /* likewise 1, 2, 3 */
#define EE_STAGE0_ARENA_BYTES <bytes, multiple of 16>   /* host estimate from tensor lifetimes; likewise 1, 2, 3 */
static const int   EE_STAGE_IN_ELEMS[4],  EE_STAGE_OUT_ELEMS[4];
static const float EE_STAGE_IN_SCALE[4],  EE_STAGE_OUT_SCALE[4];   /* informative: the firmware reads the tensors' own */
static const int   EE_STAGE_IN_ZP[4],     EE_STAGE_OUT_ZP[4];      /* params and only warns on a mismatch            */
static const unsigned char EE_STAGE_STATEFUL[4];   /* 1: reset (MicroInterpreter::Reset) before every invoke */
#define EE_TAP0_T <int>   #define EE_TAP0_P <int>                                           /* likewise 1, 2 */
static const int   EE_TAP_T[3], EE_TAP_P[3];
static const float EE_HEAD0_W[3 * EE_TAP0_P * K], EE_HEAD0_B[K];                             /* likewise 1, 2 */
static const float *const EE_HEAD_W[3], *const EE_HEAD_B[3];
static const unsigned char EE_EXIT_ENABLED[3];     /* 0 = never evaluated */
static const float EE_THRESHOLD[3];                 /* entropy (nats); unused when disabled */
static const float EE_EXIT_FLOPS[4];                /* full-cost FLOPs of leaving at exit k under this policy
                                                       (a disabled exit's entry excludes its own head) */
#define EE_FULL_FLOPS <float>                       /* the backbone alone */
#define EE_HAS_C_TAIL 0 | 1
/* when 1: */
#define EE_TAIL_T <T>   #define EE_TAIL_D <D>   #define EE_TAIL_H <H>
static const float EE_GRU_FWD_W[D * 3H], EE_GRU_FWD_U[H * 3H], EE_GRU_FWD_BI[3H], EE_GRU_FWD_BR[3H];  /* likewise BWD */
static const float EE_ATT_W[2H], EE_ATT_B[1], EE_OUT_W[2H * K], EE_OUT_B[K];
#define EE_NUM_OPS <n>
#define EE_REGISTER_OPS(r) ((r).AddConv2D() == kTfLiteOk && (r).AddAdd() == kTfLiteOk && ...)
```

### `ee_<bb>_testvec.h` (C; included by `main.c` only when `EE_SELF_TEST` is 1)
```c
#define EE_TV_COUNT <n>
static const float EE_TV_RAW[EE_TV_COUNT][EE_WINDOW_SAMPLES * EE_NUM_CHANNELS];  /* raw windows, [t][c] */
static const int   EE_TV_LABEL[EE_TV_COUNT];          /* true class */
static const int   EE_TV_EXIT[EE_TV_COUNT];           /* expected exit 0..3 (host INT8 pipeline, this arithmetic;
                                                         windows chosen with entropy and class margins) */
static const int   EE_TV_CLASS[EE_TV_COUNT];          /* expected predicted class */
static const float EE_TV_ENTROPY[EE_TV_COUNT][3];     /* expected entropy of each evaluated exit, -1 if not evaluated */
```

## Firmware (hand-written, `firmware/src/`)

- `ee_select.h`: chooses the backbone with exactly one of `EE_BACKBONE_TCN` (default), `EE_BACKBONE_TINIERHAR`,
  `EE_BACKBONE_DCL` (can also come from the compiler command line) and defines `EE_CONFIG_HEADER`, `EE_MODEL_HEADER`,
  `EE_TESTVEC_HEADER`; options `EE_FORCE_EXIT` (-1 = thresholds decide; 0..3 = always leave at that exit, evaluating the
  enabled heads before it, to measure each exit path) and `EE_SELF_TEST` (0/1).
- `ee_math.h/.c`: pure C99, no Zephyr / TFLM includes: the arithmetic above (quantise, requantise, pool, head, softmax +
  entropy, argmax, standardise; int8 functions and `_i16` variants for `EE_ACT_BITS == 16`) and the C tail
  (`ee_bigru_i8` / `ee_bigru_i16`, which dequantise stage 3's output one time step at a time, and
  `ee_attention_pool`; the dense layer is `ee_linear_head`). No dynamic allocation: the caller passes every buffer.
  Must compile on the host with `gcc -std=c99 -Wall -Wextra -Werror`.
- `ee_pipeline.h/.cpp`: TFLM glue. `int ee_init(void)` builds one `MicroInterpreter` per stage (own arena), registers
  `EE_REGISTER_OPS`, checks the `EE_ACT_BITS`-bit I/O types and element counts (stage 3's output: `EE_NUM_CLASSES`, or
  `EE_TAIL_T * EE_TAIL_D` with last dimension `EE_TAIL_D` when `EE_HAS_C_TAIL`), logs arena use per stage;
  `int ee_run(const float *x_std, int force_exit, ee_result_t *r)` runs the cascade and fills timings (cycles) per part:
  input quantisation, each stage invoke, each evaluated head, each requantisation, and the C tail (`cycles_tail`, timed
  separately from stage 3's invoke). Forcing the final exit runs the tail.
- `ee_tflm_compat.h/.cpp`: TFLM 48613f7 (NCS 3.0.2) implements TRANSPOSE for float32 and int8 only (Prepare accepts
  int16, Invoke fails), and TinierHAR's stage 3 is an int16 TRANSPOSE + RESHAPE. With `EE_ACT_BITS == 16` the
  interpreters get a resolver that forwards to the one `EE_REGISTER_OPS` filled, except that TRANSPOSE's Eval gains the
  int16 case (TFLM's `reference_ops::Transpose<int16_t>`, pure data movement). int8 builds do not use it.
- `main.c`: keeps the existing benchmark behaviour: seeded LCG synthetic input (per channel uniformly in
  `mean +- 3 std`), preprocessing = standardisation (PPK2 `prep` GPIO + cycle counter), inference = `ee_run`
  (PPK2 `infer` GPIO + cycle counter), window-marker toggle, warm-up, idle between windows, RAM power-down; logs at
  start-up the activation bits and the C tail (shape, or none), and per window: preprocessing us, inference us, exit
  taken, class (confidence), and the per-part breakdown (with `tail=` after the stages when the C tail ran). With
  `EE_SELF_TEST` it first replays the test vectors and reports exit / class matches and the largest entropy difference.
