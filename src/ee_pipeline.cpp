/*
 * ee_pipeline.cpp -- TFLM glue for the early-exit cascade
 * ======================================================
 *
 * One MicroInterpreter per stage model, each with its own statically
 * allocated arena (EE_STAGE<s>_ARENA_BYTES from the generated model header),
 * sharing one op resolver.  The exit heads, pooling, softmax / entropy, the
 * integer (re)quantisation and, when the header sets EE_HAS_C_TAIL, the
 * float32 C tail after stage 3 run in ee_math.c with the arithmetic fixed by
 * EE_INTERFACE.md.  Quantisation parameters are taken from the tensors
 * themselves; the header copies are informative and only produce a warning
 * when they disagree.
 *
 * Stage tensors are EE_ACT_BITS-bit integers: int8 (TFLite full-integer INT8)
 * or int16 (TFLite 16x8: int16 activations, int8 weights).  act_t and the
 * act_*() overloads below pick the matching tensor type, data pointer and
 * ee_math.c functions at compile time; with EE_ACT_BITS == 8 the calls are
 * exactly the int8 ones of the int8-only version of this file.
 *
 * The generated model header (four .tflite arrays) is included in this
 * translation unit only, so the models are in flash exactly once.
 *
 * No function-local statics: Zephyr provides no thread-safe-static guards
 * (__cxa_guard_*), and the libsupc++ ones drag exception-handling code into
 * the image (it would inflate the measured flash footprint).  The resolver is
 * a file-scope object (constructed before main(), as in the previous
 * single-model wrapper); the four interpreters are constructed with
 * placement new into file-scope storage inside ee_init().
 *
 * Per-part timings use the "part timer" (EE_PART_TIMER_DWT, ee_select.h): the
 * Cortex-M33 DWT cycle counter, enabled here through CMSIS because Zephyr's
 * DWT support is not available on the nRF54L15 (soc/nordic/nrf54l does not
 * select CPU_CORTEX_M_HAS_DWT), or k_cycle_get_32() when the DWT is off or
 * not usable.
 */
#include "ee_pipeline.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <new>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "ee_select.h"
#include "ee_math.h"
#include EE_CONFIG_HEADER
#include EE_MODEL_HEADER
#if !defined(EE_ACT_BITS) || !defined(EE_HAS_C_TAIL)
#error "generated model header lacks EE_ACT_BITS / EE_HAS_C_TAIL (written by export_ee_firmware.py since the int16 / C-tail extension): re-run the exporter"
#endif
#if (EE_ACT_BITS != 8) && (EE_ACT_BITS != 16)
#error "generated header: EE_ACT_BITS must be 8 (int8) or 16 (int16, TFLite 16x8)"
#endif
#if (EE_HAS_C_TAIL != 0) && (EE_HAS_C_TAIL != 1)
#error "generated header: EE_HAS_C_TAIL must be 0 or 1"
#endif
#if EE_ACT_BITS == 16
#include "ee_tflm_compat.h"
#endif

#if EE_PART_TIMER_DWT
/* CMSIS core (core_cm33.h through the nRF54L15 MDK): DWT, DCB, SystemCoreClock.
 * Zephyr's modules/cmsis puts this header on the include path of every
 * Cortex-M build. */
#include <cmsis_core.h>
#endif

LOG_MODULE_REGISTER(ee_pipeline, LOG_LEVEL_INF);

/* ---- Compile-time consistency between this file and the generated header ---- */
static_assert(EE_NUM_STAGES == EE_PIPE_NUM_STAGES, "generated header: EE_NUM_STAGES must be 4");
static_assert(EE_NUM_EARLY_EXITS == EE_PIPE_NUM_EARLY_EXITS, "generated header: EE_NUM_EARLY_EXITS must be 3");
static_assert(EE_NUM_EXITS == EE_PIPE_NUM_EARLY_EXITS + 1, "generated header: EE_NUM_EXITS must be 4");
static_assert(EE_NUM_CLASSES > 0, "generated header: EE_NUM_CLASSES must be positive");
static_assert(EE_TAP0_T > 0 && EE_TAP1_T > 0 && EE_TAP2_T > 0, "generated header: tap lengths must be positive");
static_assert(EE_TAP0_P > 0 && EE_TAP1_P > 0 && EE_TAP2_P > 0, "generated header: tap widths must be positive");
static_assert(EE_STAGE0_ARENA_BYTES % 16 == 0 && EE_STAGE1_ARENA_BYTES % 16 == 0 &&
              EE_STAGE2_ARENA_BYTES % 16 == 0 && EE_STAGE3_ARENA_BYTES % 16 == 0,
              "generated header: arena sizes must be multiples of 16");

#if EE_HAS_C_TAIL
static_assert(EE_TAIL_T > 0 && EE_TAIL_D > 0 && EE_TAIL_H > 0, "generated header: C tail T, D, H must be positive");
static_assert(sizeof(EE_GRU_FWD_W) == sizeof(float) * EE_TAIL_D * 3 * EE_TAIL_H &&
              sizeof(EE_GRU_BWD_W) == sizeof(float) * EE_TAIL_D * 3 * EE_TAIL_H,
              "generated header: EE_GRU_{FWD,BWD}_W must be [D][3H]");
static_assert(sizeof(EE_GRU_FWD_U) == sizeof(float) * EE_TAIL_H * 3 * EE_TAIL_H &&
              sizeof(EE_GRU_BWD_U) == sizeof(float) * EE_TAIL_H * 3 * EE_TAIL_H,
              "generated header: EE_GRU_{FWD,BWD}_U must be [H][3H]");
static_assert(sizeof(EE_GRU_FWD_BI) == sizeof(float) * 3 * EE_TAIL_H &&
              sizeof(EE_GRU_FWD_BR) == sizeof(float) * 3 * EE_TAIL_H &&
              sizeof(EE_GRU_BWD_BI) == sizeof(float) * 3 * EE_TAIL_H &&
              sizeof(EE_GRU_BWD_BR) == sizeof(float) * 3 * EE_TAIL_H,
              "generated header: GRU biases must be [3H]");
static_assert(sizeof(EE_ATT_W) == sizeof(float) * 2 * EE_TAIL_H && sizeof(EE_ATT_B) == sizeof(float),
              "generated header: EE_ATT_W must be [2H], EE_ATT_B [1]");
static_assert(sizeof(EE_OUT_W) == sizeof(float) * 2 * EE_TAIL_H * EE_NUM_CLASSES &&
              sizeof(EE_OUT_B) == sizeof(float) * EE_NUM_CLASSES,
              "generated header: EE_OUT_W must be [2H][K], EE_OUT_B [K]");
#endif

namespace {

constexpr int kNumStages = EE_PIPE_NUM_STAGES;
constexpr int kNumEarly = EE_PIPE_NUM_EARLY_EXITS;
constexpr int kNumClasses = EE_NUM_CLASSES;

constexpr int max3(int a, int b, int c)
{
    return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}
constexpr int kMaxTapP = max3(EE_TAP0_P, EE_TAP1_P, EE_TAP2_P);
constexpr int kMaxFeat = 3 * kMaxTapP;

/* Header-vs-tensor scale comparison (warning only): relative, so that a scale
 * printed with fewer digits than float32 needs does not trigger it. */
constexpr float kScaleRelTol = 1e-6f;

/* ---- Stage tensor type (EE_ACT_BITS) ---- */
#if EE_ACT_BITS == 8
using act_t = int8_t;
constexpr TfLiteType kActType = kTfLiteInt8;
#else
using act_t = int16_t;
constexpr TfLiteType kActType = kTfLiteInt16;
#endif
constexpr int kActBits = EE_ACT_BITS;

act_t *act_data(TfLiteTensor *t)
{
#if EE_ACT_BITS == 8
    return t->data.int8;
#else
    return t->data.i16;
#endif
}

/* ee_math.c entry points per tensor type (overloads; only the act_t ones are
 * used, and unused inline functions are not emitted). */
inline void act_quantise(const float *x, int n, float s, int zp, int8_t *q) { ee_quantise(x, n, s, zp, q); }
inline void act_quantise(const float *x, int n, float s, int zp, int16_t *q) { ee_quantise_i16(x, n, s, zp, q); }
inline int act_requantise(const int8_t *qi, int n, float si, int zi, float so, int zo, int8_t *qo)
{
    return ee_requantise(qi, n, si, zi, so, zo, qo);
}
inline int act_requantise(const int16_t *qi, int n, float si, int zi, float so, int zo, int16_t *qo)
{
    return ee_requantise_i16(qi, n, si, zi, so, zo, qo);
}
inline void act_stats_pool(const int8_t *q, int T, int P, float s, int zp, float *f) { ee_stats_pool(q, T, P, s, zp, f); }
inline void act_stats_pool(const int16_t *q, int T, int P, float s, int zp, float *f) { ee_stats_pool_i16(q, T, P, s, zp, f); }
inline int act_argmax(const int8_t *v, int n) { return ee_argmax_i8(v, n); }
inline int act_argmax(const int16_t *v, int n) { return ee_argmax_i16(v, n); }
inline float act_dequantise(int8_t q, float s, int zp) { return ee_dequantise_value(q, s, zp); }
inline float act_dequantise(int16_t q, float s, int zp) { return ee_dequantise_value_i16(q, s, zp); }

/* ---- Per-stage arenas and model data ---- */
alignas(16) uint8_t g_arena0[EE_STAGE0_ARENA_BYTES];
alignas(16) uint8_t g_arena1[EE_STAGE1_ARENA_BYTES];
alignas(16) uint8_t g_arena2[EE_STAGE2_ARENA_BYTES];
alignas(16) uint8_t g_arena3[EE_STAGE3_ARENA_BYTES];

uint8_t *const g_arena[kNumStages] = {g_arena0, g_arena1, g_arena2, g_arena3};
constexpr unsigned int g_arena_bytes[kNumStages] = {
    EE_STAGE0_ARENA_BYTES, EE_STAGE1_ARENA_BYTES, EE_STAGE2_ARENA_BYTES, EE_STAGE3_ARENA_BYTES};
const unsigned char *const g_tflite[kNumStages] = {
    ee_stage0_tflite, ee_stage1_tflite, ee_stage2_tflite, ee_stage3_tflite};
constexpr unsigned int g_tflite_len[kNumStages] = {
    EE_STAGE0_TFLITE_LEN, EE_STAGE1_TFLITE_LEN, EE_STAGE2_TFLITE_LEN, EE_STAGE3_TFLITE_LEN};

/* ---- Runtime state ---- */
struct StageRuntime {
    tflite::MicroInterpreter *interp;
    act_t *in;
    act_t *out;
    float in_scale;
    float out_scale;
    int in_zp;
    int out_zp;
    int in_elems;
    int out_elems;
    unsigned int arena_used;
};

StageRuntime g_stage[kNumStages];
struct k_mutex g_mutex;
bool g_ops_registered = false;
bool g_interp_built = false;
bool g_initialized = false;

/* One op resolver shared by the four interpreters (TFLM keeps a const
 * reference to it; it must outlive them).  EE_REGISTER_OPS runs once. */
tflite::MicroMutableOpResolver<EE_NUM_OPS> g_resolver;

/* Storage for the four interpreters, constructed in ee_init(). */
alignas(tflite::MicroInterpreter) uint8_t g_interp_mem[kNumStages][sizeof(tflite::MicroInterpreter)];

/* Exit-head scratch, sized for the widest tap (the C tail's dense softmax
 * reuses g_logits / g_probs). */
float g_feat[kMaxFeat];
float g_logits[kNumClasses];
float g_probs[kNumClasses];

#if EE_HAS_C_TAIL
/* ---- C tail: BiGRU(H) + attention pooling + dense softmax in float32 ----
 * Static buffers sized from the header: the GRU output sequence [T][2H], the
 * per-step GRU scratch (one dequantised input step, state, projections), the
 * attention weights over time and the pooled feature.  The tail reads stage
 * 3's integer output in place (no float copy of the T x D input). */
constexpr int kTailT = EE_TAIL_T;
constexpr int kTailD = EE_TAIL_D;
constexpr int kTailH = EE_TAIL_H;
float g_tail_hseq[kTailT * 2 * kTailH];
float g_tail_scratch[EE_BIGRU_SCRATCH_FLOATS(kTailD, kTailH)];
float g_tail_scores[kTailT];
float g_tail_feat[2 * kTailH];
constexpr unsigned int kTailRamBytes = (unsigned int)(sizeof(g_tail_hseq) + sizeof(g_tail_scratch) +
                                                      sizeof(g_tail_scores) + sizeof(g_tail_feat));
constexpr unsigned int kTailParams = (unsigned int)((sizeof(EE_GRU_FWD_W) + sizeof(EE_GRU_FWD_U) +
                                                     sizeof(EE_GRU_FWD_BI) + sizeof(EE_GRU_FWD_BR) +
                                                     sizeof(EE_GRU_BWD_W) + sizeof(EE_GRU_BWD_U) +
                                                     sizeof(EE_GRU_BWD_BI) + sizeof(EE_GRU_BWD_BR) +
                                                     sizeof(EE_ATT_W) + sizeof(EE_ATT_B) + sizeof(EE_OUT_W) +
                                                     sizeof(EE_OUT_B)) / sizeof(float));
const ee_gru_weights_t kGruFwd = {EE_GRU_FWD_W, EE_GRU_FWD_U, EE_GRU_FWD_BI, EE_GRU_FWD_BR};
const ee_gru_weights_t kGruBwd = {EE_GRU_BWD_W, EE_GRU_BWD_U, EE_GRU_BWD_BI, EE_GRU_BWD_BR};

inline void act_bigru(const int8_t *q, float s, int zp)
{
    ee_bigru_i8(q, kTailT, kTailD, s, zp, kTailH, &kGruFwd, &kGruBwd, g_tail_hseq, g_tail_scratch);
}
inline void act_bigru(const int16_t *q, float s, int zp)
{
    ee_bigru_i16(q, kTailT, kTailD, s, zp, kTailH, &kGruFwd, &kGruBwd, g_tail_hseq, g_tail_scratch);
}

/* Final exit through the C tail: stage-3 output q [T][D] -> class
 * probabilities in g_probs.  The argmax is left to the caller (as for the
 * heads, it is not part of the timed part). */
void run_c_tail(const act_t *q, float scale, int zero_point)
{
    act_bigru(q, scale, zero_point);
    ee_attention_pool(g_tail_hseq, kTailT, 2 * kTailH, EE_ATT_W, EE_ATT_B[0], g_tail_scores, g_tail_feat);
    ee_linear_head(g_tail_feat, 2 * kTailH, EE_OUT_W, EE_OUT_B, kNumClasses, g_logits);
    (void)ee_softmax_entropy(g_logits, kNumClasses, g_probs);
}
#endif

int tensor_elems(const TfLiteTensor *t)
{
    if (t->dims == nullptr) {
        return -1;
    }
    int n = 1;
    for (int i = 0; i < t->dims->size; i++) {
        n *= t->dims->data[i];
    }
    return n;
}

bool scale_differs(float tensor_scale, float header_scale)
{
    return fabsf(tensor_scale - header_scale) > kScaleRelTol * fabsf(header_scale);
}

/* Checks one EE_ACT_BITS-bit I/O tensor and copies out its data pointer and
 * own quantisation parameters. */
int setup_tensor(int s, const char *which, TfLiteTensor *t, int expect_elems,
                 float header_scale, int header_zp,
                 act_t **data, float *scale, int *zp, int *elems)
{
    if (t == nullptr) {
        LOG_ERR("stage %d: no %s tensor", s, which);
        return -EINVAL;
    }
    if (t->type != kActType) {
        LOG_ERR("stage %d: %s tensor type %d, expected int%d (%d; EE_ACT_BITS of the header)", s,
                which, (int)t->type, kActBits, (int)kActType);
        return -EINVAL;
    }
    const int n = tensor_elems(t);
    if (n != expect_elems || t->bytes != (size_t)n * sizeof(act_t)) {
        LOG_ERR("stage %d: %s tensor has %d elements (%u bytes), header expects %d x int%d", s,
                which, n, (unsigned int)t->bytes, expect_elems, kActBits);
        return -EINVAL;
    }
    const float sc = t->params.scale;
    const int z = (int)t->params.zero_point;
    if (!(sc > 0.0f)) {
        LOG_ERR("stage %d: %s tensor scale %e is not positive (expected per-tensor int%d)", s,
                which, (double)sc, kActBits);
        return -EINVAL;
    }
    if (scale_differs(sc, header_scale) || z != header_zp) {
        LOG_WRN("stage %d: %s quantisation (scale %.9e, zp %d) differs from the header "
                "(scale %.9e, zp %d); using the tensor's",
                s, which, (double)sc, z, (double)header_scale, header_zp);
    }
    *data = act_data(t);
    *scale = sc;
    *zp = z;
    *elems = n;
    return 0;
}

/* ---- Part timer ---- */
const char *const kPartTimerKernelName = "k_cycle_get_32";
const char *const kPartTimerDwtName = "DWT CYCCNT";

bool g_part_dwt = false;     /* true: DWT->CYCCNT, false: k_cycle_get_32() */
uint32_t g_part_hz = 0u;     /* ticks per second of the DWT counter (when used) */

#if EE_PART_TIMER_DWT
/* Enables the DWT cycle counter: trace enable (DEMCR.TRCENA), counter cleared
 * and started.  Returns true only when the core implements CYCCNT
 * (CTRL.NOCYCCNT == 0), the counter advances (it may be blocked, e.g. without
 * trace permission in a non-secure image) and the core clock is known. */
bool dwt_cycle_counter_start(void)
{
#if defined(DCB) && defined(DCB_DEMCR_TRCENA_Msk)
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
#else /* CMSIS headers before DCB was introduced */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#endif
    if ((DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) != 0u) {
        return false;
    }
    DWT->CYCCNT = 0u;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    const uint32_t c0 = DWT->CYCCNT;
    bool advanced = false;
    for (int i = 0; i < 64 && !advanced; i++) {
        advanced = (DWT->CYCCNT != c0);
    }
    return advanced && SystemCoreClock > 0u;
}
#endif

/* Selects the part timer; called once by ee_init(). */
void part_timer_init(void)
{
    g_part_dwt = false;
    g_part_hz = 0u;
#if EE_PART_TIMER_DWT
    if (dwt_cycle_counter_start()) {
        g_part_dwt = true;
        g_part_hz = (uint32_t)SystemCoreClock;
        LOG_INF("part timer: %s @ %u Hz", kPartTimerDwtName, (unsigned int)g_part_hz);
    } else {
        LOG_WRN("part timer: EE_PART_TIMER_DWT=1 but the DWT cycle counter is not usable "
                "(not implemented, not counting, or SystemCoreClock unknown); per-part timings "
                "fall back to %s @ %u Hz",
                kPartTimerKernelName, (unsigned int)sys_clock_hw_cycles_per_sec());
    }
#else
    LOG_INF("part timer: %s @ %u Hz (EE_PART_TIMER_DWT=0)", kPartTimerKernelName,
            (unsigned int)sys_clock_hw_cycles_per_sec());
#endif
}

/* Current part-timer value; differences are taken in uint32_t (wrap-safe).
 * Always inlined (Zephyr builds with -Os), so a DWT timing point costs a flag
 * test and one load, not a call. */
__attribute__((always_inline)) inline uint32_t part_now(void)
{
#if EE_PART_TIMER_DWT
    if (g_part_dwt) {
        return DWT->CYCCNT;
    }
#endif
    return k_cycle_get_32();
}

void reset_result(ee_result_t *r)
{
    memset(r, 0, sizeof(*r));
    r->exit_taken = -1;
    r->predicted_class = -1;
    r->confidence = 0.0f;
    for (int k = 0; k < kNumEarly; k++) {
        r->entropy[k] = -1.0f;
    }
}

} /* namespace */

extern "C" int ee_init(void)
{
    if (g_initialized) {
        LOG_WRN("ee_init called more than once, skipping");
        return 0;
    }

    k_mutex_init(&g_mutex);
    part_timer_init();

    /* Ops are registered once (TFLM refuses a second registration of the
     * same op, e.g. if ee_init is retried). */
    if (!g_ops_registered) {
        if (!(EE_REGISTER_OPS(g_resolver))) {
            LOG_ERR("op registration failed (EE_NUM_OPS = %d too small, or op not in TFLM)",
                    (int)EE_NUM_OPS);
            return -EINVAL;
        }
        g_ops_registered = true;
    }

    const tflite::Model *model[kNumStages];
    for (int s = 0; s < kNumStages; s++) {
        model[s] = tflite::GetModel(g_tflite[s]);
        if (model[s]->version() != TFLITE_SCHEMA_VERSION) {
            LOG_ERR("stage %d: model schema version %u, runtime %d", s,
                    (unsigned int)model[s]->version(), (int)TFLITE_SCHEMA_VERSION);
            return -ENOTSUP;
        }
    }

    /* Construct the interpreters once (placement new, no heap), each on its
     * own static arena; they are never destroyed.  With int16 stages the
     * interpreters see the generated resolver through ee_tflm_compat's
     * wrapper, which adds the int16 case that TFLM 48613f7's TRANSPOSE lacks
     * (TinierHAR's stage 3); int8 builds use the generated resolver as is. */
    if (!g_interp_built) {
#if EE_ACT_BITS == 16
        const tflite::MicroOpResolver &resolver = ee_int16_transpose_resolver(g_resolver);
#else
        const tflite::MicroOpResolver &resolver = g_resolver;
#endif
        for (int s = 0; s < kNumStages; s++) {
            g_stage[s].interp = new (g_interp_mem[s])
                tflite::MicroInterpreter(model[s], resolver, g_arena[s], g_arena_bytes[s]);
        }
        g_interp_built = true;
    }

    unsigned int total_used = 0;
    unsigned int total_alloc = 0;
    for (int s = 0; s < kNumStages; s++) {
        StageRuntime &st = g_stage[s];

        if (st.interp->AllocateTensors() != kTfLiteOk) {
            LOG_ERR("stage %d: AllocateTensors failed; increase EE_STAGE%d_ARENA_BYTES (now %u)",
                    s, s, g_arena_bytes[s]);
            return -ENOMEM;
        }
        if (st.interp->inputs_size() != 1 || st.interp->outputs_size() != 1) {
            LOG_ERR("stage %d: expected 1 input and 1 output, got %u and %u", s,
                    (unsigned int)st.interp->inputs_size(),
                    (unsigned int)st.interp->outputs_size());
            return -EINVAL;
        }

        int rc = setup_tensor(s, "input", st.interp->input(0), EE_STAGE_IN_ELEMS[s],
                              EE_STAGE_IN_SCALE[s], EE_STAGE_IN_ZP[s],
                              &st.in, &st.in_scale, &st.in_zp, &st.in_elems);
        if (rc != 0) {
            return rc;
        }
        rc = setup_tensor(s, "output", st.interp->output(0), EE_STAGE_OUT_ELEMS[s],
                          EE_STAGE_OUT_SCALE[s], EE_STAGE_OUT_ZP[s],
                          &st.out, &st.out_scale, &st.out_zp, &st.out_elems);
        if (rc != 0) {
            return rc;
        }

        st.arena_used = (unsigned int)st.interp->arena_used_bytes();
        total_used += st.arena_used;
        total_alloc += g_arena_bytes[s];

        LOG_INF("stage %d: model %u B, arena %u / %u B used", s, g_tflite_len[s], st.arena_used,
                g_arena_bytes[s]);
        LOG_INF("stage %d: in %d x int%d (s=%.6e zp=%d), out %d x int%d (s=%.6e zp=%d)", s,
                st.in_elems, kActBits, (double)st.in_scale, st.in_zp, st.out_elems, kActBits,
                (double)st.out_scale, st.out_zp);
        if (EE_STAGE_STATEFUL[s] != 0) {
            LOG_INF("stage %d: holds variable tensors (LSTM state); reset before every invoke", s);
        }
    }

    /* Shapes along the cascade. */
    if (g_stage[0].in_elems != EE_WINDOW_SAMPLES * EE_NUM_CHANNELS) {
        LOG_ERR("stage 0 input has %d elements, window is %d x %d", g_stage[0].in_elems,
                EE_WINDOW_SAMPLES, EE_NUM_CHANNELS);
        return -EINVAL;
    }
    for (int k = 0; k < kNumEarly; k++) {
        if (EE_TAP_T[k] <= 0 || EE_TAP_P[k] <= 0 || EE_TAP_P[k] > kMaxTapP ||
            EE_TAP_T[k] * EE_TAP_P[k] != g_stage[k].out_elems) {
            LOG_ERR("tap %d: T=%d P=%d does not match stage %d output (%d elements, max P %d)",
                    k, EE_TAP_T[k], EE_TAP_P[k], k, g_stage[k].out_elems, kMaxTapP);
            return -EINVAL;
        }
        if (g_stage[k + 1].in_elems != g_stage[k].out_elems) {
            LOG_ERR("stage %d output (%d) and stage %d input (%d) differ in size", k,
                    g_stage[k].out_elems, k + 1, g_stage[k + 1].in_elems);
            return -EINVAL;
        }
        const bool copy = (g_stage[k].out_scale == g_stage[k + 1].in_scale) &&
                          (g_stage[k].out_zp == g_stage[k + 1].in_zp);
        LOG_INF("stage %d -> %d: %s", k, k + 1, copy ? "same quantisation (copy)" : "requantise");
    }
#if EE_HAS_C_TAIL
    {
        /* Stage 3 ends at the GRU input (1, T, D), row-major [T][D]. */
        const TfLiteTensor *o = g_stage[kNumStages - 1].interp->output(0);
        const int last = (o->dims != nullptr && o->dims->size > 0) ? o->dims->data[o->dims->size - 1] : -1;
        if (g_stage[kNumStages - 1].out_elems != kTailT * kTailD || last != kTailD) {
            LOG_ERR("stage %d output has %d elements (last dim %d), the C tail expects T x D = %d x %d",
                    kNumStages - 1, g_stage[kNumStages - 1].out_elems, last, kTailT, kTailD);
            return -EINVAL;
        }
        LOG_INF("C tail: stage %d output %d x %d (int%d) -> BiGRU(%d per direction) + attention + "
                "dense(%d), float32; %u parameters, %u B buffers",
                kNumStages - 1, kTailT, kTailD, kActBits, kTailH, kNumClasses, kTailParams, kTailRamBytes);
    }
#else
    if (g_stage[kNumStages - 1].out_elems != kNumClasses) {
        LOG_ERR("stage %d output has %d elements, expected %d classes", kNumStages - 1,
                g_stage[kNumStages - 1].out_elems, kNumClasses);
        return -EINVAL;
    }
#endif

    LOG_INF("arenas: %u / %u B used over %d stages", total_used, total_alloc, kNumStages);

    g_initialized = true;
    return 0;
}

extern "C" int ee_run(const float *x_std, int force_exit, ee_result_t *r)
{
    if (r == nullptr) {
        return -EINVAL;
    }
    reset_result(r);
    if (!g_initialized) {
        LOG_ERR("ee_run called before a successful ee_init");
        return -ENODEV;
    }
    if (x_std == nullptr || force_exit < -1 || force_exit > EE_PIPE_EXIT_FINAL) {
        return -EINVAL;
    }

    k_mutex_lock(&g_mutex, K_FOREVER);

    int rc = 0;
    uint32_t t0 = part_now();
    act_quantise(x_std, g_stage[0].in_elems, g_stage[0].in_scale, g_stage[0].in_zp,
                 g_stage[0].in);
    r->cycles_quant = part_now() - t0;

    for (int s = 0; s < kNumStages; s++) {
        StageRuntime &st = g_stage[s];

        t0 = part_now();
        /* A stage with variable tensors (the fused LSTM's hidden / cell state, EE_STAGE_STATEFUL) would otherwise
         * carry its state from the previous window into this one; the trained model starts every window from zero
         * state.  MicroInterpreter::Reset() zeroes the variable tensors (to their zero point) and runs the ops' reset
         * hooks; it does not touch the input already written.  Counted as part of the stage. */
        TfLiteStatus status = kTfLiteOk;
        if (EE_STAGE_STATEFUL[s] != 0) {
            status = st.interp->Reset();
        }
        if (status == kTfLiteOk) {
            status = st.interp->Invoke();
        }
        r->cycles_stage[s] = part_now() - t0;
        r->stages_run = s + 1;
        if (status != kTfLiteOk) {
            LOG_ERR("stage %d: Reset/Invoke failed", s);
            rc = -EIO;
            break;
        }

        if (s == kNumStages - 1) {
#if EE_HAS_C_TAIL
            /* Final exit: the float32 C tail on stage 3's output (the GRU
             * input), timed on its own (not part of cycles_stage[3]). */
            t0 = part_now();
            run_c_tail(st.out, st.out_scale, st.out_zp);
            r->cycles_tail = part_now() - t0;
            const int cls = ee_argmax_f32(g_probs, kNumClasses);
            r->exit_taken = EE_PIPE_EXIT_FINAL;
            r->predicted_class = cls;
            r->confidence = g_probs[cls];
#else
            /* Final exit: the backbone's own (integer) softmax. */
            const int cls = act_argmax(st.out, kNumClasses);
            r->exit_taken = EE_PIPE_EXIT_FINAL;
            r->predicted_class = cls;
            r->confidence = act_dequantise(st.out[cls], st.out_scale, st.out_zp);
#endif
            break;
        }

        const bool enabled = (EE_EXIT_ENABLED[s] != 0);
        if (enabled || force_exit == s) {
            const int P = EE_TAP_P[s];
            t0 = part_now();
            act_stats_pool(st.out, EE_TAP_T[s], P, st.out_scale, st.out_zp, g_feat);
            ee_linear_head(g_feat, 3 * P, EE_HEAD_W[s], EE_HEAD_B[s], kNumClasses, g_logits);
            const float H = ee_softmax_entropy(g_logits, kNumClasses, g_probs);
            r->cycles_head[s] = part_now() - t0;
            r->entropy[s] = H;

            const bool take = (force_exit == s) ||
                              (force_exit < 0 && enabled && H < EE_THRESHOLD[s]);
            if (take) {
                const int cls = ee_argmax_f32(g_probs, kNumClasses);
                r->exit_taken = s;
                r->predicted_class = cls;
                r->confidence = g_probs[cls];
                break;
            }
        }

        StageRuntime &next = g_stage[s + 1];
        t0 = part_now();
        (void)act_requantise(st.out, st.out_elems, st.out_scale, st.out_zp, next.in_scale,
                             next.in_zp, next.in);
        r->cycles_requant[s] = part_now() - t0;
    }

    k_mutex_unlock(&g_mutex);
    return rc;
}

extern "C" uint32_t ee_part_timer_hz(void)
{
    return g_part_dwt ? g_part_hz : (uint32_t)sys_clock_hw_cycles_per_sec();
}

extern "C" const char *ee_part_timer_name(void)
{
    return g_part_dwt ? kPartTimerDwtName : kPartTimerKernelName;
}

extern "C" void ee_get_info(ee_info_t *info)
{
    if (info == nullptr) {
        return;
    }
    memset(info, 0, sizeof(*info));
    info->lambda_str = EE_LAMBDA_STR;
    info->threshold_set = EE_THRESHOLD_SET;
    for (int k = 0; k < kNumEarly; k++) {
        info->exit_enabled[k] = (EE_EXIT_ENABLED[k] != 0) ? 1 : 0;
        info->threshold[k] = EE_THRESHOLD[k];
        info->tap_t[k] = EE_TAP_T[k];
        info->tap_p[k] = EE_TAP_P[k];
    }
    for (int s = 0; s < kNumStages; s++) {
        info->exit_flops[s] = EE_EXIT_FLOPS[s];
        info->tflite_bytes[s] = g_tflite_len[s];
        info->arena_bytes[s] = g_arena_bytes[s];
        info->arena_used[s] = g_initialized ? g_stage[s].arena_used : 0u;
    }
    info->full_flops = (float)(EE_FULL_FLOPS);
    info->act_bits = kActBits;
#if EE_HAS_C_TAIL
    info->has_c_tail = 1;
    info->tail_t = kTailT;
    info->tail_d = kTailD;
    info->tail_h = kTailH;
    info->tail_params = kTailParams;
    info->tail_ram_bytes = kTailRamBytes;
#endif
}
