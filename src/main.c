/*
 * TORUS early-exit HAR benchmark -- nRF54L15 DK
 * ============================================
 *
 * Deployment characterisation of the early-exit pipeline of the paper:
 * a rotation-trained backbone (TCN-Small, TinierHAR or DeepConvLSTM-0.25,
 * chosen in ee_select.h) trained on TORUS-9, cut at three taps into four
 * TFLite Micro stage models (int8 weights; int8 activations, or int16 for
 * TinierHAR), with a stats-pooling + linear "rotation-fitted" head after
 * each of the first three stages and per-exit entropy thresholds
 * (lambda = 0.4).  TinierHAR's BiGRU + attention + dense layers run after
 * stage 4 as a float32 C tail (ee_math.c).  All constants come from the
 * headers that export_ee_firmware.py generates (see README_EE.md and
 * EE_INTERFACE.md).
 *
 * Purpose: flash footprint, tensor-arena RAM, per-window latency of
 * preprocessing and of the early-exit inference, and per-window energy on
 * the PPK2.  Input is synthetic (seeded LCG, bounded, per channel uniform in
 * mean +- 3 std of the training scaler) to remove the sensor pipeline from
 * the measurement.  Synthetic windows are far from the training data, so the
 * thresholds will mostly let them run to the final exit; build with
 * EE_FORCE_EXIT = 0..3 to measure each exit path, and with EE_SELF_TEST = 1
 * to check exits / classes / entropies on real windows against the host.
 *
 * What is measured
 * ----------------
 *   prep   : per-channel standardisation of one raw window (float [t][c]),
 *            x_std = (raw - mean) / std.  Nothing else.
 *   infer  : ee_run(): input quantisation, the stage invokes that run, the
 *            exit heads that are evaluated, the requantisations between
 *            stages, the C tail after stage 4 (TinierHAR, final exit only)
 *            and the exit decision.  The per-part breakdown line reports
 *            each part; "other" is the remainder (timer reads, mutex,
 *            argmax, loop overhead).
 * What is NOT measured
 * --------------------
 *   - Window acquisition (the synthetic generator runs before the prep
 *     marker goes high).
 *   - Pressure detrending: channel 7 of TORUS-9 is *detrended* barometric
 *     pressure.  Detrending is done offline on the host before
 *     standardisation and is NOT part of the preprocessing measured here;
 *     the synthetic generator produces already-detrended values.  A real
 *     deployment needs an on-line detrending filter, whose cost is extra.
 *   - Logging (happens after both markers are low).
 *
 * GPIO ordering for PPK2: gpio_set -> k_cycle_get -> work -> k_cycle_get
 * -> gpio_clear.  The cycle-counter region is strictly enclosed by the
 * GPIO-high region, so the two measurements agree to within ~1-2 us
 * (two gpio_pin_set_dt calls).  A larger mismatch signals an interrupted
 * measurement window; flag and discard such iterations in post-processing.
 *
 * Two clocks (both named at start-up, after ee_init):
 *   - prep / infer (headline line): k_cycle_get_32() ticks (system timer, the
 *     1 MHz GRTC on the nRF54L15), converted with k_cyc_to_us_floor32(), whole
 *     microseconds, as in the previous benchmark.
 *   - the per-part breakdown line: ticks of the part timer chosen by ee_init()
 *     (ee_part_timer_hz(): the 128 MHz DWT cycle counter with
 *     EE_PART_TIMER_DWT = 1, else k_cycle_get_32()), printed in 0.1 us.
 */

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/printk.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/pm/device.h>
#include "ram_pwrdn.h"

#include "ee_select.h"
#include EE_CONFIG_HEADER
#include "ee_math.h"
#include "ee_pipeline.h"
#if EE_SELF_TEST
#include EE_TESTVEC_HEADER
#endif

LOG_MODULE_REGISTER(torus_ee, LOG_LEVEL_INF);

#define WINDOW_ELEMS (EE_WINDOW_SAMPLES * EE_NUM_CHANNELS)

/* -------------------- GPIO markers for PPK2 -------------------- */

static const struct gpio_dt_spec prep_gpio =
    GPIO_DT_SPEC_GET(DT_PATH(zephyr_user), prep_gpios);
static const struct gpio_dt_spec infer_gpio =
    GPIO_DT_SPEC_GET(DT_PATH(zephyr_user), infer_gpios);
#if EE_USE_WINDOW_MARKER
static const struct gpio_dt_spec window_gpio =
    GPIO_DT_SPEC_GET(DT_PATH(zephyr_user), window_gpios);
static bool window_marker_ok;
#endif

static int init_marker_gpios(void)
{
    if (!gpio_is_ready_dt(&prep_gpio) || !gpio_is_ready_dt(&infer_gpio)) {
        LOG_ERR("Marker GPIOs not ready");
        return -1;
    }
    gpio_pin_configure_dt(&prep_gpio, GPIO_OUTPUT_INACTIVE);
    gpio_pin_configure_dt(&infer_gpio, GPIO_OUTPUT_INACTIVE);
#if EE_USE_WINDOW_MARKER
    if (!gpio_is_ready_dt(&window_gpio)) {
        LOG_WRN("Window marker GPIO not ready (non-fatal)");
        window_marker_ok = false;
    } else {
        gpio_pin_configure_dt(&window_gpio, GPIO_OUTPUT_INACTIVE);
        window_marker_ok = true;
    }
#endif
    return 0;
}

/* -------------------- Synthetic input generator --------------------
 *
 * Seeded LCG -- reproducible across runs (same constants and seed as the
 * previous single-model benchmark).  Each channel c is uniform in
 * [mean[c] - 3 std[c], mean[c] + 3 std[c]] of the training scaler, so the
 * standardised input lies in [-3, 3].  The compute graph executes the same
 * operations regardless of input value; with early exits, however, WHICH
 * exit is taken depends on the input, hence EE_FORCE_EXIT for per-path
 * measurements.
 */

#define SYNTH_SEED 0x12345678u

static uint32_t rng_state = SYNTH_SEED;

static inline float next_random_float(float min_val, float max_val)
{
    rng_state = rng_state * 1103515245u + 12345u;
    float r = (float)((rng_state >> 16) & 0x7FFF) / 32767.0f;
    return min_val + r * (max_val - min_val);
}

static void fill_window_synthetic(float *raw)
{
    for (int t = 0; t < EE_WINDOW_SAMPLES; t++) {
        for (int c = 0; c < EE_NUM_CHANNELS; c++) {
            const float half = 3.0f * EE_SCALER_STD[c];
            raw[t * EE_NUM_CHANNELS + c] =
                next_random_float(EE_SCALER_MEAN[c] - half, EE_SCALER_MEAN[c] + half);
        }
    }
}

/* -------------------- Static buffers -------------------- */

/* Raw window, [t][c] (acc xyz, gyro xyz, detrended pressure for TORUS-9). */
static float raw_window[WINDOW_ELEMS];

/* Standardised window consumed by ee_run(). */
static float x_std[WINDOW_ELEMS];

/* -------------------- Logging helpers -------------------- */

static const char *exit_name(int k)
{
    static const char *const names[EE_PIPE_NUM_STAGES] = {"1", "2", "3", "final"};
    return (k >= 0 && k < EE_PIPE_NUM_STAGES) ? names[k] : "none";
}

static const char *class_name(int c)
{
    return (c >= 0 && c < EE_NUM_CLASSES) ? EE_CLASS_NAMES[c] : "?";
}

/* Bounded append into a log line (integers and strings only).  Uses
 * Zephyr's vsnprintk (cbprintf, the formatter the logger already links)
 * rather than newlib's vsnprintf, which would add its own float-capable
 * printf core to the measured flash footprint. */
static void __attribute__((format(printf, 4, 5)))
line_append(char *buf, size_t cap, size_t *len, const char *fmt, ...)
{
    if (*len + 1 >= cap) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintk(buf + *len, cap - *len, fmt, ap);
    va_end(ap);
    if (n > 0) {
        *len += (size_t)n;
        if (*len >= cap) {
            *len = cap - 1;
        }
    }
}

#if EE_LOG_BREAKDOWN
/* 1 when the selected backbone ends in the float32 C tail (set from
 * ee_get_info() in log_startup): the breakdown then reports "tail" after
 * stage 4 for windows that reach the final exit. */
static bool s_has_c_tail;

/* Entropy as "d.ddd" without float formatting. */
static void append_entropy(char *buf, size_t cap, size_t *len, const char *label, float h)
{
    const long mh = lroundf(h * 1000.0f);
    line_append(buf, cap, len, " %s=%ld.%03ld", label, mh / 1000, mh % 1000);
}

/* Timer ticks at hz ticks/s -> tenths of a microsecond, rounded to nearest
 * (64-bit intermediate: ticks * 10^7 < 2^56; saturates at UINT32_MAX, i.e.
 * after 429 s, far beyond one window). */
static uint32_t ticks_to_tenth_us(uint32_t ticks, uint32_t hz)
{
    if (hz == 0u) {
        return 0u;
    }
    const uint64_t t = ((uint64_t)ticks * 10000000u + hz / 2u) / hz;
    return (t > UINT32_MAX) ? UINT32_MAX : (uint32_t)t;
}

/* " <label>=<us>.<tenth>" from tenths of a microsecond (integer formatting,
 * vsnprintk-compatible). */
static void append_tenth_us(char *buf, size_t cap, size_t *len, const char *label,
                            uint32_t tenths)
{
    line_append(buf, cap, len, " %s=%u.%u", label, (unsigned int)(tenths / 10u),
                (unsigned int)(tenths % 10u));
}

/* infer_cycles: k_cycle_get_32() ticks of the whole ee_run() (the headline
 * infer value); the parts are in part-timer ticks (ee_part_timer_hz()). */
static void log_breakdown(const ee_result_t *r, uint32_t infer_cycles)
{
    const uint32_t part_hz = ee_part_timer_hz();
    char buf[320];
    char label[16];
    size_t len = 0;
    uint64_t parts = 0; /* sum of the printed parts, tenths of a us */
    uint32_t t;

    buf[0] = '\0';
    line_append(buf, sizeof(buf), &len, "  parts us:");
    t = ticks_to_tenth_us(r->cycles_quant, part_hz);
    append_tenth_us(buf, sizeof(buf), &len, "quant", t);
    parts += t;
    line_append(buf, sizeof(buf), &len, " |");
    for (int s = 0; s < r->stages_run && s < EE_PIPE_NUM_STAGES; s++) {
        snprintk(label, sizeof(label), "stage%d", s + 1);
        t = ticks_to_tenth_us(r->cycles_stage[s], part_hz);
        append_tenth_us(buf, sizeof(buf), &len, label, t);
        parts += t;
    }
    if (s_has_c_tail && r->exit_taken == EE_PIPE_EXIT_FINAL) {
        t = ticks_to_tenth_us(r->cycles_tail, part_hz);
        append_tenth_us(buf, sizeof(buf), &len, "tail", t);
        parts += t;
    }
    line_append(buf, sizeof(buf), &len, " |");
    for (int k = 0; k < EE_PIPE_NUM_EARLY_EXITS; k++) {
        if (r->entropy[k] >= 0.0f) {
            snprintk(label, sizeof(label), "head%d", k + 1);
            t = ticks_to_tenth_us(r->cycles_head[k], part_hz);
            append_tenth_us(buf, sizeof(buf), &len, label, t);
            parts += t;
        }
    }
    line_append(buf, sizeof(buf), &len, " |");
    for (int k = 0; k + 1 < r->stages_run && k < EE_PIPE_NUM_EARLY_EXITS; k++) {
        snprintk(label, sizeof(label), "requant%d-%d", k + 1, k + 2);
        t = ticks_to_tenth_us(r->cycles_requant[k], part_hz);
        append_tenth_us(buf, sizeof(buf), &len, label, t);
        parts += t;
    }
    /* other = infer - sum(parts), clamped at 0.  infer comes from
     * k_cycle_get_32() (1 us ticks on the GRTC) and the parts from the part
     * timer (the DWT cycle counter by default): with two clocks, and infer
     * quantised to whole ticks, "other" is approximate to about +-1 us. */
    const uint32_t infer_t = ticks_to_tenth_us(infer_cycles,
                                               (uint32_t)sys_clock_hw_cycles_per_sec());
    const uint32_t other = ((uint64_t)infer_t > parts) ? (uint32_t)(infer_t - parts) : 0u;
    line_append(buf, sizeof(buf), &len, " |");
    append_tenth_us(buf, sizeof(buf), &len, "other", other);
    line_append(buf, sizeof(buf), &len, " |");
    for (int k = 0; k < EE_PIPE_NUM_EARLY_EXITS; k++) {
        if (r->entropy[k] >= 0.0f) {
            snprintk(label, sizeof(label), "H%d", k + 1);
            append_entropy(buf, sizeof(buf), &len, label, r->entropy[k]);
        }
    }
    LOG_INF("%s", buf);
}
#endif

static void log_startup(void)
{
    ee_info_t info;
    char buf[256];
    size_t len = 0;

    ee_get_info(&info);

    LOG_INF("TORUS early-exit HAR benchmark (synthetic input)");
    LOG_INF("  Backbone      : %s (%s, LOSO subject %d)", EE_BACKBONE_NAME, EE_DATASET_NAME,
            (int)EE_SUBJECT);
    LOG_INF("  Window        : %d samples x %d channels @ %d Hz", EE_WINDOW_SAMPLES,
            EE_NUM_CHANNELS, EE_SAMPLE_RATE_HZ);
    LOG_INF("  Activations   : int%d stage I/O and activations, int8 weights%s", info.act_bits,
            (info.act_bits == 16) ? " (TFLite 16x8)" : "");
    if (info.has_c_tail) {
        LOG_INF("  C tail        : stage 4 output %d x %d -> BiGRU(%d per direction) + attention + "
                "dense, float32 (%u parameters, %u B buffers)",
                info.tail_t, info.tail_d, info.tail_h, info.tail_params, info.tail_ram_bytes);
    } else {
        LOG_INF("  C tail        : none (stage 4 ends at the backbone's softmax)");
    }
#if EE_LOG_BREAKDOWN
    s_has_c_tail = (info.has_c_tail != 0);
#endif

    buf[0] = '\0';
    for (int c = 0; c < EE_NUM_CHANNELS; c++) {
        line_append(buf, sizeof(buf), &len, "%s%s", c ? ", " : "", EE_CHANNEL_NAMES[c]);
    }
    LOG_INF("  Channels      : %s", buf);

    len = 0;
    buf[0] = '\0';
    for (int c = 0; c < EE_NUM_CLASSES; c++) {
        line_append(buf, sizeof(buf), &len, "%s%s", c ? ", " : "", EE_CLASS_NAMES[c]);
    }
    LOG_INF("  Classes (%d)   : %s", EE_NUM_CLASSES, buf);

    LOG_INF("  Policy        : lambda = %s, thresholds searched on '%s' validation windows",
            info.lambda_str, info.threshold_set);
    for (int k = 0; k < EE_PIPE_NUM_EARLY_EXITS; k++) {
        if (info.exit_enabled[k]) {
            LOG_INF("  Exit %d        : enabled, stop if H < %.4f nats | tap %d x %d | "
                    "%.4f MFLOP",
                    k + 1, (double)info.threshold[k], info.tap_t[k], info.tap_p[k],
                    (double)(info.exit_flops[k] * 1e-6f));
        } else {
            LOG_INF("  Exit %d        : disabled (head never evaluated) | tap %d x %d | "
                    "%.4f MFLOP",
                    k + 1, info.tap_t[k], info.tap_p[k], (double)(info.exit_flops[k] * 1e-6f));
        }
    }
    LOG_INF("  Exit final    : %.4f MFLOP (backbone alone %.4f MFLOP)",
            (double)(info.exit_flops[EE_PIPE_EXIT_FINAL] * 1e-6f),
            (double)(info.full_flops * 1e-6f));

#if EE_FORCE_EXIT < 0
    LOG_INF("  Force exit    : off (entropy thresholds decide)");
#else
    LOG_INF("  Force exit    : always leave at exit %s (enabled heads before it are evaluated)",
            exit_name(EE_FORCE_EXIT));
#if EE_FORCE_EXIT < EE_PIPE_EXIT_FINAL
    if (!info.exit_enabled[EE_FORCE_EXIT]) {
        LOG_WRN("  exit %s is disabled in this policy; its head is evaluated only because it "
                "is forced", exit_name(EE_FORCE_EXIT));
    }
#endif
#endif
    LOG_INF("  Self-test     : %s", EE_SELF_TEST ? "on" : "off");
    LOG_INF("  Idle / warm-up: %d ms between windows, %d warm-up windows",
            EE_INFERENCE_IDLE_MS, EE_WARMUP_ITERATIONS);
}

/* Logged after ee_init(), which chooses the part timer. */
static void log_timers(void)
{
    LOG_INF("  Timers        : prep/infer = k_cycle_get_32 @ %u Hz (us); "
            "parts = %s @ %u Hz (0.1 us)",
            (unsigned int)sys_clock_hw_cycles_per_sec(), ee_part_timer_name(),
            (unsigned int)ee_part_timer_hz());
}

/* -------------------- Self-test against host reference -------------------- */

#if EE_SELF_TEST
static void run_self_test(void)
{
    int exits_ok = 0;
    int classes_ok = 0;
    int labels_ok = 0;
    int eval_set_mismatch = 0;
    int run_errors = 0;
    float max_dh = 0.0f;

    LOG_INF("self-test: %d vectors (thresholds decide; EE_FORCE_EXIT not applied)",
            EE_TV_COUNT);

    for (int i = 0; i < EE_TV_COUNT; i++) {
        ee_result_t r;

        ee_standardise(EE_TV_RAW[i], EE_SCALER_MEAN, EE_SCALER_STD, EE_WINDOW_SAMPLES,
                       EE_NUM_CHANNELS, x_std);
        const int rc = ee_run(x_std, -1, &r);
        if (rc != 0) {
            LOG_ERR("tv %d: ee_run failed (%d)", i, rc);
            run_errors++;
            continue;
        }

        const bool exit_ok = (r.exit_taken == EE_TV_EXIT[i]);
        const bool class_ok = (r.predicted_class == EE_TV_CLASS[i]);
        bool set_ok = true;
        float dh = 0.0f;
        for (int k = 0; k < EE_PIPE_NUM_EARLY_EXITS; k++) {
            const bool exp_eval = (EE_TV_ENTROPY[i][k] >= 0.0f);
            const bool got_eval = (r.entropy[k] >= 0.0f);
            if (exp_eval != got_eval) {
                set_ok = false;
            } else if (exp_eval) {
                const float d = fabsf(r.entropy[k] - EE_TV_ENTROPY[i][k]);
                if (d > dh) {
                    dh = d;
                }
            }
        }
        if (dh > max_dh) {
            max_dh = dh;
        }
        exits_ok += exit_ok ? 1 : 0;
        classes_ok += class_ok ? 1 : 0;
        labels_ok += (r.predicted_class == EE_TV_LABEL[i]) ? 1 : 0;
        eval_set_mismatch += set_ok ? 0 : 1;

        LOG_INF("tv %2d: label %s | exit %s (host %s) | class %s (host %s) | "
                "max|dH| %.2e %s",
                i, class_name(EE_TV_LABEL[i]), exit_name(r.exit_taken),
                exit_name(EE_TV_EXIT[i]), class_name(r.predicted_class),
                class_name(EE_TV_CLASS[i]), (double)dh,
                (exit_ok && class_ok && set_ok) ? "OK" : "MISMATCH");
    }

    LOG_INF("self-test: exits %d/%d, classes %d/%d, max |dH| = %.3e", exits_ok, EE_TV_COUNT,
            classes_ok, EE_TV_COUNT, (double)max_dh);
    LOG_INF("self-test: true-label hits %d/%d", labels_ok, EE_TV_COUNT);
    if (eval_set_mismatch > 0) {
        LOG_WRN("self-test: %d vectors evaluated a different set of exits than the host",
                eval_set_mismatch);
    }
    if (run_errors > 0) {
        LOG_ERR("self-test: %d vectors failed to run", run_errors);
    }
}
#endif

/* -------------------- Inference pipeline -------------------- */

static void run_one_inference(bool log_result)
{
    ee_result_t res;

    /* --- Window acquisition (not measured) --- */
    fill_window_synthetic(raw_window);

    /* --- Preprocessing (GPIO-marked): standardisation only --- */
    gpio_pin_set_dt(&prep_gpio, 1);
    uint32_t prep_start = k_cycle_get_32();

    ee_standardise(raw_window, EE_SCALER_MEAN, EE_SCALER_STD, EE_WINDOW_SAMPLES,
                   EE_NUM_CHANNELS, x_std);

    uint32_t prep_end = k_cycle_get_32();
    gpio_pin_set_dt(&prep_gpio, 0);

    /* --- Inference (GPIO-marked): the early-exit cascade --- */
    gpio_pin_set_dt(&infer_gpio, 1);
    uint32_t inf_start = k_cycle_get_32();

    int rc = ee_run(x_std, EE_FORCE_EXIT, &res);

    uint32_t inf_end = k_cycle_get_32();
    gpio_pin_set_dt(&infer_gpio, 0);

    if (rc != 0) {
        LOG_ERR("ee_run failed (%d)", rc);
        return;
    }

    if (log_result) {
        LOG_INF("prep=%u us | infer=%u us | exit=%s | class=%s (%.1f%%)",
                (unsigned int)k_cyc_to_us_floor32(prep_end - prep_start),
                (unsigned int)k_cyc_to_us_floor32(inf_end - inf_start),
                exit_name(res.exit_taken), class_name(res.predicted_class),
                (double)(res.confidence * 100.0f));
#if EE_LOG_BREAKDOWN
        log_breakdown(&res, inf_end - inf_start);
#endif
    }
}

/* -------------------- Entry point -------------------- */

int main(void)
{
    log_startup();

    /* Power housekeeping */
    const struct device *sdhc_dev =
        DEVICE_DT_GET_OR_NULL(DT_NODELABEL(sdhc0));
    if (sdhc_dev && device_is_ready(sdhc_dev)) {
        pm_device_action_run(sdhc_dev, PM_DEVICE_ACTION_SUSPEND);
    }
    power_down_unused_ram();

    /* Marker GPIOs */
    if (init_marker_gpios() != 0) {
        return -1;
    }

    /* Early-exit runtime init (part timer, arena use per stage) */
    if (ee_init() != 0) {
        LOG_ERR("ee_init failed");
        return -1;
    }
    log_timers();

#if EE_SELF_TEST
    run_self_test();
#endif

    /* Warm-up: prime any lazy allocations, caches, etc. */
    LOG_INF("Warmup x%d", EE_WARMUP_ITERATIONS);
    for (int i = 0; i < EE_WARMUP_ITERATIONS; i++) {
        run_one_inference(false);
    }
    LOG_INF("Warmup complete; starting measurement loop");

    /* Main measurement loop */
    while (1) {
#if EE_USE_WINDOW_MARKER
        /* One edge per window on this pin -- use in post-processing to
         * count windows and align / segment the PPK2 trace.
         */
        if (window_marker_ok) {
            gpio_pin_toggle_dt(&window_gpio);
        }
#endif
        run_one_inference(true);

        /* Visible idle region so PPK2 can segment cleanly */
        k_msleep(EE_INFERENCE_IDLE_MS);
    }

    return 0;
}
