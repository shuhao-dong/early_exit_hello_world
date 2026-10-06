/*
 * ee_select.h -- backbone selection and build options for the early-exit firmware
 * ===============================================================================
 *
 * Included by main.c and ee_pipeline.cpp before anything generated.  It picks
 * ONE backbone and names the three headers that export_ee_firmware.py writes
 * into this directory for it:
 *
 *   EE_CONFIG_HEADER   ee_<bb>_config.h    window / channel / class constants,
 *                                          scaler (C, included by main.c)
 *   EE_MODEL_HEADER    ee_<bb>_model.h     the four stage .tflite arrays, arena
 *                                          sizes, exit heads, thresholds
 *                                          (C++, included ONLY by ee_pipeline.cpp)
 *   EE_TESTVEC_HEADER  ee_<bb>_testvec.h   self-test vectors (C, included by
 *                                          main.c only when EE_SELF_TEST == 1)
 *
 * Selecting the backbone
 * ----------------------
 * Exactly one of the following may be defined; with none, EE_BACKBONE_TCN is
 * used.
 *
 *   EE_BACKBONE_TCN        tcn_small          -> ee_tcn_small_*.h
 *                          (int8 stages)
 *   EE_BACKBONE_TINIERHAR  tinierhar          -> ee_tinierhar_*.h
 *                          (int16 activations, TFLite 16x8; stage 4 ends at
 *                          the GRU input and the BiGRU + attention + dense
 *                          layers run as the float32 C tail in ee_math.c,
 *                          EE_HAS_C_TAIL in the model header; README_EE.md)
 *   EE_BACKBONE_DCL        deepconvlstm_0_25  -> ee_deepconvlstm_0_25_*.h
 *                          (int8 stages; stage 4 holds the fused LSTM state)
 *
 * Three equivalent ways to choose one:
 *   1. CMake (CMakeLists.txt, after find_package(Zephyr)):
 *          target_compile_definitions(app PRIVATE EE_BACKBONE_DCL)
 *   2. west command line (read by CMakeLists.txt with zephyr_get(), so it also
 *      works under sysbuild, the NCS default; always build pristine):
 *          west build -b nrf54l15dk/nrf54l15/cpuapp -p -- -DEE_BACKBONE=DCL
 *   3. Edit this file: uncomment one line in the block below.
 * Defining more than one (for example one here and another from CMake) is a
 * compile error.
 *
 * The other options below follow the same pattern: each has a default here
 * (#ifndef) and can be overridden from CMake / west (-DEE_FORCE_EXIT=2 ...)
 * or by editing the default.
 */
#ifndef EE_SELECT_H
#define EE_SELECT_H

/* ---- Option 3: edit here (leave all commented to use the default, TCN) ---- */
/* #define EE_BACKBONE_TCN */
/* #define EE_BACKBONE_TINIERHAR */
/* #define EE_BACKBONE_DCL */

#if (defined(EE_BACKBONE_TCN) + defined(EE_BACKBONE_TINIERHAR) + defined(EE_BACKBONE_DCL)) > 1
#error "ee_select.h: define at most one of EE_BACKBONE_TCN, EE_BACKBONE_TINIERHAR, EE_BACKBONE_DCL"
#endif

#if !defined(EE_BACKBONE_TCN) && !defined(EE_BACKBONE_TINIERHAR) && !defined(EE_BACKBONE_DCL)
#define EE_BACKBONE_TCN 1
#endif

#if defined(EE_BACKBONE_TCN)
#define EE_BACKBONE_KEY    "tcn_small"
#define EE_CONFIG_HEADER   "ee_tcn_small_config.h"
#define EE_MODEL_HEADER    "ee_tcn_small_model.h"
#define EE_TESTVEC_HEADER  "ee_tcn_small_testvec.h"
#elif defined(EE_BACKBONE_TINIERHAR)
#define EE_BACKBONE_KEY    "tinierhar"
#define EE_CONFIG_HEADER   "ee_tinierhar_config.h"
#define EE_MODEL_HEADER    "ee_tinierhar_model.h"
#define EE_TESTVEC_HEADER  "ee_tinierhar_testvec.h"
#elif defined(EE_BACKBONE_DCL)
#define EE_BACKBONE_KEY    "deepconvlstm_0_25"
#define EE_CONFIG_HEADER   "ee_deepconvlstm_0_25_config.h"
#define EE_MODEL_HEADER    "ee_deepconvlstm_0_25_model.h"
#define EE_TESTVEC_HEADER  "ee_deepconvlstm_0_25_testvec.h"
#endif

/* Friendly message when the exporter has not been run for this backbone. */
#if defined(__has_include)
#if !__has_include(EE_CONFIG_HEADER) || !__has_include(EE_MODEL_HEADER)
#error "Generated headers for the selected backbone are missing in firmware/src: run export_ee_firmware.py --backbone <bb> (see README_EE.md)"
#endif
#endif

/* ---------------------------------------------------------------------------
 * EE_FORCE_EXIT
 *   -1      thresholds decide (the deployed policy).
 *   0,1,2   always leave at early exit 1, 2 or 3.  The enabled heads before it
 *           are still evaluated (their cost is paid, they are just not
 *           allowed to stop the cascade); the forced exit's head is
 *           evaluated even when that exit is disabled in the policy.
 *   3       always run the full backbone (final exit); every enabled head is
 *           evaluated on the way.
 *   Use 0..3 to measure the latency / energy of each exit path on the PPK2.
 * ------------------------------------------------------------------------- */
#ifndef EE_FORCE_EXIT
#define EE_FORCE_EXIT -1
#endif
#if (EE_FORCE_EXIT < -1) || (EE_FORCE_EXIT > 3)
#error "EE_FORCE_EXIT must be -1 (thresholds decide) or 0..3"
#endif

/* ---------------------------------------------------------------------------
 * EE_SELF_TEST
 *   0  off.
 *   1  before the measurement loop, replay EE_TV_RAW (ee_<bb>_testvec.h)
 *      through the same standardisation + ee_run (thresholds decide) and
 *      compare exit, class and entropies with the host integer reference
 *      (int8 or int16 stages, plus the C tail where the backbone has one).
 *      Adds the test vectors to flash.
 * ------------------------------------------------------------------------- */
#ifndef EE_SELF_TEST
#define EE_SELF_TEST 0
#endif
#if (EE_SELF_TEST != 0) && (EE_SELF_TEST != 1)
#error "EE_SELF_TEST must be 0 or 1"
#endif
#if EE_SELF_TEST && defined(__has_include)
#if !__has_include(EE_TESTVEC_HEADER)
#error "EE_SELF_TEST=1 but the generated test-vector header for the selected backbone is missing in firmware/src: re-run export_ee_firmware.py (see README_EE.md)"
#endif
#endif

/* EE_INFERENCE_IDLE_MS: idle time between measured windows, so the PPK2 trace
 * shows a clear baseline between active regions (too short and adjacent
 * windows merge in the current trace). */
#ifndef EE_INFERENCE_IDLE_MS
#define EE_INFERENCE_IDLE_MS 2500
#endif

/* EE_WARMUP_ITERATIONS: windows run (and discarded, not logged) before the
 * measurement loop, to prime TFLM and caches. */
#ifndef EE_WARMUP_ITERATIONS
#define EE_WARMUP_ITERATIONS 5
#endif

/* EE_USE_WINDOW_MARKER: 1 = toggle the window-gpios pin once per measured
 * window (PPK2 segmentation); 0 = two-pin setup (prep + infer only). */
#ifndef EE_USE_WINDOW_MARKER
#define EE_USE_WINDOW_MARKER 1
#endif

/* EE_LOG_BREAKDOWN: 1 = after each window's summary line, log a second line
 * with the per-part microseconds (input quantisation, each stage that ran,
 * the C tail when it ran, each head evaluated, each requantisation) and the
 * evaluated entropies; 0 = summary line only. */
#ifndef EE_LOG_BREAKDOWN
#define EE_LOG_BREAKDOWN 1
#endif

/* ---------------------------------------------------------------------------
 * EE_PART_TIMER_DWT -- clock behind the per-part timings of ee_run()
 *   (ee_result_t.cycles_quant / cycles_stage / cycles_head / cycles_requant /
 *   cycles_tail)
 *   1  the Cortex-M33 DWT cycle counter (DWT->CYCCNT, CPU clock: 128 MHz on
 *      the nRF54L15, 1 tick = 7.8 ns).  ee_init() enables it (DCB DEMCR.TRCENA,
 *      CYCCNT = 0, DWT CTRL.CYCCNTENA) and keeps it only when the core has a
 *      cycle counter (CTRL.NOCYCCNT == 0) and the counter actually advances;
 *      otherwise it logs a warning and the parts fall back to k_cycle_get_32().
 *      Zephyr's own DWT support is not used: soc/nordic/nrf54l does not select
 *      CPU_CORTEX_M_HAS_DWT, so this goes through CMSIS (<cmsis_core.h>)
 *      directly.  Meant for the secure nrf54l15dk/nrf54l15/cpuapp target (the
 *      one this firmware is built for); on the /ns target the counter may be
 *      blocked, and the run-time check then selects the fallback.
 *      CYCCNT wraps after 2^32 cycles (33.5 s at 128 MHz); the unsigned
 *      differences used here are correct across one wrap.
 *   0  k_cycle_get_32(): the 1 MHz GRTC (1 us resolution, too coarse for the
 *      heads / requantisations, which take a few us).  Use 0 for a target
 *      without <cmsis_core.h> / DWT.
 *   The headline prep / infer timings of main.c always use k_cycle_get_32()
 *   inside the PPK2 GPIO markers, whatever this option says.  ee_part_timer_hz()
 *   and ee_part_timer_name() (ee_pipeline.h) report which clock is in use.
 * ------------------------------------------------------------------------- */
#ifndef EE_PART_TIMER_DWT
#define EE_PART_TIMER_DWT 1
#endif
#if (EE_PART_TIMER_DWT != 0) && (EE_PART_TIMER_DWT != 1)
#error "EE_PART_TIMER_DWT must be 0 (k_cycle_get_32) or 1 (DWT cycle counter)"
#endif

#endif /* EE_SELECT_H */
