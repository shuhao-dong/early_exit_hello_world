/*
 * ee_pipeline.h -- early-exit cascade on TensorFlow Lite Micro
 * ===========================================================
 *
 * The backbone is split at three taps into four stage models with int8
 * weights and integer input / output of EE_ACT_BITS bits (int8, or int16 for
 * TFLite 16x8; EE_INTERFACE.md).  ee_run() quantises the standardised window
 * into stage 0, then for s = 0..3 invokes stage s; after stages 0..2 it
 * evaluates the exit head when that exit is enabled (stats pooling + linear
 * head + softmax/entropy) and stops when H < threshold; otherwise the stage
 * output is requantised into the next stage's input.  Stage 3 is the final
 * exit: the backbone's own softmax, or, when the generated header sets
 * EE_HAS_C_TAIL (TinierHAR), the input of the float32 C tail (bidirectional
 * GRU + attention pooling + dense softmax, ee_math.c) that ee_run() runs
 * after it.
 *
 * Exit numbering: exit_taken 0, 1, 2 = early exits 1, 2, 3 (after stage 1, 2,
 * 3 of 4); 3 = final exit.  Stage numbering in arrays is 0-based.
 *
 * Per-part timings (ee_result_t.cycles_*) are in ticks of the "part timer":
 * the DWT cycle counter (CPU clock) when EE_PART_TIMER_DWT = 1 (ee_select.h)
 * and ee_init() found it working, k_cycle_get_32() otherwise.  Convert them
 * with ee_part_timer_hz(), never with k_cyc_to_us_*().
 */
#ifndef EE_PIPELINE_H
#define EE_PIPELINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EE_PIPE_NUM_STAGES       4
#define EE_PIPE_NUM_EARLY_EXITS  3
#define EE_PIPE_EXIT_FINAL       3

typedef struct {
    int      exit_taken;        /* 0..2 early exit 1..3, 3 final; -1 on error           */
    int      predicted_class;   /* argmax (first max); -1 on error                      */
    float    confidence;        /* early exit: softmax p of the class; final: the C     */
                                /* tail's softmax p (EE_HAS_C_TAIL), else the           */
                                /* dequantised stage-3 output s * (q - zp)              */
    float    entropy[EE_PIPE_NUM_EARLY_EXITS];   /* nats; -1 when not evaluated         */
    /* Per-part durations in part-timer ticks (ee_part_timer_hz() per second). */
    uint32_t cycles_quant;                               /* x_std -> stage-0 input      */
    uint32_t cycles_stage[EE_PIPE_NUM_STAGES];           /* Invoke(); 0 if not run      */
    uint32_t cycles_head[EE_PIPE_NUM_EARLY_EXITS];       /* pool+head+softmax+entropy;  */
                                                         /* 0 if not evaluated          */
    uint32_t cycles_requant[EE_PIPE_NUM_EARLY_EXITS];    /* stage s -> s+1; 0 if not run*/
    uint32_t cycles_tail;       /* C tail after stage 3 (BiGRU + attention + dense +    */
                                /* softmax); 0 without EE_HAS_C_TAIL or if not run      */
    int      stages_run;        /* number of stages invoked (1..4)                      */
} ee_result_t;

/* Static description of the compiled-in policy (header constants) plus the
 * arena use measured by ee_init() (0 before a successful ee_init). */
typedef struct {
    const char  *lambda_str;                             /* EE_LAMBDA_STR               */
    const char  *threshold_set;                          /* EE_THRESHOLD_SET            */
    int          exit_enabled[EE_PIPE_NUM_EARLY_EXITS];
    float        threshold[EE_PIPE_NUM_EARLY_EXITS];     /* nats                        */
    float        exit_flops[EE_PIPE_NUM_STAGES];         /* EE_EXIT_FLOPS[k], k = 0..3  */
    float        full_flops;                             /* EE_FULL_FLOPS               */
    int          tap_t[EE_PIPE_NUM_EARLY_EXITS];
    int          tap_p[EE_PIPE_NUM_EARLY_EXITS];
    unsigned int tflite_bytes[EE_PIPE_NUM_STAGES];
    unsigned int arena_bytes[EE_PIPE_NUM_STAGES];        /* allocated                   */
    unsigned int arena_used[EE_PIPE_NUM_STAGES];         /* arena_used_bytes()          */
    int          act_bits;                               /* EE_ACT_BITS: 8 or 16        */
    int          has_c_tail;                             /* EE_HAS_C_TAIL: 0 or 1       */
    int          tail_t, tail_d, tail_h;                 /* EE_TAIL_T/D/H; 0 if no tail */
    unsigned int tail_params;                            /* float parameters of the tail*/
    unsigned int tail_ram_bytes;                         /* its static float buffers    */
} ee_info_t;

/* Selects the part timer (with EE_PART_TIMER_DWT = 1: enables the DWT cycle
 * counter and keeps it when present and counting, else warns and falls back
 * to k_cycle_get_32()), builds one MicroInterpreter per stage (own static
 * arena), registers EE_REGISTER_OPS once, checks the EE_ACT_BITS-bit I/O
 * types and element counts (stage 3's output: EE_NUM_CLASSES, or the C
 * tail's T x D input), reads each tensor's own scale / zero point (warns if
 * they differ from the header) and logs arena use.  Returns 0, or a negative
 * errno value.  Calling it again after success is a no-op. */
int ee_init(void);

/* Runs the cascade on one standardised window x_std[EE_WINDOW_SAMPLES *
 * EE_NUM_CHANNELS] ([t][c]).  force_exit: -1 = thresholds decide; 0..3 =
 * leave at that exit (see EE_FORCE_EXIT in ee_select.h).  Fills *r (per-part
 * timings in part-timer ticks).  Returns 0, or a negative errno value
 * (-ENODEV before ee_init, -EINVAL on bad arguments, -EIO on Invoke failure). */
int ee_run(const float *x_std, int force_exit, ee_result_t *r);

/* Fills *info (header constants, plus the arena use measured by ee_init);
 * valid before ee_init (arena_used then 0). */
void ee_get_info(ee_info_t *info);

/* The part timer chosen by ee_init(): its rate in ticks per second
 * (SystemCoreClock for the DWT cycle counter, 128000000 on the nRF54L15;
 * sys_clock_hw_cycles_per_sec() for k_cycle_get_32()) and a printable name
 * ("DWT CYCCNT" or "k_cycle_get_32").  Before ee_init() they describe
 * k_cycle_get_32(). */
uint32_t ee_part_timer_hz(void);
const char *ee_part_timer_name(void);

#ifdef __cplusplus
}
#endif

#endif /* EE_PIPELINE_H */
