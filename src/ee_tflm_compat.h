/*
 * ee_tflm_compat.h -- int16 TRANSPOSE for the TFLM of NCS 3.0.2
 * =============================================================
 *
 * TFLM 48613f7 (NCS 3.0.2, optional/modules/lib/tflite-micro) implements
 * TRANSPOSE for float32 and int8 only: its Prepare accepts any type, so
 * AllocateTensors succeeds, but the first Invoke on an int16 tensor fails
 * with "Type INT16 is currently not supported by Transpose. Only float32 and
 * int8 is supported".  TinierHAR's stage 3 (TFLite 16x8) is TRANSPOSE
 * [0, 1, 3, 2] + RESHAPE (the Keras Permute + Reshape in front of the GRU),
 * so its final exit would always fail.  There is no CMSIS-NN TRANSPOSE, so
 * the board runs the same reference kernel as the host check.
 *
 * MicroMutableOpResolver::AddTranspose() takes no registration argument in
 * this TFLM version, so the generated EE_REGISTER_OPS() cannot be pointed at
 * another kernel.  Instead the interpreters get a resolver that forwards
 * every lookup to the generated one, except TRANSPOSE: its registration is
 * the generated resolver's own (same init / prepare / reset), with an Eval
 * that handles int16 with TFLM's reference_ops::Transpose<int16_t> (pure data
 * movement, the template the stock kernel uses for int8) and calls the stock
 * Eval for every other type.
 *
 * Used by ee_pipeline.cpp only when EE_ACT_BITS == 16; the TCN and
 * DeepConvLSTM builds (int8) pass the generated resolver directly, and the
 * linker drops this unit's code (--gc-sections).  When a later TFLM supports
 * int16 TRANSPOSE, or the exporter stops emitting an int16 TRANSPOSE, this
 * shim is harmless (the int16 case is the same data movement).
 */
#ifndef EE_TFLM_COMPAT_H
#define EE_TFLM_COMPAT_H

#include "tensorflow/lite/micro/micro_op_resolver.h"

/* Returns the resolver to give the interpreters: one that forwards every
 * lookup to `base` (which must already hold its registrations, i.e. after
 * EE_REGISTER_OPS) and adds the int16 case to TRANSPOSE when `base` has
 * TRANSPOSE.  The first call builds it (static storage, no heap); later calls
 * return the same object and ignore their argument.  `base` must outlive it. */
const tflite::MicroOpResolver &ee_int16_transpose_resolver(const tflite::MicroOpResolver &base);

#endif /* EE_TFLM_COMPAT_H */
