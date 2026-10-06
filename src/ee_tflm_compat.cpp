/*
 * ee_tflm_compat.cpp -- int16 TRANSPOSE for the TFLM of NCS 3.0.2
 *
 * See ee_tflm_compat.h.  No function-local statics (Zephyr has no
 * thread-safe-static guards) and no heap: the resolver is constructed with
 * placement new into file-scope storage on the first call, so a build that
 * never calls ee_int16_transpose_resolver() links none of this.
 */
#include "ee_tflm_compat.h"

#include <stddef.h>
#include <stdint.h>

#include <new>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

/* Tensor indices of TFLite's TRANSPOSE (as in tflite-micro kernels/transpose.cc). */
constexpr int kInputTensor = 0;
constexpr int kPermTensor = 1;
constexpr int kOutputTensor = 0;

/* Eval of the generated resolver's TRANSPOSE (float32 / int8 / whatever this
 * TFLM supports); TFLM's invoke signature carries no user pointer, hence a
 * file-scope variable.  Set once, before any interpreter exists. */
TfLiteStatus (*g_stock_transpose_invoke)(TfLiteContext *, TfLiteNode *) = nullptr;

TfLiteStatus transpose_eval(TfLiteContext *context, TfLiteNode *node)
{
    const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, kInputTensor);
    if (input == nullptr || input->type != kTfLiteInt16) {
        return g_stock_transpose_invoke(context, node);
    }
    const TfLiteEvalTensor *perm = tflite::micro::GetEvalInput(context, node, kPermTensor);
    TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);

    /* The stock Prepare has checked: input and output types equal, at most 5
     * dimensions, a 1-D int32 permutation of that length with entries in
     * range. */
    tflite::TransposeParams params;
    const int n = perm->dims->data[0];
    if (output->type != kTfLiteInt16 || n < 1 ||
        n > (int)(sizeof(params.perm) / sizeof(params.perm[0]))) {
        return kTfLiteError;
    }
    params.perm_count = (int8_t)n;
    for (int i = 0; i < n; i++) {
        params.perm[i] = perm->data.i32[i];
    }
    tflite::reference_ops::Transpose(params, tflite::micro::GetTensorShape(input),
                                     tflite::micro::GetTensorData<int16_t>(input),
                                     tflite::micro::GetTensorShape(output),
                                     tflite::micro::GetTensorData<int16_t>(output));
    return kTfLiteOk;
}

class Int16TransposeResolver : public tflite::MicroOpResolver {
public:
    explicit Int16TransposeResolver(const tflite::MicroOpResolver &base)
        : base_(base), transpose_(), has_transpose_(false)
    {
        const TFLMRegistration *stock = base.FindOp(tflite::BuiltinOperator_TRANSPOSE);
        if (stock != nullptr && stock->invoke != nullptr) {
            transpose_ = *stock;
            g_stock_transpose_invoke = stock->invoke;
            transpose_.invoke = transpose_eval;
            has_transpose_ = true;
        }
    }

    const TFLMRegistration *FindOp(tflite::BuiltinOperator op) const override
    {
        if (op == tflite::BuiltinOperator_TRANSPOSE && has_transpose_) {
            return &transpose_;
        }
        return base_.FindOp(op);
    }

    const TFLMRegistration *FindOp(const char *op) const override
    {
        return base_.FindOp(op);
    }

    tflite::TfLiteBridgeBuiltinParseFunction GetOpDataParser(tflite::BuiltinOperator op) const override
    {
        return base_.GetOpDataParser(op);
    }

    TF_LITE_REMOVE_VIRTUAL_DELETE

private:
    const tflite::MicroOpResolver &base_;
    TFLMRegistration transpose_;
    bool has_transpose_;
};

alignas(Int16TransposeResolver) uint8_t g_resolver_mem[sizeof(Int16TransposeResolver)];
Int16TransposeResolver *g_int16_resolver = nullptr;

} /* namespace */

const tflite::MicroOpResolver &ee_int16_transpose_resolver(const tflite::MicroOpResolver &base)
{
    if (g_int16_resolver == nullptr) {
        g_int16_resolver = new (g_resolver_mem) Int16TransposeResolver(base);
    }
    return *g_int16_resolver;
}
