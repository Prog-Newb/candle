#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

__device__ bool is_contiguous(
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

#define WHERE_OP(PRED_T, VAL_T, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims_and_strides, \
    const PRED_T *pred, \
    const VAL_T *on_true, \
    const VAL_T *on_false, \
    VAL_T *out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *pred_strides = dims_and_strides + 1 * num_dims; \
    const size_t *true_strides = dims_and_strides + 2 * num_dims; \
    const size_t *false_strides = dims_and_strides + 3 * num_dims; \
    bool pred_cont = is_contiguous(num_dims, dims, pred_strides); \
    bool true_cont = is_contiguous(num_dims, dims, true_strides); \
    bool false_cont = is_contiguous(num_dims, dims, false_strides); \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; \
         i += blockDim.x * gridDim.x) { \
        unsigned int tmp_i = i; \
        unsigned int pred_i = 0; \
        unsigned int true_i = 0; \
        unsigned int false_i = 0; \
        if (pred_cont) { pred_i = i; } \
        if (true_cont) { true_i = i; } \
        if (false_cont) { false_i = i; } \
        if (!pred_cont || !true_cont || !false_cont) { \
            tmp_i = i; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                if (!pred_cont)  pred_i  += i_dim * pred_strides[d]; \
                if (!true_cont)  true_i  += i_dim * true_strides[d]; \
                if (!false_cont) false_i += i_dim * false_strides[d]; \
                tmp_i /= dims[d]; \
            } \
        } \
        out[i] = pred[pred_i] != 0 ? on_true[true_i] : on_false[false_i]; \
    } \
}

/* CUDA-compatible predicate types, with ROCm value types already supported. */

WHERE_OP(uint8_t,  __half,       where_u8_f16)
WHERE_OP(uint8_t,  hip_bfloat16, where_u8_bf16)
WHERE_OP(uint8_t,  float,        where_u8_f32)
WHERE_OP(uint8_t,  double,       where_u8_f64)
WHERE_OP(uint8_t,  uint8_t,      where_u8_u8)
WHERE_OP(uint8_t,  uint32_t,     where_u8_u32)
WHERE_OP(uint8_t,  int64_t,      where_u8_i64)

WHERE_OP(uint32_t, __half,       where_u32_f16)
WHERE_OP(uint32_t, hip_bfloat16, where_u32_bf16)
WHERE_OP(uint32_t, float,        where_u32_f32)
WHERE_OP(uint32_t, double,       where_u32_f64)
WHERE_OP(uint32_t, uint8_t,      where_u32_u8)
WHERE_OP(uint32_t, uint32_t,     where_u32_u32)
WHERE_OP(uint32_t, int64_t,      where_u32_i64)

WHERE_OP(int64_t,  __half,       where_i64_f16)
WHERE_OP(int64_t,  hip_bfloat16, where_i64_bf16)
WHERE_OP(int64_t,  float,        where_i64_f32)
WHERE_OP(int64_t,  double,       where_i64_f64)
WHERE_OP(int64_t,  uint8_t,      where_i64_u8)
WHERE_OP(int64_t,  uint32_t,     where_i64_u32)
WHERE_OP(int64_t,  int64_t,      where_i64_i64)
