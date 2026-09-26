#ifndef A4908882_05AD_4E5D_99F9_EFC0700B33F0
#define A4908882_05AD_4E5D_99F9_EFC0700B33F0

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace VectorMath {

// inline means replacing the function call at the call site with the actual code
// inline functions must be implemented in the header file to satisfy the ODR
// By default, M3 Pro has 32 128 bit vector registers per core - v0 to v31
// vdupq_n_f32 is an intrinsic that takes a single 32 bit float and duplicates it across all 4 lanes of a 128 bit vector register
// vld1q_f32 loads exactly 4 32-bit floating-point numbers at a time from a pointer position
// vfmaq_f32 is a fused-multiply add that does: dot_v = dot_v + (va * vb)
// vaddvq_f32 adds together all 32-bit floating-point elements in a single 128-bit vector
// register and returns the sum as a single scalar value
inline float dot_product(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size())
        throw std::invalid_argument("Vector dimensions must match.");

    // Creates 4 128-bit vector registers initialized to zero - accumulators for the dot product
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    // Data to be processed in chunks of 16 floats at a time
    const float* const a_ptr = a.data();
    const float* const b_ptr = b.data();

    // Loop Unrolling: Doing several iterations' worth of work in a single loop iteration
    // to increase instruction-level parallelism and reduce loop overhead
    std::size_t i = 0;

    // Process 16 floats at a time using SIMD
    for (; i + 15 < a.size(); i += 16) {
        // Fused multiply-add: acc0 = acc0 + (a_ptr[i] * b_ptr[i])
        // vld1q_f32 only loads 4 elements at a time
        acc0 = vfmaq_f32(acc0, vld1q_f32(a_ptr + i), vld1q_f32(b_ptr + i));

        // Fused multiply-add: acc1 = acc1 + (a_ptr[i + 4] * b_ptr[i + 4])
        acc1 = vfmaq_f32(acc1, vld1q_f32(a_ptr + i + 4), vld1q_f32(b_ptr + i + 4));

        // Fused multiply-add: acc2 = acc2 + (a_ptr[i + 8] * b_ptr[i + 8])
        acc2 = vfmaq_f32(acc2, vld1q_f32(a_ptr + i + 8), vld1q_f32(b_ptr + i + 8));

        // Fused multiply-add: acc3 = acc3 + (a_ptr[i + 12] * b_ptr[i + 12])
        acc3 = vfmaq_f32(acc3, vld1q_f32(a_ptr + i + 12), vld1q_f32(b_ptr + i + 12));
    }

    
    // Horizontal reduction of the four SIMD accumulators.
    float sum = vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);

    // Handle any remaining elements that didn't fit into the 16-element chunks
    for (; i < a.size(); ++i)
        sum += a[i] * b[i];

    // std::clamp ensures that the result is within the range [-1.0, 1.0]
    return std::clamp(sum, -1.0f, 1.0f);
}

inline float vectorMagnitude(std::span<const float> a) {
    // Calculate the magnitude. Load 4 vectors at a time each with length 128 bit and apply square
    // on all of them
    const float* a_ptr = a.data();

    float result = 0;

    std::size_t i = 0;
    for (; i + 15 < a.size(); i += 16) {
        // Load the 4 vectors
        float32x4_t acc0 = vld1q_f32(a_ptr + i);
        float32x4_t acc1 = vld1q_f32(a_ptr + i + 4);
        float32x4_t acc2 = vld1q_f32(a_ptr + i + 8);
        float32x4_t acc3 = vld1q_f32(a_ptr + i + 12);

        result += vaddvq_f32(vmulq_f32(acc0, acc0) + vmulq_f32(acc1, acc1) + vmulq_f32(acc2, acc2) +
                             vmulq_f32(acc3, acc3));
    }

    for (; i < a.size(); ++i) {
        result += a[i] * a[i];
    }
    return std::sqrt(result);
}



}

#endif /* A4908882_05AD_4E5D_99F9_EFC0700B33F0 */