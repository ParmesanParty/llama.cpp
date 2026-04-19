#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Userdata for the fused cold MoE FFN custom op.
// Passed via ggml_custom_4d's userdata pointer.
// Must outlive the graph compute (stored on the hot cache struct).
//
// Scratch buffers are owned by this struct so there is exactly one set per
// cache instance (Finding 6 fix — was file-scope statics, which would race
// if ggml ever parallelized graph compute across slots, and would leak on
// repeated cache init/free cycles in multi-model-swap scenarios). Thread 0
// inside the compute kernel (re)allocates them on demand when total_slots
// exceeds the current max; the owning cache frees them in its destructor
// via llama_moe_fused_cold_free_scratch.
struct llama_moe_fused_cold_params {
    int n_expert;           // total experts (e.g., 256)
    int n_expert_used;      // router top-k (e.g., 8)
    int n_embd;             // embedding dim (e.g., 7168)
    int n_ff_exp;           // FFN intermediate dim per expert (e.g., 1280)

    // Scratch pointers (owned). See llama-moe-fused-cold.cpp for layout.
    // All nullptr on a zero-initialized struct — free() of nullptr is safe,
    // so the kernel's "grow on demand" path handles first-call allocation.
    void * scratch_f;        // malloc'd: n_ff_exp * max_total_slots floats
    void * scratch_q;        // malloc'd: (cur_q + act_q) row_size * max_total_slots bytes
    void * scratch_rc;       // calloc'd: n_expert int64s (row counts)
    void * scratch_rm;       // malloc'd: n_expert * max_total_slots slot_mapping entries
    void * scratch_ae;       // malloc'd: (n_expert + 1) ints (compact active list + sentinel)
    int    max_total_slots;  // high-water-mark; realloc if exceeded
};

// Custom op compute function for the fused cold MoE FFN path.
//
// Combines up + gate MUL_MAT_ID, SwiGLU, and down MUL_MAT_ID into a
// single dispatch with 3 internal thread synchronization barriers instead
// of the 7 barriers from the 4-node path.
//
// dst->src[] layout:
//   src[0] = up_exps    [n_embd,   n_ff_exp, n_expert] quantized weights
//   src[1] = gate_exps  [n_embd,   n_ff_exp, n_expert] quantized weights
//   src[2] = down_exps  [n_ff_exp, n_embd,   n_expert] quantized weights
//   src[3] = cur        [n_embd,   n_expert_used, n_tokens] F32 activations
//   src[4] = cold_ids   [n_expert_used, n_tokens] I32 (sentinel -1 = skip)
//
// dst = [n_embd, n_expert_used, n_tokens] F32
//
// Thread-parallel via ith/nth. Uses atomic work-stealing across experts
// and output rows, matching the existing MUL_MAT_ID parallelism pattern.
void llama_moe_fused_cold_compute(
    struct ggml_tensor * dst, int ith, int nth, void * userdata);

// Free the scratch buffers owned by a params struct. Safe on zero-
// initialized / partially-populated params (null checks internally).
// Called by llama_moe_hot_cache_free before free(cache).
void llama_moe_fused_cold_free_scratch(
    struct llama_moe_fused_cold_params * p);

#ifdef __cplusplus
}
#endif
