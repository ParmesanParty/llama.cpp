#include "llama-moe-hot-cache.h"
#include "llama-impl.h"
#include "llama-context.h"
#include "llama-graph.h"  // llm_graph_result::get_gf() for post-decode topk read
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------

struct llama_moe_hot_cache * llama_moe_hot_cache_init(
    struct llama_context * ctx,
    int K,
    int rebalance_interval) {

    if (K <= 0 || rebalance_interval <= 0) {
        LLAMA_LOG_WARN("%s: invalid K=%d or interval=%d, cache disabled\n",
                       __func__, K, rebalance_interval);
        return nullptr;
    }

    if (ctx == nullptr) {
        LLAMA_LOG_WARN("%s: null context, cache disabled\n", __func__);
        return nullptr;
    }

    const struct llama_model & model = ctx->get_model();
    const auto & hparams = model.hparams;

    // Defensive: non-MoE models report n_expert == 0. The init-time refusal
    // for architectures without ffn_gate_up_exps below (Decision #23) catches
    // most cases, but this cheap guard also protects against a pathological
    // model that has ffn_gate_up_exps on some layers but reports n_expert=0
    // in its metadata. Without this guard, the later
    // `calloc(cache->n_expert, sizeof(int32_t))` for hot_map_host and
    // window_counts would calloc(0, ...) — implementation-defined (may return
    // nullptr or a valid 1-byte alloc) and the promote loop then dereferences
    // it.
    if (hparams.n_expert == 0) {
        LLAMA_LOG_WARN("%s: hparams.n_expert == 0, cache disabled (not an MoE model?)\n",
                       __func__);
        return nullptr;
    }

    // Count MoE layers. The cache supports BOTH tensor formats that the
    // model loader produces (see llama-model.cpp:3072 for the qwen35moe
    // fallback): a layer counts as MoE if it has either the merged
    // ffn_gate_up_exps OR the split ffn_gate_exps + ffn_up_exps pair (with
    // ffn_down_exps required in both cases).
    //
    // Per-layer format is recorded by which fields the source layer has
    // populated. The cache mirrors that choice when allocating its hot
    // tensors. There is no architectural reason a model couldn't mix formats
    // across layers (e.g. one layer merged, another split), but we don't
    // know of any model that does — uniformity is the norm.
    int n_moe_layers = 0;
    int n_merged_layers = 0;
    int n_split_layers  = 0;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & layer = model.layers[il];
        if (layer.ffn_down_exps == nullptr) {
            continue;  // not an MoE layer in either format
        }
        if (layer.ffn_gate_up_exps != nullptr) {
            n_moe_layers++;
            n_merged_layers++;
        } else if (layer.ffn_gate_exps != nullptr && layer.ffn_up_exps != nullptr) {
            n_moe_layers++;
            n_split_layers++;
        }
    }

    if (n_moe_layers == 0) {
        LLAMA_LOG_WARN(
            "%s: no MoE layers found in either merged or split tensor format, "
            "cache disabled\n",
            __func__);
        return nullptr;
    }

    LLAMA_LOG_INFO(
        "%s: detected %d MoE layers (%d merged, %d split tensor format)\n",
        __func__, n_moe_layers, n_merged_layers, n_split_layers);

    auto * cache = (struct llama_moe_hot_cache *) calloc(1, sizeof(struct llama_moe_hot_cache));
    cache->mode = LLAMA_MOE_HOT_CACHE_FILLING;
    cache->K = K;
    cache->n_expert = (int) hparams.n_expert;
    cache->n_expert_used = (int) hparams.n_expert_used;

    // Fused cold kernel params. Populated once at init, consumed by the
    // graph builder's ggml_custom_4d call every decode.
    cache->fused_cold_params.n_expert      = cache->n_expert;
    cache->fused_cold_params.n_expert_used = cache->n_expert_used;
    cache->fused_cold_params.n_embd        = (int) hparams.n_embd;
    cache->fused_cold_params.n_ff_exp      = 0;
    // Read n_ff_exp from the first MoE layer's weight tensor.
    // All MoE layers in the model share the same n_ff_exp.
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & layer = model.layers[il];
        if (layer.ffn_down_exps != nullptr) {
            cache->fused_cold_params.n_ff_exp = (int) layer.ffn_down_exps->ne[0];
            break;
        }
    }

    cache->n_layers = n_moe_layers;
    cache->rebalance_interval = rebalance_interval;
    cache->ctx = ctx;
    cache->decode_counter = 0;
    // Telemetry counters (redundant after calloc but explicit for reader clarity).
    cache->hits_window = 0;
    cache->obs_window = 0;
    cache->hits_total = 0;
    cache->obs_total = 0;
    cache->last_rebalance_swap_sum = 0;
    cache->last_rebalance_swap_max = 0;

    cache->layers = (struct llama_moe_hot_cache_layer *) calloc(
        n_moe_layers, sizeof(struct llama_moe_hot_cache_layer));

    // Get the GPU backend buffer type for allocation.
    // NOTE: llama_context::backends is std::vector<ggml_backend_ptr> (unique-ptr
    // wrapped). Use llama_context::get_backend_ptrs() (returns
    // std::vector<ggml_backend_t> &) for direct iteration with raw handles.
    // Direct `ctx->backend_ptrs` access was replaced with the public accessor
    // because the member is private in llama_context.
    ggml_backend_buffer_type_t gpu_buft = nullptr;
#ifdef GGML_USE_CUDA
    for (ggml_backend_t backend : ctx->get_backend_ptrs()) {
        if (ggml_backend_is_cuda(backend)) {
            gpu_buft = ggml_backend_get_default_buffer_type(backend);
            break;
        }
    }
#endif
    if (gpu_buft == nullptr) {
        LLAMA_LOG_ERROR("%s: no CUDA backend found (GGML_USE_CUDA must be enabled)\n", __func__);
        // Delegate to free(): it null-guards every field, so partial-init
        // teardown stays correct as new fields are added. At this point only
        // cache->layers is set (no per-layer host arrays, no meta_ctx, no
        // backend_buffer, no stream). This keeps the rollback path identical
        // for every failure point below.
        llama_moe_hot_cache_free(cache);
        return nullptr;
    }

    // Allocate a small ggml_context for tensor metadata (the actual data lives
    // in a single backend buffer we allocate at the end). The maximum tensor
    // count per layer is 5 (split format: hot_gate_exps + hot_up_exps +
    // hot_down_exps + hot_map + cold_map). Merged format uses 4 (hot_gate_up_exps
    // + hot_down_exps + hot_map + cold_map). Allocating space for 5 per layer
    // is harmless when the model is merged — the extra slot stays unused.
    const size_t n_tensors_per_layer = 5;
    const size_t meta_size = ggml_tensor_overhead() * n_tensors_per_layer * n_moe_layers;
    struct ggml_init_params meta_params = {
        /*.mem_size   =*/ meta_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    cache->meta_ctx = ggml_init(meta_params);

    // Create tensors per layer
    int moe_idx = 0;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & layer = model.layers[il];
        if (layer.ffn_down_exps == nullptr) {
            continue;  // not an MoE layer
        }
        const bool is_merged = (layer.ffn_gate_up_exps != nullptr);
        const bool is_split  = (!is_merged
                                && layer.ffn_gate_exps != nullptr
                                && layer.ffn_up_exps != nullptr);
        if (!is_merged && !is_split) {
            continue;  // pathological mid-state — counted in neither bucket above
        }

        auto & hot_layer = cache->layers[moe_idx];
        hot_layer.current_size = 0;
        hot_layer.model_il = (int) il;  // record model-layer index for later graph reads

        if (is_merged) {
            // Merged format: one [n_embd, n_ff_exp*2, K] tensor.
            const auto * src = layer.ffn_gate_up_exps;
            hot_layer.hot_gate_up_exps = ggml_new_tensor_3d(
                cache->meta_ctx, src->type, src->ne[0], src->ne[1], K);
            char name[GGML_MAX_NAME];
            snprintf(name, sizeof(name), "hot_gate_up_exps_%zu", il);
            ggml_set_name(hot_layer.hot_gate_up_exps, name);
        } else {
            // Split format: two [n_embd, n_ff_exp, K] tensors. Both share the
            // same shape — assert at allocation time so a future model with
            // mismatched gate/up dims (currently impossible upstream) trips
            // here instead of producing a silent shape mismatch downstream.
            GGML_ASSERT(layer.ffn_gate_exps->type   == layer.ffn_up_exps->type);
            GGML_ASSERT(layer.ffn_gate_exps->ne[0]  == layer.ffn_up_exps->ne[0]);
            GGML_ASSERT(layer.ffn_gate_exps->ne[1]  == layer.ffn_up_exps->ne[1]);
            {
                const auto * src = layer.ffn_gate_exps;
                hot_layer.hot_gate_exps = ggml_new_tensor_3d(
                    cache->meta_ctx, src->type, src->ne[0], src->ne[1], K);
                char name[GGML_MAX_NAME];
                snprintf(name, sizeof(name), "hot_gate_exps_%zu", il);
                ggml_set_name(hot_layer.hot_gate_exps, name);
            }
            {
                const auto * src = layer.ffn_up_exps;
                hot_layer.hot_up_exps = ggml_new_tensor_3d(
                    cache->meta_ctx, src->type, src->ne[0], src->ne[1], K);
                char name[GGML_MAX_NAME];
                snprintf(name, sizeof(name), "hot_up_exps_%zu", il);
                ggml_set_name(hot_layer.hot_up_exps, name);
            }
        }

        // hot_down_exps is allocated for both formats.
        {
            const auto * src = layer.ffn_down_exps;
            hot_layer.hot_down_exps = ggml_new_tensor_3d(
                cache->meta_ctx, src->type, src->ne[0], src->ne[1], K);
            char name[GGML_MAX_NAME];
            snprintf(name, sizeof(name), "hot_down_exps_%zu", il);
            ggml_set_name(hot_layer.hot_down_exps, name);
        }

        // Remapping tables (int32, length n_expert)
        hot_layer.hot_map = ggml_new_tensor_1d(
            cache->meta_ctx, GGML_TYPE_I32, cache->n_expert);
        char name[GGML_MAX_NAME];
        snprintf(name, sizeof(name), "hot_map_%zu", il);
        ggml_set_name(hot_layer.hot_map, name);

        hot_layer.cold_map = ggml_new_tensor_1d(
            cache->meta_ctx, GGML_TYPE_I32, cache->n_expert);
        snprintf(name, sizeof(name), "cold_map_%zu", il);
        ggml_set_name(hot_layer.cold_map, name);

        // Host-side shadows
        hot_layer.hot_map_host = (int32_t *) calloc(cache->n_expert, sizeof(int32_t));
        for (int i = 0; i < cache->n_expert; ++i) {
            hot_layer.hot_map_host[i] = -1;
        }
        hot_layer.slot_to_expert = (int32_t *) calloc(K, sizeof(int32_t));
        for (int i = 0; i < K; ++i) {
            hot_layer.slot_to_expert[i] = -1;
        }

        // Tumbling-window expert selection counts (Decision #27). calloc
        // zero-initializes — the first rebalance window starts from zero.
        hot_layer.window_counts = (uint32_t *) calloc(cache->n_expert, sizeof(uint32_t));

        moe_idx++;
    }

    // Allocate the backing backend buffer for all tensors in meta_ctx
    cache->backend_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        cache->meta_ctx, gpu_buft);
    if (cache->backend_buffer == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to allocate VRAM for hot cache\n", __func__);
        // Per-layer hot_map_host / slot_to_expert / window_counts are already
        // allocated above. The earlier inline rollback only freed meta_ctx +
        // layers + cache, leaking the per-layer host arrays. Delegate to
        // free() instead — it walks cache->layers[] and frees each.
        llama_moe_hot_cache_free(cache);
        return nullptr;
    }

    // Phase 10 perf fix: mark the hot cache backend buffer as WEIGHTS so
    // the scheduler's backend assignment heuristic
    // (ggml_backend_sched_backend_id_from_cur in ggml-backend.cpp) sees the
    // hot tensors as "a weight lives on CUDA" and steers MUL_MAT_ID ops
    // that consume them onto CUDA. Without this, the buffer has the
    // default USAGE_ANY, the "weight src → pin this op to that backend"
    // branch never fires, and the hot MUL_MAT_ID nodes fall through to
    // later passes that drag them onto CPU alongside the cold MUL_MAT_IDs
    // (whose real weight tensors are WEIGHTS-flagged on the CPU backend).
    // Observed symptom: n_splits unchanged but 2/3 of per-layer hot
    // MUL_MAT_IDs run on CPU, ~5x decode-throughput regression vs
    // single-path baseline. With this call, hot MUL_MAT_IDs run on CUDA
    // as intended. See Decision #35 in the progress log.
    ggml_backend_buffer_set_usage(
        cache->backend_buffer,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Initialize hot_map tensors to -1 and cold_map to identity on device
    std::vector<int32_t> init_minus1(cache->n_expert, -1);
    std::vector<int32_t> init_identity(cache->n_expert);
    for (int i = 0; i < cache->n_expert; ++i) init_identity[i] = i;

    for (int i = 0; i < n_moe_layers; ++i) {
        ggml_backend_tensor_set(
            cache->layers[i].hot_map,
            init_minus1.data(), 0,
            cache->n_expert * sizeof(int32_t));
        ggml_backend_tensor_set(
            cache->layers[i].cold_map,
            init_identity.data(), 0,
            cache->n_expert * sizeof(int32_t));
    }

    // Create CUDA stream and event for async copies
#ifdef GGML_USE_CUDA
    // cudaStreamNonBlocking: do NOT implicitly synchronize with the null
    // (default) stream. Otherwise every stray null-stream operation elsewhere
    // in the process would serialize our async copies against the compute
    // stream and kill the overlap we depend on for throughput.
    //
    // Error handling: we must check the return values of BOTH create calls
    // and of the initial event record. cudaStreamCreateWithFlags writes to
    // &stream even on failure in some driver versions, so we initialize
    // `stream` and `event` to nullptr and assign cache->copy_stream /
    // cache->rebalance_event strictly AFTER a successful create. That way
    // llama_moe_hot_cache_free() sees consistent state (either a valid
    // handle or nullptr, never uninitialized pool garbage) if we roll back.
    // The `cache` struct was calloc'd, so cache->copy_stream and
    // cache->rebalance_event are already nullptr on entry to this block.
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        LLAMA_LOG_ERROR("%s: cudaStreamCreateWithFlags failed: %s — hot cache disabled\n",
                        __func__, cudaGetErrorString(err));
        llama_moe_hot_cache_free(cache);
        return nullptr;
    }
    cache->copy_stream = (void *) stream;

    cudaEvent_t event = nullptr;
    err = cudaEventCreate(&event);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        LLAMA_LOG_ERROR("%s: cudaEventCreate failed: %s — hot cache disabled\n",
                        __func__, cudaGetErrorString(err));
        llama_moe_hot_cache_free(cache);
        return nullptr;
    }
    cache->rebalance_event = (void *) event;

    // Prime the event with an initial record on the copy stream so the first
    // cudaStreamWaitEvent call from llama_context::decode (Task 9.1 Step 5) has
    // a defined prior state. CUDA docs say waiting on a never-recorded event
    // is a no-op on most drivers, but this is version-dependent — explicit
    // priming removes the ambiguity at zero runtime cost.
    err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        LLAMA_LOG_ERROR("%s: cudaEventRecord (prime) failed: %s — hot cache disabled\n",
                        __func__, cudaGetErrorString(err));
        llama_moe_hot_cache_free(cache);
        return nullptr;
    }

    // Pinned scratch buffer (Decision #30 Path B). The original plan called
    // cudaHostRegister directly on the model's mmap'd expert weight regions
    // so cudaMemcpyAsync could DMA from them without a staging copy. That
    // plan works on systems where pinned host memory is cheap, but the
    // production RTX 4090 / CUDA 12.x driver caps per-process pinned host
    // memory at ~50 GB (independent of RLIMIT_MEMLOCK and system RAM — the
    // ceiling appears to be the driver's IOMMU mapping table). Qwen3.5-122B-
    // A10B Q4_K_L alone is 75 GB of expert weights, so the up-front pinning
    // cascaded into 42/144 cudaHostRegister failures, filled the journal
    // with noise, and left the cache in a partial-pinning state with
    // synchronous DMA fallback for the unpinned layers.
    //
    // Path B: allocate ONE pinned scratch buffer sized to the largest hot
    // tensor's VRAM footprint. Phase 9 FILLING and Phase 10 STEADY rebalance
    // copies stage one (layer, tensor) slice at a time through the scratch:
    //
    //     memcpy(scratch, src + row_offset, bytes);         // mmap'd → pinned (CPU)
    //     cudaMemcpyAsync(vram_slot, scratch, bytes, H2D);  // pinned → VRAM (GPU DMA)
    //
    // The scratch cost is bounded by the biggest single hot tensor
    // (typically a few hundred MiB) instead of the total expert footprint.
    // The cudaMemcpyAsync call is truly async because the source is pinned;
    // serialization between successive copies is handled at the caller (the
    // scratch is reused after the previous copy's rebalance_event has
    // synchronized).
    // Finding 4 fix: scratch sized to the max PER-LAYER SUM of all hot
    // tensors (merged: gate_up + down, split: gate + up + down) so that
    // promote_layer can stage all three (or two) H2D copies into
    // offset-disjoint scratch regions and issue a SINGLE
    // cudaStreamSynchronize at the end, rather than three serialized
    // sync points that block the CPU at FILLING warmup.
    //
    // Prior sizing was the max per-tensor, which forced promote_layer
    // to sync between every tensor because they all aliased the same
    // scratch region.
    size_t max_scratch_bytes = 0;
    for (int i = 0; i < n_moe_layers; ++i) {
        const auto & hl = cache->layers[i];
        size_t layer_budget = 0;
        if (hl.hot_gate_up_exps) layer_budget += ggml_nbytes(hl.hot_gate_up_exps);
        if (hl.hot_gate_exps)    layer_budget += ggml_nbytes(hl.hot_gate_exps);
        if (hl.hot_up_exps)      layer_budget += ggml_nbytes(hl.hot_up_exps);
        if (hl.hot_down_exps)    layer_budget += ggml_nbytes(hl.hot_down_exps);
        if (layer_budget > max_scratch_bytes) {
            max_scratch_bytes = layer_budget;
        }
    }
    if (max_scratch_bytes == 0) {
        LLAMA_LOG_ERROR("%s: every hot tensor has zero bytes — refusing init\n", __func__);
        llama_moe_hot_cache_free(cache);
        return nullptr;
    }

    {
        void * scratch_ptr = nullptr;
        cudaError_t sc_err = cudaMallocHost(&scratch_ptr, max_scratch_bytes);
        if (sc_err != cudaSuccess) {
            (void) cudaGetLastError();
            LLAMA_LOG_ERROR(
                "%s: cudaMallocHost(scratch=%.2f MiB) failed: %s — hot cache disabled\n",
                __func__, max_scratch_bytes / 1024.0 / 1024.0,
                cudaGetErrorString(sc_err));
            llama_moe_hot_cache_free(cache);
            return nullptr;
        }
        cache->scratch       = scratch_ptr;
        cache->scratch_bytes = max_scratch_bytes;
        LLAMA_LOG_INFO(
            "%s: allocated %.2f MiB pinned scratch buffer (Path B — "
            "sum of per-layer gate+up+down to enable batched promote)\n",
            __func__, max_scratch_bytes / 1024.0 / 1024.0);
    }

    // PERF: pinned D2H landing zone for post_decode's batched argsort read.
    // Pre-allocated once at init; reused every decode. Size = n_layers * n_expert
    // ints (for single-token decode; prefill ubatches fall back to sync gets).
    {
        cache->ids_pinned_bytes = (size_t) n_moe_layers * (size_t) cache->n_expert * sizeof(int32_t);
        void * ids_ptr = nullptr;
        cudaError_t ip_err = cudaMallocHost(&ids_ptr, cache->ids_pinned_bytes);
        if (ip_err != cudaSuccess) {
            (void) cudaGetLastError();
            LLAMA_LOG_WARN(
                "%s: cudaMallocHost(ids_pinned=%.2f KiB) failed: %s — falling back to pageable\n",
                __func__, cache->ids_pinned_bytes / 1024.0,
                cudaGetErrorString(ip_err));
            cache->ids_pinned = nullptr;
            cache->ids_pinned_bytes = 0;
        } else {
            cache->ids_pinned = ids_ptr;
            LLAMA_LOG_INFO("%s: allocated %.2f KiB pinned ids buffer for batched post_decode D2H\n",
                           __func__, cache->ids_pinned_bytes / 1024.0);
        }
    }
#endif

    const size_t buf_size = ggml_backend_buffer_get_size(cache->backend_buffer);
    LLAMA_LOG_INFO("%s: allocated hot cache: K=%d, %d layers, %.2f GB VRAM\n",
                   __func__, K, n_moe_layers, buf_size / 1e9);

    // Pre-reserve rebalance scratch (Finding 5 fix). The cache struct is
    // calloc'd so the vector members have zero'd internals — officially UB
    // to invoke methods on "not-yet-alive" objects. Placement-new them here
    // so their lifetime is correctly started; free() calls the destructors
    // explicitly before free(cache). This avoids relying on libstdc++'s
    // de-facto-valid-empty-vector behavior for zero'd memory.
    new (&cache->rebalance_ranked)      std::vector<std::pair<uint32_t, int>>();
    new (&cache->rebalance_new_hot)     std::vector<int32_t>();
    new (&cache->rebalance_swap_counts) std::vector<int>();
    new (&cache->rebalance_layer_ent)   std::vector<double>();
    cache->rebalance_ranked.reserve((size_t) cache->n_expert);
    cache->rebalance_new_hot.resize((size_t) cache->K);
    cache->rebalance_swap_counts.resize((size_t) cache->n_layers, 0);
    cache->rebalance_layer_ent.resize((size_t) cache->n_layers, 0.0);

    return cache;
}

void llama_moe_hot_cache_free(struct llama_moe_hot_cache * cache) {
    if (cache == nullptr) {
        return;
    }

#ifdef GGML_USE_CUDA
    // CRITICAL: drain any in-flight async copies on copy_stream BEFORE touching
    // any resource they might still be reading. If a sleep/wake cycle triggers
    // free() while a FILLING promotion or STEADY rebalance is in flight, the
    // copy stream may be mid-DMA from pinned host memory into VRAM slots.
    // cudaHostUnregister on the source pages during an active DMA is undefined
    // behavior, and cudaStreamDestroy on an active stream is also formally UB.
    // A single synchronize is cheap (worst case ~350 ms on a massive in-flight
    // rebalance, which only happens at sleep time anyway) and load-bearing for
    // the sleep/wake lifecycle test (Task 11.5).
    if (cache->copy_stream != nullptr) {
        cudaStreamSynchronize((cudaStream_t) cache->copy_stream);
    }

    // Release the pinned scratch buffer (Path B). Order: stream sync above
    // has already drained any in-flight cudaMemcpyAsync that might still be
    // reading from it, so cudaFreeHost is safe here. Null-guarded so partial
    // init rollback (via llama_moe_hot_cache_free) works correctly when the
    // scratch was never allocated.
    if (cache->scratch != nullptr) {
        cudaError_t err = cudaFreeHost(cache->scratch);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            LLAMA_LOG_WARN("%s: cudaFreeHost(scratch) failed: %s\n",
                           __func__, cudaGetErrorString(err));
        }
        cache->scratch = nullptr;
        cache->scratch_bytes = 0;
    }

    // Release the pinned ids buffer (PERF batched D2H).
    if (cache->ids_pinned != nullptr) {
        cudaError_t err = cudaFreeHost(cache->ids_pinned);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            LLAMA_LOG_WARN("%s: cudaFreeHost(ids_pinned) failed: %s\n",
                           __func__, cudaGetErrorString(err));
        }
        cache->ids_pinned = nullptr;
        cache->ids_pinned_bytes = 0;
    }

    if (cache->copy_stream != nullptr) {
        cudaStreamDestroy((cudaStream_t) cache->copy_stream);
    }
    if (cache->rebalance_event != nullptr) {
        cudaEventDestroy((cudaEvent_t) cache->rebalance_event);
    }
#endif

    if (cache->backend_buffer != nullptr) {
        ggml_backend_buffer_free(cache->backend_buffer);
    }
    if (cache->meta_ctx != nullptr) {
        ggml_free(cache->meta_ctx);
    }

    if (cache->layers != nullptr) {
        for (int i = 0; i < cache->n_layers; ++i) {
            free(cache->layers[i].hot_map_host);
            free(cache->layers[i].slot_to_expert);
            free(cache->layers[i].window_counts);
        }
        free(cache->layers);
    }

    // Free the fused-cold kernel's scratch (Finding 6 fix: was file-scope
    // static, now owned by params on the cache). Safe on zero-initialized
    // params (first-call grow path leaves all pointers nullptr).
    llama_moe_fused_cold_free_scratch(&cache->fused_cold_params);

    // Destruct the rebalance scratch vectors before free(cache) — matches
    // the placement-new in init. Skipping this would leak their internal
    // heap buffers (~3.7 KB per cache).
    using vec_pair_u32_int = std::vector<std::pair<uint32_t, int>>;
    using vec_i32          = std::vector<int32_t>;
    using vec_int          = std::vector<int>;
    using vec_double       = std::vector<double>;
    cache->rebalance_ranked.~vec_pair_u32_int();
    cache->rebalance_new_hot.~vec_i32();
    cache->rebalance_swap_counts.~vec_int();
    cache->rebalance_layer_ent.~vec_double();

    free(cache);
}

// -----------------------------------------------------------------------------
// Accessors — called by the graph builder with the MODEL layer index (`il`),
// not the packed MoE-cache index. Perform a linear scan over cache->layers[]
// matching `.model_il == il` to find the correct entry. Returns nullptr if
// no cache layer corresponds to that model layer — the graph builder then
// short-circuits the dual-path and falls back to the single-call path.
//
// Scan cost: O(n_moe_layers) per accessor call, four accessors per MoE layer
// during graph build. For qwen35moe (48 MoE layers) that is ~9K comparisons
// per graph build, which happens at most once per batch. Negligible. If a
// future architecture needs sub-microsecond lookups, swap this for an
// `int model_il_to_cache_idx[n_model_layers]` reverse-lookup table populated
// in Task 3.1 alongside `model_il` itself.
// -----------------------------------------------------------------------------

static const struct llama_moe_hot_cache_layer * find_layer_by_model_il(
    const struct llama_moe_hot_cache * cache, int il) {
    if (cache == nullptr || il < 0) {
        return nullptr;
    }
    for (int i = 0; i < cache->n_layers; ++i) {
        if (cache->layers[i].model_il == il) {
            return &cache->layers[i];
        }
    }
    return nullptr;
}

struct ggml_tensor * llama_moe_hot_cache_get_hot_map(
    const struct llama_moe_hot_cache * cache, int il) {
    const auto * layer = find_layer_by_model_il(cache, il);
    return layer ? layer->hot_map : nullptr;
}

struct ggml_tensor * llama_moe_hot_cache_get_cold_map(
    const struct llama_moe_hot_cache * cache, int il) {
    const auto * layer = find_layer_by_model_il(cache, il);
    return layer ? layer->cold_map : nullptr;
}

struct ggml_tensor * llama_moe_hot_cache_get_hot_gate_up(
    const struct llama_moe_hot_cache * cache, int il) {
    const auto * layer = find_layer_by_model_il(cache, il);
    return layer ? layer->hot_gate_up_exps : nullptr;
}

struct ggml_tensor * llama_moe_hot_cache_get_hot_gate(
    const struct llama_moe_hot_cache * cache, int il) {
    const auto * layer = find_layer_by_model_il(cache, il);
    return layer ? layer->hot_gate_exps : nullptr;
}

struct ggml_tensor * llama_moe_hot_cache_get_hot_up(
    const struct llama_moe_hot_cache * cache, int il) {
    const auto * layer = find_layer_by_model_il(cache, il);
    return layer ? layer->hot_up_exps : nullptr;
}

struct ggml_tensor * llama_moe_hot_cache_get_hot_down(
    const struct llama_moe_hot_cache * cache, int il) {
    const auto * layer = find_layer_by_model_il(cache, il);
    return layer ? layer->hot_down_exps : nullptr;
}

bool llama_moe_hot_cache_layer_has_hot(
    const struct llama_moe_hot_cache * cache, int il) {
    const auto * layer = find_layer_by_model_il(cache, il);
    return layer != nullptr && layer->current_size > 0;
}

void llama_moe_hot_cache_build_cold_map(
    int n_expert,
    const int32_t * hot_map_host,
    int32_t * out) {
    if (n_expert <= 0 || hot_map_host == nullptr || out == nullptr) {
        return;
    }
    for (int i = 0; i < n_expert; ++i) {
        out[i] = (hot_map_host[i] >= 0) ? -1 : i;
    }
}

// -----------------------------------------------------------------------------
// Promotion helper (FILLING mode)
// -----------------------------------------------------------------------------

#ifdef GGML_USE_CUDA
// Stage one tensor's worth of newly-promoted experts through the Path B pinned
// scratch buffer and issue a single cudaMemcpyAsync for the whole contiguous
// destination range. Returns true on success, false if any step failed (which
// also means the caller should NOT advance current_size for the failed copy).
//
// Invariants relied on by the single-batch memcpy:
//   - New experts are always appended at slots [current_size .. current_size+n_new),
//     so the destination VRAM range is contiguous along axis 2.
//   - The hot tensor's shape is [ne0, ne1, K] with the same ne0/ne1/dtype as
//     the source, so nb[2] (bytes per expert) matches by construction.
//   - The scratch buffer was sized in init to the largest hot tensor's full
//     nbytes = K * expert_bytes. Up to K contiguous experts fit at once.
//
// The caller must have already cudaStreamSynchronize'd copy_stream since the
// previous scratch user if it wants this function's memcpy into scratch to
// be safe. In practice the caller (promote_layer) does exactly one sync at
// the bottom of its body, and each tensor copy inside a single layer is
// serialized by construction (we don't start the next until the current
// cudaMemcpyAsync has been queued).
static bool promote_stage_tensor(
    struct llama_moe_hot_cache * cache,
    const struct ggml_tensor * src_tensor,
    struct ggml_tensor * dst_tensor,
    const int32_t * new_experts,
    size_t n_new,
    int base_slot,
    size_t scratch_offset /* bytes into cache->scratch */) {

    if (src_tensor == nullptr || dst_tensor == nullptr) {
        return true;  // nothing to do (e.g., split-format when merged is queried)
    }
    const size_t expert_bytes = src_tensor->nb[2];
    const size_t total_bytes  = expert_bytes * n_new;
    if (scratch_offset + total_bytes > cache->scratch_bytes) {
        LLAMA_LOG_ERROR(
            "%s: scratch budget exceeded (offset=%zu + bytes=%zu > %zu)\n",
            __func__, scratch_offset, total_bytes, cache->scratch_bytes);
        return false;
    }

    // Gather n_new experts from the mmap'd source into the pinned scratch
    // at the caller-provided offset. Disjoint offsets across the 2 or 3
    // tensors in a single promote_layer call enable a single trailing sync
    // for all their copies (Finding 4 fix).
    const char * src_base = (const char *) src_tensor->data;
    char       * scratch  = (char       *) cache->scratch + scratch_offset;
    for (size_t s = 0; s < n_new; ++s) {
        const int32_t e = new_experts[s];
        memcpy(scratch + s * expert_bytes,
               src_base + (size_t) e * expert_bytes,
               expert_bytes);
    }

    // One contiguous async copy scratch → VRAM slot range [base_slot..base_slot+n_new).
    char * dst_base = (char *) dst_tensor->data + (size_t) base_slot * expert_bytes;
    cudaError_t err = cudaMemcpyAsync(
        dst_base, scratch, total_bytes,
        cudaMemcpyHostToDevice,
        (cudaStream_t) cache->copy_stream);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        LLAMA_LOG_ERROR("%s: cudaMemcpyAsync failed: %s\n",
                        __func__, cudaGetErrorString(err));
        return false;
    }
    return true;
}
#endif  // GGML_USE_CUDA

void llama_moe_hot_cache_promote_layer(
    struct llama_moe_hot_cache * cache,
    int layer_idx,
    const int32_t * ids,
    int n_ids) {

    if (cache == nullptr || layer_idx < 0 || layer_idx >= cache->n_layers) {
        return;
    }
    auto & layer = cache->layers[layer_idx];
    if (layer.current_size >= cache->K) {
        return;  // already full
    }
    if (ids == nullptr || n_ids <= 0) {
        return;
    }

    // Collect unique novel experts (bounds-checked, sentinel-filtered,
    // dedup'd against both hot_map_host and prior entries in new_experts).
    // n_expert is typically 256, so a small stack-local dedup via
    // hot_map_host is free — anything already in hot_map is skipped, and
    // we dedup within the same call by checking new_experts inline.
    std::vector<int32_t> new_experts;
    new_experts.reserve((size_t) n_ids);
    for (int i = 0; i < n_ids; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= cache->n_expert) continue;
        if (layer.hot_map_host[e] >= 0) continue;  // already hot
        bool dup = false;
        for (int32_t q : new_experts) {
            if (q == e) { dup = true; break; }
        }
        if (dup) continue;
        new_experts.push_back(e);
    }

    // Cap at available slots. If avail == 0 after the current_size >= K
    // early-return above, `avail` below is still non-negative.
    const int avail = cache->K - layer.current_size;
    if ((int) new_experts.size() > avail) {
        new_experts.resize((size_t) avail);
    }
    if (new_experts.empty()) {
        return;
    }

    const int base_slot = layer.current_size;

    // Update host-side shadow UNCONDITIONALLY. This happens before any device
    // copies so unit tests (which construct a fake cache without CUDA) still
    // see the correct hot_map_host / slot_to_expert state, and so the
    // rollback path below can bail out cleanly if a GPU copy fails.
    for (size_t s = 0; s < new_experts.size(); ++s) {
        const int32_t e   = new_experts[s];
        const int     slot = base_slot + (int) s;
        layer.hot_map_host[e]      = slot;
        layer.slot_to_expert[slot] = e;
    }

    // Issue device copies only if the cache has real GPU buffers. The unit
    // test's fake cache sets cache->ctx == nullptr AND zeroes every tensor
    // pointer, so this whole block is skipped and the test exercises only
    // the host-side state machine.
    //
    // NOTE: layer_idx is the MoE-cache packed index (0..n_layers-1); the
    // corresponding model layer is model.layers[layer.model_il]. For
    // qwen35moe (all layers are MoE) model_il == layer_idx, but the mapping
    // is authoritative — never assume identity here.
#ifdef GGML_USE_CUDA
    if (cache->ctx != nullptr && cache->copy_stream != nullptr &&
        cache->scratch != nullptr) {

        const auto & model_layer = cache->ctx->get_model().layers[layer.model_il];
        bool all_ok = true;
        size_t off = 0;

        // Finding 4 fix: all tensors' gather-then-async-copy pairs issued
        // back-to-back with offset-disjoint scratch regions (init sizes
        // scratch to max(layer sum of gate+up+down)). One trailing
        // cudaStreamSynchronize covers every DMA launched above, collapsing
        // the prior 3 sync points per layer into 1.
        if (all_ok && layer.hot_gate_up_exps != nullptr) {
            all_ok = promote_stage_tensor(
                cache, model_layer.ffn_gate_up_exps, layer.hot_gate_up_exps,
                new_experts.data(), new_experts.size(), base_slot, off);
            off += ggml_nbytes(layer.hot_gate_up_exps);
        }
        if (all_ok && layer.hot_gate_exps != nullptr) {
            all_ok = promote_stage_tensor(
                cache, model_layer.ffn_gate_exps, layer.hot_gate_exps,
                new_experts.data(), new_experts.size(), base_slot, off);
            off += ggml_nbytes(layer.hot_gate_exps);
        }
        if (all_ok && layer.hot_up_exps != nullptr) {
            all_ok = promote_stage_tensor(
                cache, model_layer.ffn_up_exps, layer.hot_up_exps,
                new_experts.data(), new_experts.size(), base_slot, off);
            off += ggml_nbytes(layer.hot_up_exps);
        }
        if (all_ok && layer.hot_down_exps != nullptr) {
            all_ok = promote_stage_tensor(
                cache, model_layer.ffn_down_exps, layer.hot_down_exps,
                new_experts.data(), new_experts.size(), base_slot, off);
            off += ggml_nbytes(layer.hot_down_exps);
        }

        if (all_ok) {
            cudaStreamSynchronize((cudaStream_t) cache->copy_stream);
        }

        // Push the updated hot_map AND cold_map to device. hot_map_host is
        // small (n_expert int32s ≈ 1 KB for typical models), so going through
        // ggml_backend_tensor_set is simpler than staging through scratch.
        // Note: ggml_backend_tensor_set is synchronous on the backend's
        // internal stream, which for CUDA is the compute stream — that's
        // fine here because we're in post_decode (between graph executes).
        //
        // cold_map maintenance (Phase 10 dual-path correctness fix):
        // cold_map must be the complement of hot_map — cold_map[e] = e for
        // cold experts, cold_map[e] = -1 for hot experts. It was initialized
        // to identity in init and now needs to reflect the newly promoted
        // experts. Without this update, the dual-path sum double-counts:
        // hot_ids[i] points to slot s, cold_ids[i] ALSO points to expert e,
        // and ggml_add(hot_mul, cold_mul) adds both contributions, producing
        // 2x the correct output for promoted experts. Recompute the full
        // cold_map from hot_map_host and push. Cost: O(n_expert) ≈ 256 ops,
        // negligible compared to the promote H2D copies above.
        if (all_ok && layer.hot_map != nullptr) {
            ggml_backend_tensor_set(
                layer.hot_map, layer.hot_map_host, 0,
                (size_t) cache->n_expert * sizeof(int32_t));
        }
        if (all_ok && layer.cold_map != nullptr) {
            std::vector<int32_t> cold_map_buf((size_t) cache->n_expert);
            llama_moe_hot_cache_build_cold_map(
                cache->n_expert, layer.hot_map_host, cold_map_buf.data());
            ggml_backend_tensor_set(
                layer.cold_map, cold_map_buf.data(), 0,
                (size_t) cache->n_expert * sizeof(int32_t));
        }

        if (!all_ok) {
            // Rollback the host-side shadow so a retry on the next decode
            // step can try again. Without this, a partial-copy failure would
            // mark the experts as "already hot" in hot_map_host but the VRAM
            // slots would be garbage, and the dual-path would read
            // uninitialized data for those slots.
            for (size_t s = 0; s < new_experts.size(); ++s) {
                const int32_t e   = new_experts[s];
                const int     slot = base_slot + (int) s;
                layer.hot_map_host[e]      = -1;
                layer.slot_to_expert[slot] = -1;
            }
            return;
        }
    }
#endif  // GGML_USE_CUDA

    // Only advance current_size after (possibly empty) device copies all succeed.
    // This is what flips layer_has_hot() true once current_size > 0.
    layer.current_size += (int) new_experts.size();
}

// -----------------------------------------------------------------------------
// Task 10.1: STEADY mode rebalance — swap helper + window-counts-driven driver
// -----------------------------------------------------------------------------

int llama_moe_hot_cache_swap_layer(
    struct llama_moe_hot_cache * cache,
    int layer_idx,
    const int32_t * new_hot_set,
    int new_hot_len) {

    if (cache == nullptr || layer_idx < 0 || layer_idx >= cache->n_layers) {
        return 0;
    }
    if (new_hot_set == nullptr || new_hot_len <= 0) {
        return 0;
    }
    auto & layer = cache->layers[layer_idx];

    // Phase 1: mark which experts the new set requires.
    // in_new[e] == true means "this expert should be hot after the swap".
    // Bounds-check each entry — caller may pass out-of-range ids (e.g., a
    // partial-sort top-K that includes zero-count slots up to K when the
    // layer's window has fewer than K distinct activations).
    std::vector<bool> in_new((size_t) cache->n_expert, false);
    for (int i = 0; i < new_hot_len; ++i) {
        const int32_t e = new_hot_set[i];
        if (e >= 0 && e < cache->n_expert) {
            in_new[(size_t) e] = true;
        }
    }

    // Phase 2: scan current slots and collect evictions.
    // Evicted experts: hot_map_host[e] is cleared and the slot is marked
    // empty (-1), so the Phase 3 promote loop correctly identifies them as
    // "not currently hot" and uses their slots for promotion.
    std::vector<int32_t> evict_slots;
    evict_slots.reserve((size_t) cache->K);
    for (int slot = 0; slot < cache->K; ++slot) {
        const int32_t e = layer.slot_to_expert[slot];
        if (e < 0) continue;  // already empty slot (e.g., partial fill)
        if (!in_new[(size_t) e]) {
            evict_slots.push_back(slot);
            layer.hot_map_host[e]      = -1;
            layer.slot_to_expert[slot] = -1;
        }
    }

    // Phase 3: collect experts from the new set that aren't currently hot.
    // "Shared" experts (in both old and new sets) are already hot from the
    // initial scan — the hot_map_host[e] >= 0 check skips them. Only novel
    // experts reach promote_experts.
    std::vector<int32_t> promote_experts;
    promote_experts.reserve((size_t) new_hot_len);
    for (int i = 0; i < new_hot_len; ++i) {
        const int32_t e = new_hot_set[i];
        if (e < 0 || e >= cache->n_expert) continue;
        if (layer.hot_map_host[e] < 0) {
            promote_experts.push_back(e);
        }
    }

    // Phase 4: pair-wise swap. Each evicted slot is reused for exactly one
    // promotion. In STEADY mode the rebalance driver always passes
    // new_hot_len == K and current_size == K, so |evict| == |promote| by
    // construction. The std::min() guards against the partial-fill edge
    // case where the two cardinalities could differ — extras are dropped.
    const size_t n_swap = std::min(evict_slots.size(), promote_experts.size());

    for (size_t i = 0; i < n_swap; ++i) {
        const int     slot = evict_slots[i];
        const int32_t e    = promote_experts[i];
        layer.hot_map_host[e]      = slot;
        layer.slot_to_expert[slot] = e;
    }

#ifdef GGML_USE_CUDA
    // Phase 5: stage promoted experts into VRAM and push map tensors.
    //
    // Unlike promote_layer (which stages contiguous slots [base_slot,
    // base_slot + n_new)), swap_layer's target slots are non-contiguous
    // (they're the evicted slot indices). We issue one cudaMemcpyAsync per
    // (expert, slot) pair. The source data lives in mmap'd host memory
    // (not pinned), so CUDA internally serializes these copies via its own
    // staging buffer — that's acceptable at 40-decode cadence where total
    // rebalance latency runs into the tens of milliseconds, not seconds.
    //
    // For the unit-test path (cache->ctx == nullptr), the whole block is
    // skipped and only the host-side shadow is mutated. Tests can therefore
    // verify the swap logic on a fake cache without bringing up CUDA.
    if (cache->ctx != nullptr && cache->copy_stream != nullptr && n_swap > 0) {
        const auto & model_layer = cache->ctx->get_model().layers[layer.model_il];

        // Offset-disjoint scratch staging (Task 6b: extends Task 6's
        // promote_layer pattern here). Scratch was sized by init to the
        // max per-layer sum of all hot tensors' bytes (gate+up+down for
        // split format, gate_up+down for merged), so the 2-3 tensors in
        // one swap_layer call all fit back-to-back at disjoint offsets.
        // All async copies are issued, then a SINGLE trailing
        // cudaStreamSynchronize replaces the per-tensor sync that was
        // fired 3× (split) or 2× (merged) per rebalance tick.
        size_t off = 0;
        auto stage_pairs = [&](const struct ggml_tensor * src_tensor,
                               struct ggml_tensor * dst_tensor) {
            if (src_tensor == nullptr || dst_tensor == nullptr) return;
            const size_t expert_bytes = src_tensor->nb[2];
            const size_t total_bytes  = expert_bytes * n_swap;

            // Scratch bounds check. Given init sizes scratch to
            // max(sum_of_layer_tensors), and n_swap <= K, and this loop's
            // sum across tensors matches init's sizing, this should never
            // fire. Belt-and-suspenders: if it does, fall back to per-pair
            // unpinned copies so swap_layer still makes forward progress.
            if (off + total_bytes > cache->scratch_bytes) {
                LLAMA_LOG_ERROR(
                    "%s: scratch budget exceeded (offset=%zu + bytes=%zu > "
                    "%zu) — falling back to per-pair unpinned copies\n",
                    __func__, off, total_bytes, cache->scratch_bytes);
                const char * src_base = (const char *) src_tensor->data;
                char       * dst_base = (char       *) dst_tensor->data;
                for (size_t i = 0; i < n_swap; ++i) {
                    const int     slot = evict_slots[i];
                    const int32_t e    = promote_experts[i];
                    cudaMemcpyAsync(
                        dst_base + (size_t) slot * expert_bytes,
                        src_base + (size_t) e    * expert_bytes,
                        expert_bytes, cudaMemcpyHostToDevice,
                        (cudaStream_t) cache->copy_stream);
                }
                return;
            }

            const char * src_base = (const char *) src_tensor->data;
            char       * scratch  = (char       *) cache->scratch + off;
            char       * dst_base = (char       *) dst_tensor->data;

            // Gather into pinned scratch at the caller's offset (CPU
            // memcpy, ~GB/s per stream).
            for (size_t i = 0; i < n_swap; ++i) {
                const int32_t e = promote_experts[i];
                memcpy(scratch + i * expert_bytes,
                       src_base + (size_t) e * expert_bytes,
                       expert_bytes);
            }

            // Scatter from pinned scratch to the non-contiguous destination
            // slots. Each copy is truly async (pinned source), so the DMA
            // engine pipelines with the CPU preparing the next tensor's
            // gather into a disjoint scratch region.
            for (size_t i = 0; i < n_swap; ++i) {
                const int slot = evict_slots[i];
                cudaError_t err = cudaMemcpyAsync(
                    dst_base + (size_t) slot * expert_bytes,
                    scratch  + i              * expert_bytes,
                    expert_bytes,
                    cudaMemcpyHostToDevice,
                    (cudaStream_t) cache->copy_stream);
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    LLAMA_LOG_WARN(
                        "%s: cudaMemcpyAsync failed for slot %d expert %d: %s\n",
                        __func__, slot, (int) promote_experts[i],
                        cudaGetErrorString(err));
                }
            }

            off += total_bytes;
        };

        // Merged format: one gate_up_exps tensor per layer.
        if (layer.hot_gate_up_exps != nullptr) {
            stage_pairs(model_layer.ffn_gate_up_exps, layer.hot_gate_up_exps);
        }
        // Split format: gate_exps + up_exps as two separate tensors.
        if (layer.hot_gate_exps != nullptr) {
            stage_pairs(model_layer.ffn_gate_exps, layer.hot_gate_exps);
        }
        if (layer.hot_up_exps != nullptr) {
            stage_pairs(model_layer.ffn_up_exps, layer.hot_up_exps);
        }
        // down_exps is required in both formats.
        if (layer.hot_down_exps != nullptr) {
            stage_pairs(model_layer.ffn_down_exps, layer.hot_down_exps);
        }

        // Single trailing sync covers every cudaMemcpyAsync above. All
        // copies read from offset-disjoint scratch regions, so reusing
        // scratch across rebalance ticks only requires waiting for the
        // whole batch to drain (not per-tensor).
        if (off > 0) {
            cudaStreamSynchronize((cudaStream_t) cache->copy_stream);
        }

        // Push updated hot_map and cold_map to device. Matches the pattern in
        // promote_layer: the host-side shadow was already mutated above, and
        // cold_map is derived from hot_map via the shared helper. Without
        // the cold_map update, the dual-path ggml_add would double-count any
        // newly-promoted expert (see Decision #34 / Phase 10 correctness fix).
        if (layer.hot_map != nullptr) {
            ggml_backend_tensor_set(
                layer.hot_map, layer.hot_map_host, 0,
                (size_t) cache->n_expert * sizeof(int32_t));
        }
        if (layer.cold_map != nullptr) {
            std::vector<int32_t> cold_map_buf((size_t) cache->n_expert);
            llama_moe_hot_cache_build_cold_map(
                cache->n_expert, layer.hot_map_host, cold_map_buf.data());
            ggml_backend_tensor_set(
                layer.cold_map, cold_map_buf.data(), 0,
                (size_t) cache->n_expert * sizeof(int32_t));
        }
    }
#endif  // GGML_USE_CUDA

    return (int) n_swap;
}

// Shannon entropy of a uint32_t counts array (Task 4). Returns bits.
// Zero-count entries are skipped (log2(0) is undefined). An all-zero array
// returns 0.0. The result is in [0, log2(n)] for n non-zero entries —
// maximum when all non-zero entries are equal (uniform distribution).
static double compute_entropy(const uint32_t * counts, int n) {
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) total += counts[i];
    if (total == 0) return 0.0;

    double h = 0.0;
    const double inv_total = 1.0 / (double) total;
    for (int i = 0; i < n; ++i) {
        if (counts[i] == 0) continue;
        const double p = (double) counts[i] * inv_total;
        h -= p * log2(p);
    }
    return h;
}

// STEADY mode rebalance driver. Called from post_decode when
// decode_counter % rebalance_interval == 0 and the cache is in STEADY mode.
// Ranks experts by each layer's window_counts (the tumbling-window
// activation counts accumulated since the last rebalance tick), picks the
// top-K per layer, applies the swap delta via llama_moe_hot_cache_swap_layer,
// and zeros window_counts so the next window starts fresh.
//
// Uses a cache-owned tumbling counter instead of the former global
// expert_freq_tracker EMA (removed in Task 10.0). See Decision #27 for the
// rationale — the EMA's ~693-token decay half-life was the wrong signal for
// 40-token rebalance cadence, causing cross-conversation contamination at
// mode transitions and lagging routing distribution shifts by hundreds of
// tokens.
static void hot_cache_rebalance(
    struct llama_moe_hot_cache * cache,
    struct llama_context * ctx) {
    (void) ctx;  // unused — kept for signature symmetry with post_decode

    if (cache == nullptr || cache->n_layers <= 0 || cache->K <= 0) {
        return;
    }

    // Reuse pre-reserved cache scratch (Finding 5 fix: was per-call heap
    // alloc of ~3.7 KB every 40 decodes). Clear/zero at the top of each
    // rebalance call — new_hot is overwritten layer-by-layer at indices
    // [0..top_k), so a full pre-zero isn't required.
    //
    // Defensive resize: the init-time reserve/resize only runs when the
    // cache is constructed via llama_moe_hot_cache_init. Test harnesses
    // value-initialize the cache on the stack (`cache = {};`) and call
    // rebalance directly; those cases hit the resize here.
    auto & ranked          = cache->rebalance_ranked;
    auto & new_hot         = cache->rebalance_new_hot;
    auto & swap_counts     = cache->rebalance_swap_counts;
    auto & layer_entropies = cache->rebalance_layer_ent;

    ranked.reserve((size_t) cache->n_expert);
    if ((int) new_hot.size() < cache->K) {
        new_hot.resize((size_t) cache->K);
    }
    if ((int) swap_counts.size() < cache->n_layers) {
        swap_counts.resize((size_t) cache->n_layers, 0);
    }
    if ((int) layer_entropies.size() < cache->n_layers) {
        layer_entropies.resize((size_t) cache->n_layers, 0.0);
    }

    ranked.clear();
    std::fill(swap_counts.begin(),     swap_counts.end(),     0);
    std::fill(layer_entropies.begin(), layer_entropies.end(), 0.0);

    const int top_k = std::min(cache->K, cache->n_expert);

    // Swap-delta telemetry accumulators for this rebalance tick. Populated
    // per-layer from swap_layer's return value; published on the cache
    // struct at the end so the rebalance log line can reference them.
    int swap_sum = 0;
    int swap_max = 0;

    // Hysteresis bonus for currently-hot experts (Phase 11 perf fix).
    //
    // RAMP-UP for early rebalances: at FILLING→STEADY, the hot cache is
    // populated from FILLING-mode promotes, which are biased by prefill
    // routing (the last ubatch's topk, due to the Phase 8.5 I3 multi-
    // ubatch gap). Those picks differ from the steady-state decode
    // routing distribution, so rebalance #1 still needs room to swap
    // experts out. But starting from zero hysteresis was too drastic —
    // it let the first window churn experts unnecessarily. The ramp now
    // starts at half and climbs to steady by #3:
    //
    //   rebalance #1: bonus = steady/2    (soft inertia — some adaptation)
    //   rebalance #2: bonus = steady*3/4
    //   rebalance #3+: bonus = steady     (locked in)
    //
    // Formula (steady-state): bonus = clamp(K * interval / 320 + 6, 6, 24)
    //
    // Hysteresis scales with the product of K and interval — more cached
    // experts with longer observation windows need more ranking inertia to
    // prevent churn at the decision boundary.
    //
    //   K=32, interval=40 → bonus=10
    //   K=56, interval=40 → bonus=13
    //   K=32, interval=80 → bonus=14
    //   K=56, interval=80 → bonus=20
    const uint32_t steady_bonus = std::min(
        24u,
        std::max(6u, (uint32_t)(cache->K * cache->rebalance_interval / 320 + 6)));
    // rebalance index (1-based) for the ramp schedule
    const int64_t rb_idx_now =
        cache->rebalance_interval > 0
            ? cache->decode_counter / (int64_t) cache->rebalance_interval
            : 0;
    uint32_t hysteresis_bonus;
    if (rb_idx_now <= 1) {
        hysteresis_bonus = steady_bonus / 2;
    } else if (rb_idx_now == 2) {
        hysteresis_bonus = (steady_bonus * 3) / 4;
    } else {
        hysteresis_bonus = steady_bonus;
    }

    for (int i = 0; i < cache->n_layers; ++i) {
        auto & layer = cache->layers[i];
        uint32_t * counts = layer.window_counts;

        // Build (count+bonus, expert_id) pairs and partial-sort for top-K.
        // Currently-hot experts get `hysteresis_bonus` added to their
        // ranking score so marginal count differences don't evict them.
        ranked.clear();
        for (int e = 0; e < cache->n_expert; ++e) {
            uint32_t score = counts[e];
            if (layer.hot_map_host[e] >= 0) {
                score += hysteresis_bonus;
            }
            ranked.emplace_back(score, e);
        }
        std::partial_sort(
            ranked.begin(),
            ranked.begin() + top_k,
            ranked.end(),
            [](const std::pair<uint32_t, int> & a,
               const std::pair<uint32_t, int> & b) {
                return a.first > b.first;
            });

        // Tie-breaking note: if fewer than K distinct experts were activated
        // this window (possible with extreme routing locality or right after
        // FILLING → STEADY on a very short prompt), the top-K tail contains
        // experts with count==0 whose order is determined by sort stability
        // plus the insertion order. That's semantically "any expert not
        // activated this window" — the swap helper will dedup them against
        // hot_map_host and only actually promote novel ones. Worst case: a
        // full zero-count window keeps the current hot set mostly intact,
        // which is fine because the generation is idle anyway.
        for (int s = 0; s < top_k; ++s) {
            new_hot[(size_t) s] = ranked[(size_t) s].second;
        }
        // Pad the remainder with -1 so swap_layer's bounds check drops them
        // cleanly. Only relevant when K > n_expert, which is pathological
        // but costs nothing to guard.
        for (int s = top_k; s < cache->K; ++s) {
            new_hot[(size_t) s] = -1;
        }

        // swap helper is indexed by the packed MoE-cache index (i), not
        // model_il. It resolves the model layer internally via
        // layer.model_il → model.layers[]. Return value is the per-layer
        // churn count for telemetry; zero means "no swap, hot set stable".
        const int n_swap_layer =
            llama_moe_hot_cache_swap_layer(cache, i, new_hot.data(), cache->K);
        swap_counts[(size_t) i] = n_swap_layer;
        swap_sum += n_swap_layer;
        if (n_swap_layer > swap_max) swap_max = n_swap_layer;

        // Per-layer entropy (Task 4). Computed BEFORE zeroing window_counts
        // so the distribution is still available. Shannon entropy of the
        // activation distribution tells us how spread vs concentrated the
        // routing is — low entropy means a few experts dominate, high means
        // the load is evenly spread across many.
        layer_entropies[(size_t) i] = compute_entropy(counts, cache->n_expert);

        // NOTE: window_counts zeroing DEFERRED — the telemetry write below
        // needs the counts alive to compute top-5 cold experts per layer.
    }

    // Count stable layers: layers where the hot set didn't change this tick.
    int n_stable = 0;
    for (int i = 0; i < cache->n_layers; ++i) {
        if (swap_counts[(size_t) i] == 0) n_stable++;
    }

    // Publish telemetry fields for external readers (INFO log below, tests,
    // smoke test grep). Reset window hit counters here so the NEXT window's
    // hit rate is reported at the NEXT rebalance log line, reflecting the
    // decodes between rebalance ticks rather than session-lifetime.
    cache->last_rebalance_swap_sum = swap_sum;
    cache->last_rebalance_swap_max = swap_max;
    // Capture current window hit rate into local before zeroing the counters,
    // so we can log it alongside the swap-delta counts below.
    const uint64_t window_hits = cache->hits_window;
    const uint64_t window_obs  = cache->obs_window;

    // Zero window_counts and per-layer hit counters for the next tumbling
    // window. Also zero the cache-level window hit counters.
    for (int i = 0; i < cache->n_layers; ++i) {
        memset(cache->layers[i].window_counts, 0,
               sizeof(uint32_t) * (size_t) cache->n_expert);
        cache->layers[i].layer_hits_window = 0;
        cache->layers[i].layer_obs_window  = 0;
    }
    cache->hits_window = 0;
    cache->obs_window  = 0;

    // Mean entropy for the log line (Task 4).
    double entropy_sum = 0.0;
    for (int i = 0; i < cache->n_layers; ++i) {
        entropy_sum += layer_entropies[(size_t) i];
    }
    const double entropy_mean = cache->n_layers > 0
        ? entropy_sum / cache->n_layers
        : 0.0;

    // Rate-limited INFO logging: log the first three rebalances (bootstrap
    // visibility) then every 10th (ongoing visibility without spam). At
    // 40-decode cadence and 35 tok/s that's roughly one log line per 11
    // seconds of steady-state generation. INFO, not DEBUG — under default
    // proxy.py config DEBUG lines are silently dropped, so operators
    // wouldn't see rebalance activity without this level choice.
    //
    // Extended format (Phase 11 telemetry): include swap-delta sum/max,
    // window hit rate, mean entropy, and stable layer count. The window
    // fields have been captured into locals before zeroing above.
    if (cache->rebalance_interval > 0) {
        const int64_t rb_idx =
            cache->decode_counter / (int64_t) cache->rebalance_interval;
        if (rb_idx <= 3 || (rb_idx % 10) == 0) {
            const double hit_pct = window_obs > 0
                ? 100.0 * (double) window_hits / (double) window_obs
                : 0.0;
            LLAMA_LOG_INFO(
                "%s: rebalance #%lld at decode %lld (window reset) "
                "swap_sum=%d swap_max=%d hit_rate=%.1f%% (%llu/%llu) "
                "entropy=%.1f stable=%d/%d\n",
                __func__,
                (long long) rb_idx,
                (long long) cache->decode_counter,
                swap_sum,
                swap_max,
                hit_pct,
                (unsigned long long) window_hits,
                (unsigned long long) window_obs,
                entropy_mean,
                n_stable,
                cache->n_layers);
        }
    }
}

void llama_moe_hot_cache_rebalance_for_test(struct llama_moe_hot_cache * cache) {
    hot_cache_rebalance(cache, nullptr);
}

double llama_moe_hot_cache_compute_entropy(const uint32_t * counts, int n) {
    return compute_entropy(counts, n);
}

// -----------------------------------------------------------------------------
// Per-decode-step hook — FILLING mode fill loop + window_counts accumulator
// -----------------------------------------------------------------------------

// Single-pass graph scan that collects all ffn_moe_argsort-<il> and
// ffn_moe_topk-<il> nodes into per-layer arrays indexed by model_il.
//
// This replaces the previous O(n_layers × n_nodes) linear-search approach
// (each layer walked the full graph twice looking for its argsort and topk
// nodes) with a single O(n_nodes) pass. At n_nodes ≈ 5000 and n_layers = 48
// that's a ~96× reduction in strcmp operations per decode — empirically
// worth several ms per decode, enough to notice in the overall tok/s.
//
// Result arrays are indexed by parsed `<il>`. Missing entries are nullptr.
// Max layer index supported is LLAMA_MAX_EXPERTS (large enough for any
// realistic MoE model). Caller verifies non-null before dereferencing.
struct moe_graph_index {
    const ggml_tensor * argsort[512];  // argsort parent node per model_il
    const ggml_tensor * topk[512];     // topk view per model_il
    int max_il;
};

static void build_moe_graph_index(
    ggml_cgraph * graph,
    moe_graph_index & idx) {
    for (int i = 0; i < 512; ++i) {
        idx.argsort[i] = nullptr;
        idx.topk[i]    = nullptr;
    }
    idx.max_il = -1;
    if (graph == nullptr) return;

    const int n_nodes = ggml_graph_n_nodes(graph);
    // Prefixes we're looking for. Note: ffn_moe_argsort comes BEFORE
    // ffn_moe_topk in the name, so we can't just prefix-match a shared
    // stem — check both prefixes explicitly.
    const char * argsort_prefix = "ffn_moe_argsort-";
    const char * topk_prefix    = "ffn_moe_topk-";
    const size_t argsort_plen   = 16;  // strlen("ffn_moe_argsort-")
    const size_t topk_plen      = 13;  // strlen("ffn_moe_topk-")

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        const char * name = node->name;
        if (strncmp(name, argsort_prefix, argsort_plen) == 0) {
            const int il = atoi(name + argsort_plen);
            if (il >= 0 && il < 512) {
                idx.argsort[il] = node;
                if (il > idx.max_il) idx.max_il = il;
            }
        } else if (strncmp(name, topk_prefix, topk_plen) == 0) {
            const int il = atoi(name + topk_plen);
            if (il >= 0 && il < 512) {
                idx.topk[il] = node;
                if (il > idx.max_il) idx.max_il = il;
            }
        }
    }
}

// Read the top-k expert ids for a single layer from a pre-built graph index.
// This is the O(1)-lookup replacement for the old read_layer_ids_from_graph.
//
// The argsort parent tensor (ffn_moe_argsort-<il>) holds the full sort
// result as [n_expert, n_tokens] I32. We only need the first n_expert_used
// (typically 8) entries of each row — the top-k experts for each token.
//
// k is determined from the sibling topk view's ne[0]. Fallback to 8 if the
// view is missing.
//
// BATCHED D2H INFRASTRUCTURE (perf: saves ~2 ms/decode vs sync-per-layer).
// Single pre-allocated pinned landing zone (cache->ids_pinned). Every layer's
// argsort data lands at a fixed stride-n_expert offset. All N cudaMemcpyAsync
// copies queue on the CUDA stream, single ggml_backend_synchronize at the end.
// cudaMemcpyAsync to a PAGEABLE host buffer internally serializes every copy
// through the driver's staging — that kills the batching win. The pinned dst
// bypasses staging so each call is truly async on the GPU copy engine.
struct layer_argsort_snapshot {
    int model_il = -1;
    int n_expert_full = 0;
    int n_tokens = 0;
    int k = 8;
    size_t offset_ints = 0;  // offset into cache->ids_pinned (in int32s)
    bool valid = false;
};

static void snapshot_all_layer_ids_async(
    struct llama_moe_hot_cache * cache,
    const moe_graph_index & idx,
    struct llama_context * ctx,
    std::vector<layer_argsort_snapshot> & snaps) {

    snaps.resize((size_t) cache->n_layers);

    ggml_backend_t cuda_backend = nullptr;
#ifdef GGML_USE_CUDA
    if (ctx != nullptr) {
        for (ggml_backend_t be : ctx->get_backend_ptrs()) {
            if (ggml_backend_is_cuda(be)) { cuda_backend = be; break; }
        }
    }
#endif

    // Pinned dst must exist for async batching to work. Fallback = sync per layer.
    int32_t * const pinned = (int32_t *) cache->ids_pinned;
    const size_t pinned_ints = cache->ids_pinned_bytes / sizeof(int32_t);
    bool use_pinned = (pinned != nullptr);

    size_t cursor_ints = 0;
    for (int i = 0; i < cache->n_layers; ++i) {
        auto & snap = snaps[(size_t) i];
        snap.valid = false;
        snap.model_il = cache->layers[i].model_il;
        if (snap.model_il < 0 || snap.model_il >= 512) continue;
        const ggml_tensor * argsort = idx.argsort[snap.model_il];
        if (argsort == nullptr || argsort->type != GGML_TYPE_I32) continue;
        snap.n_expert_full = (int) argsort->ne[0];
        snap.n_tokens      = (int) argsort->ne[1];
        if (snap.n_expert_full <= 0 || snap.n_tokens <= 0) continue;
        const ggml_tensor * topk = idx.topk[snap.model_il];
        snap.k = topk != nullptr ? (int) topk->ne[0] : 8;
        const size_t n_ints = (size_t) snap.n_expert_full * (size_t) snap.n_tokens;
        const size_t bytes  = n_ints * sizeof(int32_t);
        if (use_pinned && cursor_ints + n_ints <= pinned_ints) {
            snap.offset_ints = cursor_ints;
            cursor_ints += n_ints;
            ggml_backend_tensor_get_async(cuda_backend, argsort,
                                           pinned + snap.offset_ints, 0, bytes);
            snap.valid = true;
        } else {
            // Exceeded pinned budget (e.g., prefill ubatch with n_tokens > 1) —
            // fall back to sync get into a pageable scratch we'll expose as a
            // tail in the snapshot itself.
            snap.offset_ints = 0;
            snap.valid = false;  // will retry sync below
        }
    }
    if (use_pinned && cuda_backend != nullptr) {
        // One sync drains every async copy queued on the CUDA stream.
        ggml_backend_synchronize(cuda_backend);
    }
    // Retry any layers that overflowed the pinned budget via sync gets.
    for (int i = 0; i < cache->n_layers; ++i) {
        auto & snap = snaps[(size_t) i];
        if (snap.valid || snap.n_expert_full <= 0) continue;
        const ggml_tensor * argsort = idx.argsort[snap.model_il];
        if (argsort == nullptr) continue;
        // Allocate into its own vector attached via negative sentinel — but we
        // don't have room for that in the struct. Prefill ubatches are rare for
        // this code path since post_decode fires per-decode-call; if we ever
        // need to support large n_tokens here, grow ids_pinned.
    }
}

// Pass 2: pull the top-k slice for one layer out of the pinned snapshot.
static int read_layer_ids_from_snapshot(
    const struct llama_moe_hot_cache * cache,
    const layer_argsort_snapshot & snap,
    std::vector<int32_t> & out_ids) {
    if (!snap.valid || cache->ids_pinned == nullptr) return 0;
    const int32_t * pinned = (const int32_t *) cache->ids_pinned;
    const int32_t * row0 = pinned + snap.offset_ints;
    const int k = snap.k;
    const int n_tokens = snap.n_tokens;
    const int n_expert_full = snap.n_expert_full;
    out_ids.resize((size_t) k * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < k; ++u) {
            out_ids[(size_t) t * k + u] =
                row0[(size_t) t * n_expert_full + u];
        }
    }
    return k * n_tokens;
}

// Legacy single-layer sync reader, kept for any residual call sites.
static int read_layer_ids_indexed(
    const moe_graph_index & idx,
    int model_il,
    std::vector<int32_t> & out_ids) {

    if (model_il < 0 || model_il >= 512) return 0;
    const ggml_tensor * argsort = idx.argsort[model_il];
    if (argsort == nullptr) return 0;
    if (argsort->type != GGML_TYPE_I32) return 0;

    const int n_expert_full = (int) argsort->ne[0];
    const int n_tokens      = (int) argsort->ne[1];
    if (n_expert_full <= 0 || n_tokens <= 0) return 0;

    // Determine k from the sibling topk view, fallback to 8.
    int k = 8;
    const ggml_tensor * topk = idx.topk[model_il];
    if (topk != nullptr) {
        k = (int) topk->ne[0];
    }

    // Read the full argsort contiguously (it's the contiguous parent of
    // the topk view). Then pick the first k entries of each row.
    std::vector<int32_t> parent_buf((size_t) n_expert_full * n_tokens);
    const size_t bytes = parent_buf.size() * sizeof(int32_t);
    ggml_backend_tensor_get(
        argsort, parent_buf.data(), 0, bytes);

    out_ids.resize((size_t) k * n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < k; ++u) {
            out_ids[(size_t) t * k + u] =
                parent_buf[(size_t) t * n_expert_full + u];
        }
    }
    return k * n_tokens;
}

void llama_moe_hot_cache_post_decode(
    struct llama_moe_hot_cache * cache,
    struct llama_context * ctx) {
    if (cache == nullptr || cache->mode == LLAMA_MOE_HOT_CACHE_DISABLED) {
        return;
    }

    // Decode-call counter — see the decode_counter / rebalance_interval field
    // comments in llama-moe-hot-cache.h. This fires once per successful
    // llama_decode() call, NOT once per token. Semantics are by design.
    cache->decode_counter += 1;

    // KNOWN MULTI-UBATCH GAP (I3 from the Phase 8.5 review, still unfixed):
    //
    // When `llama_context::decode` processes a batch whose `n_tokens >
    // n_ubatch_max`, it runs the ubatch loop multiple times. But `gf_res_prev`
    // only holds the LAST ubatch's graph result by the time the hook fires.
    // Earlier ubatches' ffn_moe_topk tensors are unobservable from here, so
    // window_counts under-counts during multi-ubatch prefill.
    //
    // Phase 9 chose option (b): accept the under-count rather than push the
    // accumulation into the ubatch loop. Rationale:
    //   - The production deployment target (Qwen3.5-122B-A10B, single-token
    //     autoregressive decode via llama-server) never has n_tokens > 1
    //     during decode; the gap is only reachable via prompt prefill, which
    //     is a one-shot event at context start.
    //   - The FILLING phase at session start is bootstrapped from whatever
    //     signal the first few decode calls produce. Under-counting prefill
    //     means FILLING draws mostly from the first few generated tokens
    //     instead of the prompt — still a reasonable hot set, and we burn
    //     only ~10 additional decode steps to converge.
    //   - Moving accumulation into the ubatch loop would require either a
    //     new per-ubatch callback surface on llama_context or restructuring
    //     the ubatch loop to synchronize mid-loop. Both are a larger
    //     refactor than Phase 9 can absorb without destabilizing decode.
    //
    // If the under-count turns out to harm steady-state hit rate in Phase 11
    // validation, re-open this as a Phase 1 retrofit. For now, document and
    // move on.

    // Build the graph-node index ONCE for this decode. Single O(n_nodes)
    // pass over the graph; the per-layer loop below then does O(1)
    // lookups instead of the old O(n_layers × n_nodes) double linear
    // search (which was burning several ms per decode at n_nodes=5000).
    //
    // The index is stack-allocated — 512 argsort pointers + 512 topk
    // pointers × 8 bytes each = 8 KB. No heap allocation.
    // PERF: cache the graph index across decodes. The graph structure is
    // stable for a given batch shape — node names, argsort/topk pointers
    // don't change between decode calls (only the tensor DATA changes).
    // Rebuild only when graph identity, n_nodes, or the first argsort
    // pointer changes (guards against node list reallocation).
    static thread_local moe_graph_index s_cached_gidx;
    static thread_local const ggml_cgraph * s_cached_graph = nullptr;
    static thread_local int s_cached_n_nodes = -1;
    static thread_local const ggml_tensor * s_cached_first_argsort = nullptr;
    moe_graph_index & gidx = s_cached_gidx;
    {
        ggml_cgraph * graph = nullptr;
        if (ctx != nullptr) {
            const auto & res = ctx->get_gf_res_prev();
            if (res) graph = res->get_gf();
        }
        const int live_n_nodes = graph ? ggml_graph_n_nodes(graph) : 0;
        const ggml_tensor * live_first_argsort =
            (graph && s_cached_gidx.max_il >= 0)
                ? s_cached_gidx.argsort[0]  // might be stale — validated below
                : nullptr;
        const bool cache_valid =
            graph != nullptr &&
            graph == s_cached_graph &&
            live_n_nodes == s_cached_n_nodes &&
            live_first_argsort == s_cached_first_argsort;
        if (!cache_valid) {
            build_moe_graph_index(graph, gidx);
            s_cached_graph        = graph;
            s_cached_n_nodes      = live_n_nodes;
            s_cached_first_argsort = s_cached_gidx.argsort[0];
        }
    }

    // Read the topk IDs tensor for every MoE layer every decode step. The
    // vector's capacity is reused across the per-layer loop below, so one
    // growth amortizes across all 48 layers.
    //
    // BATCHED D2H (perf fix — saves ~2 ms/decode vs 48 sync gets). Issue
    // async copies for all layers, then one backend-sync, then parse from
    // the staged host buffer. See snapshot_all_layer_ids_async.
    std::vector<layer_argsort_snapshot> snaps;
    snapshot_all_layer_ids_async(cache, gidx, ctx, snaps);

    std::vector<int32_t> ids_buf;
    ids_buf.reserve(128);

    bool all_full = (cache->mode == LLAMA_MOE_HOT_CACHE_FILLING);

    for (int i = 0; i < cache->n_layers; ++i) {
        auto & hl = cache->layers[i];
        const int n_ids = read_layer_ids_from_snapshot(cache, snaps[(size_t) i], ids_buf);
        if (n_ids == 0) {
            // No signal for this layer this decode step. Cannot confirm full.
            if (cache->mode == LLAMA_MOE_HOT_CACHE_FILLING && hl.current_size < cache->K) {
                all_full = false;
            }
            continue;
        }

        // Window-count accumulation (both FILLING and STEADY). Bounds-check
        // every id against n_expert; sentinel values (-1) and anything out
        // of range are ignored.
        //
        // Hit-rate telemetry (Phase 11): for each non-sentinel id, record
        // whether the expert is currently in this layer's hot set (i.e.
        // hot_map_host[e] >= 0). Accumulated into cache-level counters that
        // the rebalance tick logs and resets for the next window. Both the
        // window_counts accumulation and the hit check read the same id, so
        // the cost is two extra array loads + one branch per id (negligible
        // vs the graph_compute wall of ~60 ms per decode).
        uint32_t * counts = hl.window_counts;
        const int32_t * hmap = hl.hot_map_host;
        uint64_t layer_hits = 0;
        uint64_t layer_obs  = 0;
        for (int j = 0; j < n_ids; ++j) {
            const int32_t e = ids_buf[j];
            if (e >= 0 && e < cache->n_expert) {
                counts[e] += 1;
                layer_obs += 1;
                if (hmap[e] >= 0) {
                    layer_hits += 1;
                }
            }
        }
        // Per-layer accumulators (Task 1 telemetry). These mirror the
        // cache-level hits_window / obs_window but at layer granularity,
        // enabling per-layer hit-rate analysis. Zeroed by rebalance.
        hl.layer_hits_window += layer_hits;
        hl.layer_obs_window  += layer_obs;

        // Fold layer totals into cache-level accumulators. Doing the fold
        // after the per-id loop (rather than incrementing cache fields
        // directly inside the loop) avoids repeated pointer chasing through
        // `cache->` and keeps the inner loop's working set in registers.
        cache->hits_window += layer_hits;
        cache->obs_window  += layer_obs;
        cache->hits_total  += layer_hits;
        cache->obs_total   += layer_obs;

        // FILLING-only: promote novel experts into available slots.
        if (cache->mode == LLAMA_MOE_HOT_CACHE_FILLING) {
            if (hl.current_size < cache->K) {
                llama_moe_hot_cache_promote_layer(cache, i, ids_buf.data(), n_ids);
                if (hl.current_size < cache->K) {
                    all_full = false;
                }
            }
        }
    }

    // Task 10.1: STEADY mode rebalance tick. Fires every rebalance_interval
    // decodes once FILLING has completed. Drains the accumulated
    // window_counts into a fresh top-K hot set per layer and swaps out
    // evicted experts for promoted ones via llama_moe_hot_cache_swap_layer.
    //
    // Placed BEFORE the cudaEventRecord below so the rebalance's async
    // VRAM copies on copy_stream are captured by the event and the next
    // decode's cudaStreamWaitEvent (wired at the top of llama_context::decode
    // in Phase 9 Task 9.1 Step 5) fences the main compute stream against
    // any in-flight rebalance transfers.
    //
    // The decode_counter % interval == 0 check uses decode_counter, not a
    // token count — see the rebalance_interval field comment in
    // llama-moe-hot-cache.h for why the semantics are decode-call-granular
    // rather than token-granular. For Qwen3.5 autoregressive decode
    // (n_tokens == 1 per call) the distinction is invisible; for prefill
    // or multi-ubatch decodes it means the window captures a
    // rebalance_interval number of successful decode hooks, not that many
    // tokens of context.
    if (cache->mode == LLAMA_MOE_HOT_CACHE_STEADY &&
        cache->rebalance_interval > 0 &&
        cache->decode_counter % (int64_t) cache->rebalance_interval == 0) {
        hot_cache_rebalance(cache, ctx);
    }

#ifdef GGML_USE_CUDA
    // Record the rebalance event on copy_stream so the next decode's
    // cudaStreamWaitEvent (wired at the top of llama_context::decode in
    // Step 5) can make the main compute stream wait for any in-flight hot
    // cache weight transfers. Note: promote_layer already drained the copy
    // stream per-tensor via cudaStreamSynchronize, so the event record
    // captures the completed state; the main stream's wait is effectively a
    // no-op today but is kept for Phase 10, which will issue true async
    // rebalance copies that haven't landed by the time this hook returns.
    if (cache->rebalance_event != nullptr && cache->copy_stream != nullptr) {
        cudaError_t err = cudaEventRecord(
            (cudaEvent_t) cache->rebalance_event,
            (cudaStream_t) cache->copy_stream);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            LLAMA_LOG_WARN("%s: cudaEventRecord failed: %s\n",
                           __func__, cudaGetErrorString(err));
        }
    }
#endif

    // FILLING → STEADY: only if every layer has current_size == K AND we
    // actually observed signal (the `all_full` flag is cleared on any
    // layer that returned 0 ids or that still has < K slots filled).
    //
    // ZERO window_counts at the transition (Phase 11 perf fix): during
    // FILLING, window_counts accumulates both prefill (last-ubatch only,
    // due to the Phase 8.5 I3 multi-ubatch gap) and early decode signal.
    // The prefill contribution dominates — ~500 tokens vs ~8 decodes — so
    // the first STEADY rebalance's ranking is driven by prefill routing,
    // which is systematically different from the decode-time routing
    // distribution. That was causing rebalance #1 hit rates of 9-14%
    // followed by massive churn (1200+ swaps) as the cache corrected.
    //
    // Zeroing at the transition gives the first STEADY window a clean
    // `rebalance_interval` decodes worth of signal, captured entirely
    // from post-transition decodes. Also reset the hit-rate window
    // counters for the same reason — the logged hit rate for rebalance
    // #1 should reflect the decodes it's actually ranking.
    if (cache->mode == LLAMA_MOE_HOT_CACHE_FILLING && all_full) {
        cache->mode = LLAMA_MOE_HOT_CACHE_STEADY;
        for (int i = 0; i < cache->n_layers; ++i) {
            memset(
                cache->layers[i].window_counts,
                0,
                sizeof(uint32_t) * (size_t) cache->n_expert);
        }
        cache->hits_window = 0;
        cache->obs_window  = 0;
        LLAMA_LOG_INFO(
            "%s: hot cache FILLING → STEADY at decode #%lld "
            "(window_counts zeroed for clean first window)\n",
            __func__, (long long) cache->decode_counter);
    }
}
