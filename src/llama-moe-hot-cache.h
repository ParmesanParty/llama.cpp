#pragma once

#include "llama.h"
#include "llama-moe-fused-cold.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

// Forward declarations
struct ggml_tensor;
struct ggml_context;
struct llama_context;
struct llama_model;

enum llama_moe_hot_cache_mode {
    LLAMA_MOE_HOT_CACHE_DISABLED = 0,
    LLAMA_MOE_HOT_CACHE_FILLING  = 1,
    LLAMA_MOE_HOT_CACHE_STEADY   = 2,
};

struct llama_moe_hot_cache_layer {
    // Compact hot weight buffers (GPU-resident). The cache supports BOTH
    // tensor formats that llama.cpp loaders produce for MoE expert weights:
    //
    //   merged: a single ffn_gate_up_exps tensor with the gate and up
    //           projections concatenated along the n_ff dimension. The
    //           cache holds hot_gate_up_exps with the K experts compacted.
    //
    //   split:  separate ffn_gate_exps and ffn_up_exps tensors. The cache
    //           holds hot_gate_exps and hot_up_exps with the K experts
    //           compacted. This is the path used by qwen3moe and by some
    //           qwen35moe quants (e.g. Qwen3.5-122B-A10B Q4_K_L) where the
    //           upstream loader at llama-model.cpp:3072 falls back to the
    //           split format because the GGUF lacks the merged tensor.
    //
    // Exactly one of these formats applies per layer. The choice is made at
    // init time by inspecting which fields the source `llama_layer` has
    // populated, and is recorded by which of the three pointers below is
    // non-null:
    //
    //   merged source → hot_gate_up_exps != nullptr,
    //                   hot_gate_exps == hot_up_exps == nullptr
    //   split  source → hot_gate_up_exps == nullptr,
    //                   hot_gate_exps != nullptr && hot_up_exps != nullptr
    //
    // Memory cost is identical in both formats — the merged tensor is just
    // cat([gate, up], dim=ne[1]). Compute cost differs only at the graph
    // builder level (one MUL_MAT_ID for merged, two for split).
    struct ggml_tensor * hot_gate_up_exps;   // [n_embd, n_ff_exp*2, K] — merged-format only
    struct ggml_tensor * hot_gate_exps;      // [n_embd, n_ff_exp, K]   — split-format only
    struct ggml_tensor * hot_up_exps;        // [n_embd, n_ff_exp, K]   — split-format only
    struct ggml_tensor * hot_down_exps;      // [n_ff_exp, n_embd, K]   — both formats

    // Note: per-expert scale tensors (up_exps_s / down_exps_s) are NOT compacted.
    // The dual-path emits hot and cold MUL_MAT_ID nodes, sums them via ggml_add,
    // then applies the original per-expert scales (indexed by original expert id
    // via selected_experts) to the merged tensor. Sentinel-skipped slots are zero,
    // so applying scales on the merged tensor is equivalent to applying them
    // separately. This saves allocation and transfer work.

    // Remapping tables (GPU-resident, int32)
    // hot_map[e]  = slot ∈ [0,K) if expert e is hot, else -1
    // cold_map[e] = e if expert e is cold, else -1
    struct ggml_tensor * hot_map;
    struct ggml_tensor * cold_map;

    // Host-side shadow copies for fast mutation during fill/rebalance
    int32_t * hot_map_host;    // length n_expert; mirror of hot_map device data
    int32_t * slot_to_expert;  // length K; inverse of hot_map_host for hot slots only

    // Index of this MoE layer within llama_model::layers (NOT the MoE-cache index).
    // Needed because cache->layers[i] is packed (only MoE layers); i is not the
    // same as the model layer index for hybrid dense+MoE architectures. Populated
    // in Task 3.1 during allocation, consumed by Tasks 9.1/10.1 when reading
    // "ffn_moe_topk-<il>" from the graph or indexing model.layers[] for source
    // tensor addresses. For qwen35moe (all layers MoE) model_il == i, but do not
    // rely on that — always read this field.
    int model_il;

    // Tumbling-window expert selection counts for this layer. Length n_expert.
    // Incremented by the post-decode hook every decode step for each expert id
    // seen in this layer's topk tensor. Zeroed by hot_cache_rebalance() after
    // each rebalance tick, so it holds exactly the last `rebalance_interval`
    // tokens worth of per-expert activation counts at the next rebalance.
    //
    // This REPLACES the former global expert_freq_tracker EMA that lived in
    // ggml_backend_sched — Task 10.0 tears that machinery out wholesale because
    // its ~693-token decay half-life was 17x longer than the 40-token rebalance
    // cadence and caused cross-conversation contamination at mode transitions.
    // See Decision #27.
    uint32_t * window_counts;

    // Per-layer hit/observation accumulators. These are the layer-granular
    // equivalents of cache->hits_window / cache->obs_window. Incremented by
    // post_decode alongside the cache-level fold, zeroed by hot_cache_rebalance
    // together with window_counts. Enables per-layer hit-rate telemetry
    // (e.g. identifying layers whose hot set is stale or thrashing).
    uint64_t layer_hits_window;
    uint64_t layer_obs_window;

    // Current fill progress (0..K)
    int current_size;
};

struct llama_moe_hot_cache {
    enum llama_moe_hot_cache_mode mode;

    int K;                    // hot set size per layer
    int n_expert;             // total experts per layer (e.g., 256)
    int n_expert_used;        // experts activated per token (e.g., 8)
    int n_layers;             // number of MoE layers (e.g., 48)

    // Decode-call (not token) cadence between STEADY-mode rebalances. The
    // external API (`--moe-hot-rebalance-interval`, LLAMA_ARG_MOE_HOT_REBALANCE_
    // INTERVAL) uses the word "tokens" because that's operator-intuitive, but
    // internally the rebalance trigger compares against `decode_counter` which
    // increments exactly once per successful llama_decode() call regardless of
    // how many tokens are in the batch. For Qwen3.5 production single-token
    // autoregressive decode, 1 decode call == 1 token and the distinction is
    // invisible. For prefill batches (n_tokens > 1) or multi-ubatch decodes,
    // one call advances the counter by 1 but consumes many tokens of context,
    // which makes the window semantics decode-driven rather than token-driven.
    // This is by design — Phase 9's window_counts accumulator reads activation
    // data from the last ubatch's ffn_moe_topk, which is also decode-call-
    // granular (see the multi-ubatch caveat in llama-context.cpp's hook site).
    int rebalance_interval;   // decode calls between steady-state rebalances (e.g., 40)

    // Monotonic counter of successful llama_decode() calls observed by
    // the post-decode hook. NOT a token count — see the rebalance_interval
    // comment above for the intentional decode-call semantics. Phase 9 reads
    // this counter to fire window-count flushes at `decode_counter %
    // rebalance_interval == 0`.
    int64_t decode_counter;

    // Rebalance diagnostics (Phase 11 telemetry — always on, ~zero cost).
    //
    // hits_window / obs_window reset at every rebalance tick and measure the
    // fraction of expert ids observed by post_decode (after sentinel/range
    // filtering) that landed in a currently-hot slot, across all layers, for
    // the last rebalance_interval decodes. hits_total / obs_total are the
    // lifetime-of-session equivalents and never reset.
    //
    // last_rebalance_swap_sum / last_rebalance_swap_max record the swap-delta
    // totals from the most recent hot_cache_rebalance() call. sum == 0 means
    // the window's top-K exactly matched the current hot set (zero churn).
    // max is the largest per-layer swap count in that rebalance — useful for
    // spotting single outlier layers thrashing while others are stable.
    //
    // All four fields are read in hot_cache_rebalance()'s INFO log and used
    // by external probes (tests, smoke tests). They are NOT surfaced on any
    // public API; extending the public API for this is a future concern.
    uint64_t hits_window;
    uint64_t obs_window;
    uint64_t hits_total;
    uint64_t obs_total;
    int      last_rebalance_swap_sum;
    int      last_rebalance_swap_max;

    struct llama_moe_hot_cache_layer * layers; // length n_layers

    // Rebalance scratch vectors (Finding 5 fix). Pre-reserved at init to
    // avoid per-rebalance heap churn; cleared at the top of each
    // hot_cache_rebalance call. Sizes:
    //   rebalance_ranked:      reserved n_expert  entries of (count, expert_id)
    //   rebalance_new_hot:     resized to K         entries of int32
    //   rebalance_swap_counts: resized to n_layers  entries of int
    //   rebalance_layer_ent:   resized to n_layers  entries of double
    std::vector<std::pair<uint32_t, int>> rebalance_ranked;
    std::vector<int32_t>                  rebalance_new_hot;
    std::vector<int>                      rebalance_swap_counts;
    std::vector<double>                   rebalance_layer_ent;

    // Tensor metadata context + backing VRAM buffer. All per-layer hot tensors
    // live inside this single ggml_context; the backend_buffer owns their data.
    struct ggml_context    * meta_ctx;
    ggml_backend_buffer_t    backend_buffer;

    // CUDA resources (opaque; implementation allocates/frees on the copy stream)
    void * copy_stream;       // cudaStream_t
    void * rebalance_event;   // cudaEvent_t

    // Pinned scratch buffer for staged host→device copies during FILLING /
    // STEADY rebalance (Decision #30 Path B). Allocated once at init via
    // cudaMallocHost so the source address is pinned without having to
    // cudaHostRegister the model's mmap'd weight regions, which hits a
    // ~50 GB per-process driver ceiling on systems with IOMMU-managed
    // pinned memory (production RTX 4090 / CUDA 12.x). The scratch is
    // sized to the largest hot tensor's VRAM footprint so a single copy
    // operation can stage one (layer, tensor) slice at a time:
    //
    //     memcpy(scratch, src + row_offset, bytes);         // CPU→pinned
    //     cudaMemcpyAsync(vram_slot, scratch, ..., stream); // pinned→VRAM
    //
    // The same scratch is reused across all layers/copies; Phase 9/10 copy
    // paths synchronize on `rebalance_event` before the next memcpy, which
    // serializes scratch access naturally.
    void * scratch;           // cudaMallocHost-pinned host memory
    size_t scratch_bytes;

    // PERF: pinned D2H landing zone for post_decode's batched argsort read.
    // Sized to n_layers * n_expert * sizeof(int32_t). Unlocked via cudaMallocHost
    // so cudaMemcpyAsync is truly async (pageable dst would serialize every copy).
    // Laid out in row-major [layer_idx][expert] — each layer's argsort data lands
    // at offset layer_idx * n_expert * sizeof(int32_t).
    void * ids_pinned;
    size_t ids_pinned_bytes;

    // Weak reference back to the owning context (for backend access)
    struct llama_context * ctx;

    // Params for the fused cold kernel custom op. Initialized once at init,
    // passed as userdata to ggml_custom_4d. Must outlive graph compute.
    struct llama_moe_fused_cold_params fused_cold_params;
};

// Lifecycle
struct llama_moe_hot_cache * llama_moe_hot_cache_init(
    struct llama_context * ctx,
    int K,
    int rebalance_interval);

void llama_moe_hot_cache_free(struct llama_moe_hot_cache * cache);

// Graph build accessors (return the per-layer device tensors for use in build_moe_ffn).
// `il` is the MODEL layer index — the same counter passed to build_moe_ffn by the
// graph builder (e.g., qwen35moe's per-layer loop). This is NOT the MoE-cache
// packed index; the accessor internally looks up cache->layers[i] where
// cache->layers[i].model_il == il. Returns nullptr if no MoE cache layer matches
// that model layer (e.g., dense layer in a hybrid architecture) — callers then
// short-circuit the dual-path and fall back to the single-call path.
struct ggml_tensor * llama_moe_hot_cache_get_hot_map(
    const struct llama_moe_hot_cache * cache, int il);
struct ggml_tensor * llama_moe_hot_cache_get_cold_map(
    const struct llama_moe_hot_cache * cache, int il);
// Returns the merged hot_gate_up_exps tensor for the given model layer, or
// nullptr if the layer is split-format (use get_hot_gate + get_hot_up
// instead) or no MoE cache layer matches that model layer.
struct ggml_tensor * llama_moe_hot_cache_get_hot_gate_up(
    const struct llama_moe_hot_cache * cache, int il);

// Returns the split hot_gate_exps tensor for the given model layer, or
// nullptr if the layer is merged-format (use get_hot_gate_up instead) or no
// MoE cache layer matches that model layer.
struct ggml_tensor * llama_moe_hot_cache_get_hot_gate(
    const struct llama_moe_hot_cache * cache, int il);

// Returns the split hot_up_exps tensor for the given model layer, or
// nullptr if the layer is merged-format (use get_hot_gate_up instead) or no
// MoE cache layer matches that model layer.
struct ggml_tensor * llama_moe_hot_cache_get_hot_up(
    const struct llama_moe_hot_cache * cache, int il);

struct ggml_tensor * llama_moe_hot_cache_get_hot_down(
    const struct llama_moe_hot_cache * cache, int il);

// Returns true if the layer has at least one hot expert populated
// (current_size > 0). The dual-path graph emission gates on this because
// the Phase 1 kernel sentinel-skip patches (which zero-fill dst and skip
// negative ids in mm_ids_helper) do NOT cover the all-negative degenerate
// case: when every id in a MUL_MAT_ID ids tensor is -1, mm_ids_helper
// writes nothing to ids_src1/ids_dst and the downstream quantize_mmq_q8_1
// reads uninitialized pool memory → "CUDA error: an illegal memory
// access was encountered". Until Phase 1's sentinel coverage is extended
// to the all-negative case (or until Phase 9 guarantees a mixed ids
// tensor via per-layer promotion), the graph builder must emit only the
// vanilla single-path when this accessor returns false.
//
// Returns false for dense layers in hybrid architectures (find_layer_by_
// model_il returns nullptr) and for MoE layers whose cache has not yet
// been populated by FILLING / STEADY rebalance (current_size == 0).
bool llama_moe_hot_cache_layer_has_hot(
    const struct llama_moe_hot_cache * cache, int il);

// Per-decode-step hook (called by llama_context after each successful decode)
void llama_moe_hot_cache_post_decode(
    struct llama_moe_hot_cache * cache,
    struct llama_context * ctx);

// Promote novel experts from a per-layer ids buffer into the layer's hot
// slots. Updates hot_map_host and slot_to_expert synchronously; issues async
// device copies (via the Decision #30 Path B pinned scratch buffer) when
// GPU buffers are present. Called from llama_moe_hot_cache_post_decode in
// FILLING mode and from unit tests.
//
// `layer_idx` is the packed MoE-cache index (0..n_layers-1), NOT the model
// layer index — the function reaches model.layers[] via the stored
// cache->layers[layer_idx].model_il. See Decision #26 and Decision #28.
//
// `ids` points to a buffer of int32 expert ids (typically read from the
// "ffn_moe_topk-<model_il>" graph tensor); sentinel values (-1) and
// out-of-range ids are ignored. Duplicate ids are deduplicated. The helper
// caps new promotions at `cache->K - layer.current_size`, so it is safe to
// call with n_ids > available slots.
void llama_moe_hot_cache_promote_layer(
    struct llama_moe_hot_cache * cache,
    int layer_idx,
    const int32_t * ids,
    int n_ids);

// Derive a cold-map buffer from a hot-map shadow. cold_map is the complement
// of hot_map: for each expert e in [0, n_expert),
//
//     out[e] = (hot_map_host[e] >= 0) ? -1 : e
//
// i.e., -1 when the expert is currently hot (any valid slot), and the expert
// id itself when it is still cold. Promotion and rebalance both use this to
// rebuild cold_map on the host before a single backend_tensor_set push to
// device. Factored out of the promote/swap call sites so it is exercised by
// a direct unit test without requiring a real CUDA context.
//
// `out` must point to at least `n_expert` int32 slots. `hot_map_host` must
// point to at least `n_expert` int32 slots holding the current hot-map
// shadow.
void llama_moe_hot_cache_build_cold_map(
    int n_expert,
    const int32_t * hot_map_host,
    int32_t * out);

// STEADY rebalance swap helper. Given a new desired hot set of length
// `new_hot_len`, diff against the layer's current hot set (via
// slot_to_expert / hot_map_host) and perform in-place slot reuse:
//
//   - Evict experts that fell out of the new set (hot_map_host[e] = -1,
//     slot_to_expert[slot] = -1).
//   - Promote experts that entered the new set into the freed slots,
//     staging expert weights into VRAM via cudaMemcpyAsync.
//   - Push updated hot_map + cold_map to device so the next decode's
//     dual-path graph reads the post-swap mapping.
//
// Synchronous on host-side state (hot_map_host and slot_to_expert are
// updated before any CUDA work). Async on the copy stream for VRAM copies.
// Rollback on CUDA failure is NOT implemented because the mmap'd source
// data does not move — a failed copy leaves stale VRAM that will be
// overwritten on the next successful rebalance.
//
// `layer_idx` is the packed MoE-cache index (0..n_layers-1), NOT the model
// layer index — the function reaches model.layers[] via the stored
// cache->layers[layer_idx].model_il. See Decision #26 / Decision #28.
//
// The 1:1 eviction/promotion pairing means only the minimum of
// |evict_set| and |promote_set| swaps actually happen per call — if
// more experts enter the new set than leave it (e.g., expanding the hot
// set from partial fill), the extras are dropped. In STEADY mode where
// current_size == K on every call, evict_set and promote_set have the
// same cardinality by construction, so this is not reachable.
//
// Returns the number of (evict, promote) pairs actually swapped (== the
// per-layer "churn" count for telemetry). Shared experts in both old and
// new sets contribute 0 swaps; a fully-stable layer returns 0, a
// completely-replaced layer returns min(K, new_hot_len). Caller may
// ignore the return if churn tracking is not needed.
int llama_moe_hot_cache_swap_layer(
    struct llama_moe_hot_cache * cache,
    int layer_idx,
    const int32_t * new_hot_set,
    int new_hot_len);

// Testable wrapper for the internal hot_cache_rebalance(). Exposed only so
// tests/test-moe-hot-cache.cpp can drive the tumbling-window rebalance path
// without needing a real llama_context. Production callers should not use
// this — the rebalance fires automatically from post_decode.
void llama_moe_hot_cache_rebalance_for_test(struct llama_moe_hot_cache * cache);

// Testable wrapper for the internal compute_entropy(). Returns Shannon entropy
// in bits of the given counts array. Exposed so unit tests can verify the
// entropy computation in isolation without requiring a real cache or telemetry
// context.
double llama_moe_hot_cache_compute_entropy(const uint32_t * counts, int n);
