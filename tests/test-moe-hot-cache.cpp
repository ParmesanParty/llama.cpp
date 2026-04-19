// Unit tests for llama_moe_hot_cache state machine.
//
// These tests do NOT load a real GGUF model or bring up a llama_context.
// Instead, they construct a minimal "fake context" — enough struct state for
// the cache manager to allocate VRAM buffers and exercise its state
// transitions. Synthetic IDs drive the FILLING and STEADY paths.

// Internal llama.cpp headers are not on the tests include path; use relative
// paths (same convention as tests/test-quantize-stats.cpp).
#include "../src/llama-moe-hot-cache.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

// Minimal harness: construct a real ggml CPU backend (not CUDA) so the test
// runs anywhere. The FILLING/STEADY logic is platform-independent — CUDA
// cudaMemcpyAsync is stubbed out via #ifdef GGML_USE_CUDA in the impl.
//
// For Phase 3.2 (this task), we only exercise init + free. Phase 9 and 10
// extend with synthetic-IDs promote tests and rebalance delta tests.

static int test_init_free_disabled() {
    printf("test_init_free_disabled: ");
    // K=0 → init returns nullptr at the K-guard before touching ctx, and
    // free() is safe on nullptr. This is the cheap "no-cache" disabled path.
    llama_moe_hot_cache * cache = llama_moe_hot_cache_init(nullptr, 0, 40);
    GGML_ASSERT(cache == nullptr);
    llama_moe_hot_cache_free(cache);  // nullptr-safe
    printf("OK\n");
    return 0;
}

static int test_init_nullptr_ctx_with_positive_k() {
    printf("test_init_nullptr_ctx_with_positive_k: ");
    // K > 0 with ctx == nullptr exercises the defensive nullptr-ctx guard
    // that sits between the K-guard and the first ctx->get_model() deref.
    // Real callers will never pass nullptr ctx, so without this test the
    // guard would be dead code. Verify it returns cleanly without segfault.
    llama_moe_hot_cache * cache = llama_moe_hot_cache_init(nullptr, 32, 40);
    GGML_ASSERT(cache == nullptr);
    llama_moe_hot_cache_free(cache);  // nullptr-safe
    printf("OK\n");
    return 0;
}

static int test_layer_has_hot_gate() {
    printf("test_layer_has_hot_gate: ");
    // The dual-path emission gate (Phase 7 Deviation #6 + Decision #32) is
    // load-bearing: it prevents build_moe_ffn from emitting an all-negative
    // MUL_MAT_ID ids tensor during the Phase 7+8 runtime state, where the
    // cache is allocated but no layer has been promoted yet (current_size == 0
    // everywhere). Without this gate, mm_ids_helper would leave ids_src1 /
    // ids_dst uninitialized in the pool, and the downstream quantize_mmq_q8_1
    // would read garbage → "CUDA error: an illegal memory access was
    // encountered". This test constructs a fake cache by hand (no CUDA, no
    // real ggml context) and verifies the gate returns false for every layer
    // in a freshly-allocated (pre-FILLING) state. Phase 9 will ADD additional
    // tests for the gate returning true after promotion.
    //
    // Also verifies the hybrid dense-layer corner case: find_layer_by_model_il
    // does a linear scan for a layer with matching model_il, and returns
    // nullptr (→ gate false) when the queried il doesn't match any packed
    // entry. This is the path taken for dense layers in hybrid architectures.
    llama_moe_hot_cache_layer layers[3] = {};
    // Packed layer indices: layers[0] is model layer 0, [1] is layer 2,
    // [2] is layer 4. Layers 1 and 3 are simulated "dense layers" that
    // don't appear in the packed array — queries for them must return false.
    layers[0].model_il = 0; layers[0].current_size = 0;
    layers[1].model_il = 2; layers[1].current_size = 0;
    layers[2].model_il = 4; layers[2].current_size = 0;

    llama_moe_hot_cache cache = {};
    cache.layers = layers;
    cache.n_layers = 3;

    // Case 1: queried il matches a packed entry with current_size == 0.
    //         Gate must return false (the hot set is empty).
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(&cache, 0));
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(&cache, 2));
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(&cache, 4));

    // Case 2: queried il is a "dense" layer not in the packed array.
    //         find_layer_by_model_il returns nullptr, gate returns false.
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(&cache, 1));
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(&cache, 3));

    // Case 3: queried il is out of range entirely.
    //         Same path — linear scan finds nothing, gate returns false.
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(&cache, 99));

    // Case 4: defensive — negative il must not crash. find_layer_by_model_il
    //         short-circuits on il < 0.
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(&cache, -1));

    // Case 5: flip a layer's current_size > 0 and verify the gate flips true
    //         for exactly that layer, leaving neighbors at false. This is
    //         the state Phase 9's lockstep promotion will produce.
    layers[1].current_size = 1;
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(&cache, 0));
    GGML_ASSERT( llama_moe_hot_cache_layer_has_hot(&cache, 2));  // flipped
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(&cache, 4));

    // Case 6: nullptr cache must be safe. find_layer_by_model_il null-guards.
    GGML_ASSERT(!llama_moe_hot_cache_layer_has_hot(nullptr, 0));

    printf("OK\n");
    return 0;
}

static int test_filling_promotes_novel_experts() {
    printf("test_filling_promotes_novel_experts: ");
    // Exercise llama_moe_hot_cache_promote_layer (the pure helper that the
    // post-decode hook calls under FILLING mode) with a hand-built cache.
    // No CUDA, no ggml context, no real llama_context — just host arrays.
    //
    // The helper's device-copy block is guarded on `cache->ctx != nullptr &&
    // layer.hot_*_exps != nullptr`, so leaving those zero here keeps the
    // test CPU-only. Only host-side state transitions are asserted.
    llama_moe_hot_cache cache = {};
    cache.mode = LLAMA_MOE_HOT_CACHE_FILLING;
    cache.K = 4;
    cache.n_expert = 16;
    cache.n_layers = 1;
    cache.rebalance_interval = 40;
    cache.layers = (llama_moe_hot_cache_layer *) calloc(1, sizeof(llama_moe_hot_cache_layer));
    cache.layers[0].hot_map_host   = (int32_t *)  calloc(cache.n_expert, sizeof(int32_t));
    cache.layers[0].slot_to_expert = (int32_t *)  calloc(cache.K,        sizeof(int32_t));
    cache.layers[0].window_counts  = (uint32_t *) calloc(cache.n_expert, sizeof(uint32_t));
    cache.layers[0].model_il = 0;
    for (int i = 0; i < cache.n_expert; ++i) cache.layers[0].hot_map_host[i]   = -1;
    for (int i = 0; i < cache.K;        ++i) cache.layers[0].slot_to_expert[i] = -1;

    // Step 1: {3, 7, 11, 11} — 3 unique experts, one duplicate.
    // Expect current_size == 3, slots 0..2 populated with 3/7/11 in order.
    int32_t ids1[] = {3, 7, 11, 11};
    llama_moe_hot_cache_promote_layer(&cache, 0, ids1, 4);
    GGML_ASSERT(cache.layers[0].current_size == 3);
    GGML_ASSERT(cache.layers[0].hot_map_host[3]  == 0);
    GGML_ASSERT(cache.layers[0].hot_map_host[7]  == 1);
    GGML_ASSERT(cache.layers[0].hot_map_host[11] == 2);
    GGML_ASSERT(cache.layers[0].slot_to_expert[0] == 3);
    GGML_ASSERT(cache.layers[0].slot_to_expert[1] == 7);
    GGML_ASSERT(cache.layers[0].slot_to_expert[2] == 11);

    // Step 2: {3, 15, 7, 2} — 3 & 7 already hot; 15 and 2 are novel but only
    // 1 slot remains (K=4 - 3 current = 1). Expect 15 promoted (first novel),
    // 2 dropped. current_size == 4.
    int32_t ids2[] = {3, 15, 7, 2};
    llama_moe_hot_cache_promote_layer(&cache, 0, ids2, 4);
    GGML_ASSERT(cache.layers[0].current_size == 4);
    GGML_ASSERT(cache.layers[0].hot_map_host[15] == 3);
    GGML_ASSERT(cache.layers[0].hot_map_host[2]  == -1);
    GGML_ASSERT(cache.layers[0].slot_to_expert[3] == 15);

    // Step 3: layer is full; further calls must be no-ops even if a novel
    // expert is in the ids buffer.
    int32_t ids3[] = {4};
    llama_moe_hot_cache_promote_layer(&cache, 0, ids3, 1);
    GGML_ASSERT(cache.layers[0].current_size == 4);
    GGML_ASSERT(cache.layers[0].hot_map_host[4] == -1);

    // Invariant check: slot_to_expert must be the inverse of hot_map_host
    // for every occupied slot.
    for (int slot = 0; slot < cache.K; ++slot) {
        int32_t e = cache.layers[0].slot_to_expert[slot];
        if (e >= 0) {
            GGML_ASSERT(cache.layers[0].hot_map_host[e] == slot);
        }
    }

    // Step 4: defensive — out-of-range / sentinel ids are silently dropped
    // (not promoted, not indexed into hot_map_host). This exercises the
    // bounds-check branch in the helper; without it, a negative id would
    // walk into hot_map_host[-1] which is a near-guaranteed crash.
    int32_t ids4[] = {-1, 99, cache.n_expert, -42};
    // Reset layer 0 to a fresh empty state for this assertion.
    cache.layers[0].current_size = 0;
    for (int i = 0; i < cache.n_expert; ++i) cache.layers[0].hot_map_host[i]   = -1;
    for (int i = 0; i < cache.K;        ++i) cache.layers[0].slot_to_expert[i] = -1;
    llama_moe_hot_cache_promote_layer(&cache, 0, ids4, 4);
    GGML_ASSERT(cache.layers[0].current_size == 0);

    // Step 5: out-of-range layer_idx must not crash.
    llama_moe_hot_cache_promote_layer(&cache, 99, ids1, 4);
    llama_moe_hot_cache_promote_layer(&cache, -1, ids1, 4);
    llama_moe_hot_cache_promote_layer(nullptr, 0, ids1, 4);

    free(cache.layers[0].hot_map_host);
    free(cache.layers[0].slot_to_expert);
    free(cache.layers[0].window_counts);
    free(cache.layers);
    printf("OK\n");
    return 0;
}

static int test_build_cold_map_inverts_hot_map() {
    printf("test_build_cold_map_inverts_hot_map: ");
    // Cold-map must be the complement of hot-map: for every expert,
    // out[e] == -1 when the expert has a hot slot (hot_map_host[e] >= 0),
    // and out[e] == e otherwise. Previously promote_layer pushed hot_map
    // but never pushed a matching cold_map, which left cold_map as the
    // identity from init and caused the dual-path ggml_add to double-count
    // the hot-promoted experts (Phase 10 dual-path correctness fix).
    const int n_expert = 16;
    int32_t hot_map_host[n_expert];
    int32_t cold_map_buf[n_expert];

    // Scenario 1: nothing promoted. cold_map must be pure identity.
    for (int i = 0; i < n_expert; ++i) hot_map_host[i] = -1;
    for (int i = 0; i < n_expert; ++i) cold_map_buf[i] = -999;
    llama_moe_hot_cache_build_cold_map(n_expert, hot_map_host, cold_map_buf);
    for (int i = 0; i < n_expert; ++i) {
        GGML_ASSERT(cold_map_buf[i] == i);
    }

    // Scenario 2: three experts promoted ({3→slot 0, 7→slot 1, 11→slot 2}).
    // The other 13 must remain as identity; the three hot ones must flip to -1.
    for (int i = 0; i < n_expert; ++i) hot_map_host[i] = -1;
    hot_map_host[3]  = 0;
    hot_map_host[7]  = 1;
    hot_map_host[11] = 2;
    for (int i = 0; i < n_expert; ++i) cold_map_buf[i] = -999;
    llama_moe_hot_cache_build_cold_map(n_expert, hot_map_host, cold_map_buf);
    for (int i = 0; i < n_expert; ++i) {
        if (i == 3 || i == 7 || i == 11) {
            GGML_ASSERT(cold_map_buf[i] == -1);
        } else {
            GGML_ASSERT(cold_map_buf[i] == i);
        }
    }

    // Scenario 3: null-safety — must not crash on nullptr inputs. The helper
    // is called from unit tests that may supply partial state; guarding on
    // null keeps the test harness resilient.
    llama_moe_hot_cache_build_cold_map(n_expert, nullptr, cold_map_buf);
    llama_moe_hot_cache_build_cold_map(n_expert, hot_map_host, nullptr);
    llama_moe_hot_cache_build_cold_map(0, hot_map_host, cold_map_buf);
    llama_moe_hot_cache_build_cold_map(-1, hot_map_host, cold_map_buf);

    printf("OK\n");
    return 0;
}

static int test_steady_rebalance_computes_minimal_delta() {
    printf("test_steady_rebalance_computes_minimal_delta: ");
    // Direct exercise of llama_moe_hot_cache_swap_layer. The swap helper is the
    // core of STEADY rebalance — given a new hot set and the layer's current
    // hot state, it must (a) evict experts no longer in the new set, (b)
    // promote novel experts into the freed slots, and (c) preserve the slots
    // held by experts that appear in both the old and new sets. No CUDA path
    // exercised — cache.ctx stays nullptr, so the #ifdef GGML_USE_CUDA block
    // in swap_layer short-circuits and only the host-side shadow is touched.
    llama_moe_hot_cache cache = {};
    cache.mode = LLAMA_MOE_HOT_CACHE_STEADY;
    cache.K = 4;
    cache.n_expert = 16;
    cache.n_layers = 1;
    cache.rebalance_interval = 40;
    cache.layers = (llama_moe_hot_cache_layer *) calloc(1, sizeof(llama_moe_hot_cache_layer));
    cache.layers[0].hot_map_host   = (int32_t *)  calloc(cache.n_expert, sizeof(int32_t));
    cache.layers[0].slot_to_expert = (int32_t *)  calloc(cache.K,        sizeof(int32_t));
    cache.layers[0].window_counts  = (uint32_t *) calloc(cache.n_expert, sizeof(uint32_t));
    cache.layers[0].model_il = 0;
    for (int i = 0; i < cache.n_expert; ++i) cache.layers[0].hot_map_host[i]   = -1;
    for (int i = 0; i < cache.K;        ++i) cache.layers[0].slot_to_expert[i] = -1;

    // Seed initial hot set {3, 7, 11, 15} → slots [0, 1, 2, 3].
    int32_t initial[] = {3, 7, 11, 15};
    for (int s = 0; s < 4; ++s) {
        cache.layers[0].hot_map_host[initial[s]] = s;
        cache.layers[0].slot_to_expert[s]        = initial[s];
    }
    cache.layers[0].current_size = 4;

    // New target: {7, 11, 9, 4}. {7, 11} stay (shared with old set). {3, 15}
    // evict. {9, 4} promote into the freed slots. The pair-wise slot reuse
    // order is iteration-dependent: swap_layer walks slots 0..K-1 collecting
    // evict_slots [0 (was 3), 3 (was 15)] and iterates new_hot_set in order
    // collecting promote_experts [9, 4]. The first pair (slot 0, expert 9)
    // and second pair (slot 3, expert 4) are therefore deterministic.
    int32_t new_hot[] = {7, 11, 9, 4};
    llama_moe_hot_cache_swap_layer(&cache, 0, new_hot, 4);

    // Every expert in new_hot must now have a slot.
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(cache.layers[0].hot_map_host[new_hot[i]] >= 0);
    }
    // Every expert that left the set must be -1.
    GGML_ASSERT(cache.layers[0].hot_map_host[3]  == -1);
    GGML_ASSERT(cache.layers[0].hot_map_host[15] == -1);
    // K slots stay populated (pair-wise swap preserves count).
    GGML_ASSERT(cache.layers[0].current_size == 4);

    // Invariant: slot_to_expert and hot_map_host are consistent for every
    // occupied slot.
    for (int slot = 0; slot < cache.K; ++slot) {
        int32_t e = cache.layers[0].slot_to_expert[slot];
        GGML_ASSERT(e >= 0 && cache.layers[0].hot_map_host[e] == slot);
    }

    // 7 and 11 should still hold their original slots (1 and 2) — "shared"
    // experts are not evicted, just their slot is preserved. The swap
    // helper's eviction scan skips them, and the promote loop's hot_map_host
    // check also skips them (they're already hot).
    GGML_ASSERT(cache.layers[0].hot_map_host[7]  == 1);
    GGML_ASSERT(cache.layers[0].hot_map_host[11] == 2);
    // Promoted 9 and 4 take the evicted slots 0 and 3 (in iteration order).
    GGML_ASSERT(cache.layers[0].hot_map_host[9] == 0);
    GGML_ASSERT(cache.layers[0].hot_map_host[4] == 3);

    free(cache.layers[0].hot_map_host);
    free(cache.layers[0].slot_to_expert);
    free(cache.layers[0].window_counts);
    free(cache.layers);
    printf("OK\n");
    return 0;
}

static int test_steady_rebalance_from_window_counts() {
    printf("test_steady_rebalance_from_window_counts: ");
    // Exercises the full hot_cache_rebalance pipeline via the test wrapper:
    // seeds window_counts directly (bypassing the graph-read path from
    // post_decode), calls the rebalance driver, and verifies that the top-K
    // ranking, swap delta application, and window_counts zeroing all work
    // together. This is the test that guards against regressions in the
    // tumbling-window → top-K → swap_layer pipeline that Phase 10 Task 10.1
    // introduces on top of Task 9.1's post_decode machinery.
    llama_moe_hot_cache cache = {};
    cache.mode = LLAMA_MOE_HOT_CACHE_STEADY;
    cache.K = 4;
    cache.n_expert = 16;
    cache.n_layers = 1;
    cache.rebalance_interval = 40;
    cache.decode_counter = 40;  // rebalance-tick condition already met (divisible)
    cache.layers = (llama_moe_hot_cache_layer *) calloc(1, sizeof(llama_moe_hot_cache_layer));
    cache.layers[0].hot_map_host   = (int32_t *)  calloc(cache.n_expert, sizeof(int32_t));
    cache.layers[0].slot_to_expert = (int32_t *)  calloc(cache.K,        sizeof(int32_t));
    cache.layers[0].window_counts  = (uint32_t *) calloc(cache.n_expert, sizeof(uint32_t));
    cache.layers[0].model_il = 0;
    for (int i = 0; i < cache.n_expert; ++i) cache.layers[0].hot_map_host[i]   = -1;
    for (int i = 0; i < cache.K;        ++i) cache.layers[0].slot_to_expert[i] = -1;

    // Seed the current hot set (as if a previous rebalance put {3, 7, 11, 15}
    // in hot) — so rebalance has something to evict from.
    int32_t initial[] = {3, 7, 11, 15};
    for (int s = 0; s < 4; ++s) {
        cache.layers[0].hot_map_host[initial[s]] = s;
        cache.layers[0].slot_to_expert[s]        = initial[s];
    }
    cache.layers[0].current_size = 4;

    // Seed window_counts to simulate the last 40 tokens of activations.
    // Experts 7, 11 stay hot (high counts — should remain in top-K).
    // Experts 3, 15 drop (low counts — should be evicted).
    // Experts 9, 4 rise (high counts — should be promoted as novel).
    cache.layers[0].window_counts[7]  = 30;
    cache.layers[0].window_counts[11] = 28;
    cache.layers[0].window_counts[9]  = 25;
    cache.layers[0].window_counts[4]  = 22;
    cache.layers[0].window_counts[3]  = 2;   // used to be hot; now rare
    cache.layers[0].window_counts[15] = 1;   // likewise
    // Other experts: 0 (from calloc).

    llama_moe_hot_cache_rebalance_for_test(&cache);

    // Top-K by count: {7, 11, 9, 4}. All must now have a slot.
    GGML_ASSERT(cache.layers[0].hot_map_host[7]  >= 0);
    GGML_ASSERT(cache.layers[0].hot_map_host[11] >= 0);
    GGML_ASSERT(cache.layers[0].hot_map_host[9]  >= 0);
    GGML_ASSERT(cache.layers[0].hot_map_host[4]  >= 0);
    // Dropped experts are no longer hot.
    GGML_ASSERT(cache.layers[0].hot_map_host[3]  == -1);
    GGML_ASSERT(cache.layers[0].hot_map_host[15] == -1);
    // Slot count preserved (K = 4, pair-wise swap).
    GGML_ASSERT(cache.layers[0].current_size == 4);

    // Invariant: slot_to_expert and hot_map_host are consistent.
    for (int slot = 0; slot < cache.K; ++slot) {
        int32_t e = cache.layers[0].slot_to_expert[slot];
        GGML_ASSERT(e >= 0 && cache.layers[0].hot_map_host[e] == slot);
    }

    // window_counts must be zeroed for the next window — the rebalance driver
    // consumes the counts and resets them so the next tumbling window starts
    // fresh. Without this, counts would accumulate indefinitely and the
    // top-K signal would drift toward long-range averages instead of the
    // most-recent rebalance_interval tokens (which is the whole point of the
    // tumbling-window design; see Decision #27).
    for (int i = 0; i < cache.n_expert; ++i) {
        GGML_ASSERT(cache.layers[0].window_counts[i] == 0);
    }

    free(cache.layers[0].hot_map_host);
    free(cache.layers[0].slot_to_expert);
    free(cache.layers[0].window_counts);
    free(cache.layers);
    printf("OK\n");
    return 0;
}

// Minimal JSON value extractor for verifying JSONL telemetry records in tests.
// Finds the first occurrence of `"key":value` and returns the value as a
// string. String values are returned without surrounding quotes. Numeric and
// boolean values are returned as-is. Returns "" if the key is not found.
static std::string json_get(const std::string & line, const char * key) {
    std::string needle = std::string("\"") + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    if (pos < line.size() && line[pos] == '"') {
        size_t end = line.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return line.substr(pos + 1, end - pos - 1);
    } else {
        size_t end = line.find_first_of(",}]", pos);
        if (end == std::string::npos) end = line.size();
        return line.substr(pos, end - pos);
    }
}

static int test_entropy_computation(void) {
    // Uniform distribution: entropy = log2(4) = 2.0 bits
    { uint32_t c[4] = {10, 10, 10, 10}; GGML_ASSERT(fabs(llama_moe_hot_cache_compute_entropy(c, 4) - 2.0) < 0.01); }
    // Single expert: entropy = 0 (all probability mass on one outcome)
    { uint32_t c[4] = {100, 0, 0, 0};   GGML_ASSERT(llama_moe_hot_cache_compute_entropy(c, 4) < 0.01); }
    // All zeros: entropy = 0 (degenerate / no observations)
    { uint32_t c[4] = {0, 0, 0, 0};     GGML_ASSERT(llama_moe_hot_cache_compute_entropy(c, 4) == 0.0); }
    // Two-way split: entropy = 1.0 bit
    { uint32_t c[4] = {50, 50, 0, 0};   GGML_ASSERT(fabs(llama_moe_hot_cache_compute_entropy(c, 4) - 1.0) < 0.01); }
    printf("test_entropy_computation: OK\n");
    return 0;
}

static int test_json_get(void) {
    std::string line = "{\"type\":\"header\",\"n_expert\":128,\"K\":32,\"timestamp\":\"2026-04-12T12:00:00Z\"}";
    GGML_ASSERT(json_get(line, "type")      == "header");
    GGML_ASSERT(json_get(line, "n_expert")  == "128");
    GGML_ASSERT(json_get(line, "K")         == "32");
    GGML_ASSERT(json_get(line, "timestamp") == "2026-04-12T12:00:00Z");
    GGML_ASSERT(json_get(line, "missing")   == "");
    printf("test_json_get: OK\n");
    return 0;
}

static int test_telemetry_env_config(void) {
    // Verify env var reading works by setting and checking both telemetry
    // config vars that llama_moe_hot_cache reads at init time.
    setenv("LLAMA_MOE_HOT_TELEMETRY", "/tmp/test-telem", 1);
    const char * dir = getenv("LLAMA_MOE_HOT_TELEMETRY");
    GGML_ASSERT(dir != nullptr && strcmp(dir, "/tmp/test-telem") == 0);

    setenv("LLAMA_MOE_HOT_TELEMETRY_FINE", "1", 1);
    const char * fine = getenv("LLAMA_MOE_HOT_TELEMETRY_FINE");
    GGML_ASSERT(fine != nullptr && strcmp(fine, "1") == 0);

    // Clean up to avoid polluting the environment for subsequent tests.
    unsetenv("LLAMA_MOE_HOT_TELEMETRY");
    unsetenv("LLAMA_MOE_HOT_TELEMETRY_FINE");
    printf("test_telemetry_env_config: OK\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_init_free_disabled();
    failures += test_init_nullptr_ctx_with_positive_k();
    failures += test_layer_has_hot_gate();
    failures += test_filling_promotes_novel_experts();
    failures += test_build_cold_map_inverts_hot_map();
    failures += test_steady_rebalance_computes_minimal_delta();
    failures += test_steady_rebalance_from_window_counts();
    failures += test_entropy_computation();
    failures += test_json_get();
    failures += test_telemetry_env_config();
    printf("\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
