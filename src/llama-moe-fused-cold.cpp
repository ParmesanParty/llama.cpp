#include "llama-moe-fused-cold.h"
#include "llama-moe-perf.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <omp.h>

static inline float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

void llama_moe_fused_cold_compute(
    struct ggml_tensor * dst, int ith, int nth, void * userdata) {

    auto * p = (struct llama_moe_fused_cold_params *) userdata;

    const struct ggml_tensor * up_exps   = dst->src[0];
    const struct ggml_tensor * gate_exps = dst->src[1];
    const struct ggml_tensor * down_exps = dst->src[2];
    const struct ggml_tensor * cur       = dst->src[3];
    const struct ggml_tensor * cold_ids  = dst->src[4];

    const int n_expert      = p->n_expert;
    const int n_expert_used = p->n_expert_used;
    const int n_embd        = p->n_embd;
    const int n_ff          = p->n_ff_exp;
    const int n_tokens      = (int) cold_ids->ne[1];
    const int total_slots   = n_expert_used * n_tokens;

    const auto * traits     = ggml_get_type_traits_cpu(up_exps->type);
    const auto   vdt        = traits->vec_dot_type;
    const auto * vdt_traits = ggml_get_type_traits_cpu(vdt);

    const size_t up_row_stride   = up_exps->nb[1];
    const size_t up_exp_stride   = up_exps->nb[2];
    const size_t gate_row_stride = gate_exps->nb[1];
    const size_t gate_exp_stride = gate_exps->nb[2];
    const size_t down_row_stride = down_exps->nb[1];
    const size_t down_exp_stride = down_exps->nb[2];
    const size_t cur_q_row_size  = ggml_row_size(vdt, n_embd);
    const size_t act_q_row_size  = ggml_row_size(vdt, n_ff);

    // ── Scratch (owned by params, shared across cache lifetime) ─────────
    // Finding 6 fix: replaces file-scope statics that (a) would race if ggml
    // ever parallelized across slots and (b) couldn't be freed without a
    // global reset on cache destruction. Scratch pointers live on the params
    // struct; thread 0 grows them on demand then ALL threads read via p.
    struct slot_mapping { int slot; int token; };

    // PERF: thread-0 wall-time spans. Captured only when LLAMA_MOE_HOT_PERF=1
    // — the `if (LLAMA_MOE_PERF_ON)` branch folds away when disabled. Spans
    // are bounded by OpenMP barriers: thread 0's elapsed between barriers
    // is the tight lower bound on phase cost (slowest thread extends the
    // barrier but thread 0 observes that wait as part of the phase).
    llama_moe_perf::time_point t_call_start;
    llama_moe_perf::time_point t_phase_start;
    if (ith == 0 && LLAMA_MOE_PERF_ON) {
        t_call_start  = llama_moe_perf::now();
        t_phase_start = t_call_start;
    }

    // ── Phase 0: thread 0 setup ─────────────────────────────────────────
    if (ith == 0) {
        // Grow scratch only when total_slots exceeds the current high-water
        // mark. free(nullptr) is a no-op so the first call falls through to
        // the malloc/calloc below without a guard.
        //
        // Finding F4 fix: if any malloc/calloc fails, thread 0 must NOT
        // silently leave nullptr pointers in the params struct. The other
        // threads read those pointers after BARRIER 1 and would deref
        // straight through into a segfault. Instead, on any allocation
        // failure we free back to a consistent zero state, publish a
        // sentinel (-1 stored in active_experts[n_expert]) that the other
        // threads check, and then fall through to zeroing `dst` so the
        // caller sees a well-defined output (zeros: the cold path's
        // contribution for this ubatch is lost, but correctness is
        // preserved — the hot branch still contributes and the sum is
        // whatever the hot branch alone produced).
        if (total_slots > p->max_total_slots) {
            free(p->scratch_f);  p->scratch_f  = nullptr;
            free(p->scratch_q);  p->scratch_q  = nullptr;
            free(p->scratch_rc); p->scratch_rc = nullptr;
            free(p->scratch_rm); p->scratch_rm = nullptr;
            free(p->scratch_ae); p->scratch_ae = nullptr;
            p->max_total_slots = 0;

            p->scratch_f  = malloc((size_t)n_ff * total_slots * sizeof(float));
            p->scratch_q  = malloc((cur_q_row_size + act_q_row_size) * total_slots);
            p->scratch_rc = calloc(n_expert, sizeof(int64_t));
            p->scratch_rm = malloc((size_t)n_expert * total_slots * sizeof(slot_mapping));
            // +1 so the compact list has room for the n_active sentinel past
            // the tail — see the "Compact list" loop below.
            p->scratch_ae = malloc((size_t)(n_expert + 1) * sizeof(int));

            const bool alloc_ok =
                p->scratch_f  != nullptr &&
                p->scratch_q  != nullptr &&
                p->scratch_rc != nullptr &&
                p->scratch_rm != nullptr &&
                p->scratch_ae != nullptr;

            if (!alloc_ok) {
                fprintf(stderr,
                    "WARN llama_moe_fused_cold_compute: scratch alloc failed "
                    "(total_slots=%d, n_ff=%d, n_expert=%d) — skipping cold "
                    "compute for this ubatch, dst will be zero\n",
                    total_slots, n_ff, n_expert);
                free(p->scratch_f);  p->scratch_f  = nullptr;
                free(p->scratch_q);  p->scratch_q  = nullptr;
                free(p->scratch_rc); p->scratch_rc = nullptr;
                free(p->scratch_rm); p->scratch_rm = nullptr;
                free(p->scratch_ae); p->scratch_ae = nullptr;
                p->max_total_slots = 0;
                // Zero dst so downstream ggml_add sees well-defined values.
                // All threads will cooperatively bail after BARRIER 1 by
                // checking for the OOM sentinel and skipping to the
                // memset, so we don't need to memset here — the shared
                // Phase-between-B1-B2 block (lines ~137-153) zeros dst
                // already, but that path depends on active_experts being
                // readable. Since we just freed active_experts, publish
                // the sentinel via a local marker: set n_active == -1
                // in a side-channel. We reuse scratch_ae==nullptr as the
                // "alloc failed" sentinel — readable by every thread.
                // Fall through to the barrier below with nullptr pointers
                // everywhere; the post-barrier block detects this and
                // zeros dst cooperatively.
            } else {
                p->max_total_slots = total_slots;
            }
        }

        // If scratch is available, perform grouping. If not (alloc failed
        // above), skip — the post-barrier block handles dst zeroing and
        // early bail.
        if (p->scratch_ae != nullptr) {
            int64_t *      row_counts_t0     = (int64_t *)      p->scratch_rc;
            slot_mapping * row_map_t0        = (slot_mapping *) p->scratch_rm;
            int *          active_experts_t0 = (int *)          p->scratch_ae;

            memset(row_counts_t0, 0, n_expert * sizeof(int64_t));
            for (int t = 0; t < n_tokens; ++t) {
                for (int s = 0; s < n_expert_used; ++s) {
                    const int32_t eid = ((const int32_t *) cold_ids->data)[t * n_expert_used + s];
                    if (eid < 0 || eid >= n_expert) continue;
                    row_map_t0[eid * total_slots + row_counts_t0[eid]] = {s, t};
                    row_counts_t0[eid] += 1;
                }
            }

            // Build compact list of active experts (typically 6-8 of 256). Store
            // n_active in the sentinel slot [n_expert] so the other threads can
            // recover it after BARRIER 1 without a separate params field.
            int n_active_t0 = 0;
            for (int e = 0; e < n_expert; ++e) {
                if (row_counts_t0[e] > 0) active_experts_t0[n_active_t0++] = e;
            }
            active_experts_t0[n_expert] = n_active_t0;
        }
    }

    // ── BARRIER 1: pointers + grouping ready ────────────────────────────
    #pragma omp barrier

    // Finding F4 fix — cooperative OOM bail. If thread 0's scratch alloc
    // failed, scratch_ae is nullptr and dereferencing active_experts below
    // would segfault. Every thread checks the sentinel here and, on OOM,
    // cooperatively zeros its share of dst (so downstream ggml_add sees a
    // defined value) and returns. Correctness: the cold branch's
    // contribution is lost for this one ubatch; the hot branch still
    // contributes, and the router wasn't affected. FILLING/STEADY counters
    // are untouched because post_decode reads the argsort tensor
    // separately from this kernel's output.
    if (p->scratch_ae == nullptr) {
        const size_t dst_bytes = ggml_nbytes(dst);
        const size_t bytes_per_thread = (dst_bytes + nth - 1) / nth;
        const size_t off = std::min((size_t)ith * bytes_per_thread, dst_bytes);
        const size_t end = std::min(off + bytes_per_thread, dst_bytes);
        if (end > off) memset((char *)dst->data + off, 0, end - off);
        return;
    }

    // All threads re-materialize local handles into params-owned scratch.
    // Thread 0's writes above are visible after the barrier. Other threads
    // recover n_active from the sentinel at active_experts[n_expert].
    float *          act_buf        = (float *)          p->scratch_f;
    char  *          cur_q          = (char *)           p->scratch_q;
    char  *          act_q          = (char *)           p->scratch_q
                                        + cur_q_row_size * total_slots;
    int64_t *        row_counts     = (int64_t *)        p->scratch_rc;
    slot_mapping *   row_map        = (slot_mapping *)   p->scratch_rm;
    int *            active_experts = (int *)            p->scratch_ae;
    const int        n_active       = active_experts[n_expert];

    if (ith == 0 && LLAMA_MOE_PERF_ON) {
        llama_moe_perf::stat_fc_setup.add(
            llama_moe_perf::elapsed_us(t_phase_start));
        llama_moe_perf::stat_fc_n_active.add_value((uint64_t) n_active);
        llama_moe_perf::stat_fc_total_slots.add_value((uint64_t) total_slots);
        t_phase_start = llama_moe_perf::now();
    }

    // Parallel memset + cur_q quantize (all threads)
    {
        // Zero dst (sentinel slots must produce zero)
        const size_t dst_bytes = ggml_nbytes(dst);
        const size_t bytes_per_thread = (dst_bytes + nth - 1) / nth;
        const size_t off = std::min((size_t)ith * bytes_per_thread, dst_bytes);
        const size_t end = std::min(off + bytes_per_thread, dst_bytes);
        if (end > off) memset((char *)dst->data + off, 0, end - off);

        // Quantize cur → vec_dot_type
        for (int s = ith; s < total_slots; s += nth) {
            const int t = s / n_expert_used;
            const int slot = s % n_expert_used;
            const float * src_row = (const float *)((const char *) cur->data +
                slot * cur->nb[1] + t * cur->nb[2]);
            vdt_traits->from_float(src_row, cur_q + (size_t)s * cur_q_row_size, n_embd);
        }
    }

    // ── BARRIER 2: dst zeroed + cur_q ready ─────────────────────────────
    // (BARRIER 1 was the grouping barrier at line 114; the sequence is
    //  B1 → memset+quant_cur → B2 → Phase 1 → B3 → quant_act → B4 → Phase 2.)
    #pragma omp barrier

    if (ith == 0 && LLAMA_MOE_PERF_ON) {
        llama_moe_perf::stat_fc_quant_cur.add(
            llama_moe_perf::elapsed_us(t_phase_start));
        t_phase_start = llama_moe_perf::now();
    }

    // ── Phase 1: fused up + gate + SwiGLU ───────────────────────────────
    // Slot outer, row inner (Finding 2 fix — was row-outer/slot-inner):
    //   - act_buf writes step by 4 bytes per row iteration (sequential),
    //     replacing the old 5 KB-stride-per-inner-iter write pattern that
    //     defeated the L1 cache prefetcher.
    //   - cur_q row for this flat slot stays resident in L1 for the full
    //     inner loop (it depends only on `flat`, which is fixed here).
    //   - Weight row pointers w_up/w_gate advance by up/gate_row_stride
    //     per row iteration — identical stream pattern to the prior layout.
    for (int ai = 0; ai < n_active; ++ai) {
        const int e = active_experts[ai];
        const int64_t n_rows_e = row_counts[e];

        const char * up_w   = (const char *) up_exps->data   + (size_t)e * up_exp_stride;
        const char * gate_w = (const char *) gate_exps->data  + (size_t)e * gate_exp_stride;

        const int rows_per_thread = (n_ff + nth - 1) / nth;
        const int row_start = ith * rows_per_thread;
        const int row_end   = std::min(row_start + rows_per_thread, n_ff);

        for (int64_t r = 0; r < n_rows_e; ++r) {
            const auto m = row_map[e * total_slots + r];
            const int flat = m.token * n_expert_used + m.slot;
            const char * q = cur_q + (size_t)flat * cur_q_row_size;
            float * act_row = act_buf + (size_t)flat * n_ff;

            // PERF: software prefetch the next few rows' weight data into L1/L2
            // before the compute catches up. Zen4 / AVX2 memory prefetch is ~30
            // cycles ahead of use; for Q4_K blocks (~700 bytes/row) touching the
            // first cache line per row is enough to kick off the DRAM fetch.
            constexpr int PREFETCH_AHEAD = 4;
            for (int pr = row_start; pr < std::min(row_start + PREFETCH_AHEAD, row_end); ++pr) {
                __builtin_prefetch(up_w   + (size_t)pr * up_row_stride,   0, 1);
                __builtin_prefetch(gate_w + (size_t)pr * gate_row_stride, 0, 1);
            }

            for (int row = row_start; row < row_end; ++row) {
                const char * w_up   = up_w   + (size_t)row * up_row_stride;
                const char * w_gate = gate_w + (size_t)row * gate_row_stride;

                // Prefetch the row PREFETCH_AHEAD ahead so its weights arrive
                // from DRAM while the current vec_dots compute.
                const int pf_row = row + PREFETCH_AHEAD;
                if (pf_row < row_end) {
                    __builtin_prefetch(up_w   + (size_t)pf_row * up_row_stride,   0, 1);
                    __builtin_prefetch(gate_w + (size_t)pf_row * gate_row_stride, 0, 1);
                }

                float up_val = 0.0f, gate_val = 0.0f;
                traits->vec_dot(n_embd, &up_val,   0, w_up,   0, q, 0, 1);
                traits->vec_dot(n_embd, &gate_val, 0, w_gate, 0, q, 0, 1);
                act_row[row] = silu_f32(gate_val) * up_val;
            }
        }
    }

    // ── BARRIER 3: Phase 1 done (act_buf ready) ─────────────────────────
    #pragma omp barrier

    if (ith == 0 && LLAMA_MOE_PERF_ON) {
        llama_moe_perf::stat_fc_phase1.add(
            llama_moe_perf::elapsed_us(t_phase_start));
        t_phase_start = llama_moe_perf::now();
    }

    // Quantize act → vec_dot_type
    for (int s = ith; s < total_slots; s += nth) {
        vdt_traits->from_float(
            act_buf + (size_t)s * n_ff,
            act_q + (size_t)s * act_q_row_size, n_ff);
    }

    // ── BARRIER 4: act_q ready for Phase 2 ──────────────────────────────
    #pragma omp barrier

    if (ith == 0 && LLAMA_MOE_PERF_ON) {
        llama_moe_perf::stat_fc_quant_act.add(
            llama_moe_perf::elapsed_us(t_phase_start));
        t_phase_start = llama_moe_perf::now();
    }

    // ── Phase 2: down matmul ────────────────────────────────────────────
    const auto * down_traits = ggml_get_type_traits_cpu(down_exps->type);

    for (int ai = 0; ai < n_active; ++ai) {
        const int e = active_experts[ai];
        const int64_t n_rows_e = row_counts[e];

        const char * down_w = (const char *) down_exps->data + (size_t)e * down_exp_stride;

        const int rows_per_thread = (n_embd + nth - 1) / nth;
        const int row_start = ith * rows_per_thread;
        const int row_end   = std::min(row_start + rows_per_thread, n_embd);

        for (int row = row_start; row < row_end; ++row) {
            const char * w = down_w + (size_t)row * down_row_stride;

            for (int64_t r = 0; r < n_rows_e; ++r) {
                const auto m = row_map[e * total_slots + r];
                const int flat = m.token * n_expert_used + m.slot;

                float d = 0.0f;
                down_traits->vec_dot(n_ff, &d, 0, w, 0,
                    act_q + (size_t)flat * act_q_row_size, 0, 1);

                float * out = (float *)((char *) dst->data +
                    (size_t)m.slot * dst->nb[1] + (size_t)m.token * dst->nb[2]);
                out[row] = d;
            }
        }
    }

    // PERF: Phase 2 span is measured only from thread 0's perspective —
    // there is no final barrier here (the enclosing OpenMP parallel region
    // provides one after return). Other threads may still be computing
    // when thread 0 falls out. At balanced row-partitioned work this is a
    // tight approximation; at imbalanced n_active it under-reports the
    // slowest thread's share. If we want true phase 2 cost, add an
    // explicit `#pragma omp barrier` before the record — but that adds a
    // real sync to every call, so leave it off by default.
    if (ith == 0 && LLAMA_MOE_PERF_ON) {
        llama_moe_perf::stat_fc_phase2.add(
            llama_moe_perf::elapsed_us(t_phase_start));
        llama_moe_perf::stat_fc_total.add(
            llama_moe_perf::elapsed_us(t_call_start));
    }
}

// Free scratch buffers owned by a params struct (Finding 6). Null-safe:
// free(nullptr) is a no-op so this works on zero-initialized, partially-
// populated, or fully-allocated params alike. Resets max_total_slots so a
// later compute call would re-allocate from scratch if needed.
void llama_moe_fused_cold_free_scratch(
    struct llama_moe_fused_cold_params * p) {
    if (p == nullptr) {
        return;
    }
    free(p->scratch_f);  p->scratch_f  = nullptr;
    free(p->scratch_q);  p->scratch_q  = nullptr;
    free(p->scratch_rc); p->scratch_rc = nullptr;
    free(p->scratch_rm); p->scratch_rm = nullptr;
    free(p->scratch_ae); p->scratch_ae = nullptr;
    p->max_total_slots = 0;
}
