// llama-moe-perf.h — env-gated wall-time instrumentation for the MoE hot
// cache and cold fused kernel. Header-only; no CMake wiring required.
//
// Usage:
//   if (LLAMA_MOE_PERF_ON) {
//       auto t0 = llama_moe_perf::now();
//       ... code ...
//       llama_moe_perf::stat_phase1.add(llama_moe_perf::elapsed_us(t0));
//   }
//
// Enabled at runtime via `LLAMA_MOE_HOT_PERF=1`. When disabled, every call
// to `llama_moe_perf::enabled()` reads a cached int and the compiler folds
// the if-branch away (the actual timing code is unreachable). When
// enabled, stats are accumulated in atomic counters and flushed every N
// calls via `maybe_report()` (called from post_decode).
//
// Design notes:
//   * Atomics used for thread safety — the fused cold kernel is OpenMP-
//     parallel so thread 0 writes while other threads may also be timing.
//     Relaxed ordering — stats are advisory.
//   * std::chrono::steady_clock: monotonic, ~ns resolution on Linux.
//   * One-shot env read (first call), cached in a static int.
//   * Reporting cadence: env LLAMA_MOE_HOT_PERF_REPORT_EVERY (default 200)
//     — one summary block per ~200 post_decode calls. At 35 tok/s that is
//     one log group every ~5.7s of steady-state generation.

#pragma once

#include "llama-impl.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace llama_moe_perf {

inline bool enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("LLAMA_MOE_HOT_PERF");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        if (cached) {
            // WARN-prefix so the llm-proxy stderr filter (proxy.py
            // _LLAMA_WARN_RE) promotes this to INFO level in journalctl.
            // This isn't a warning; it's a sprint-time diagnostic that
            // needs to be visible without raising the llama-server log
            // verbosity.
            fprintf(stderr,
                "WARN [moe-perf] wall-time markers ENABLED "
                "(LLAMA_MOE_HOT_PERF=%s). Summary every "
                "LLAMA_MOE_HOT_PERF_REPORT_EVERY decodes (default 200).\n",
                env);
            fflush(stderr);
        }
    }
    return cached != 0;
}

inline uint64_t report_every() {
    static uint64_t cached = 0;
    if (cached == 0) {
        const char * env = getenv("LLAMA_MOE_HOT_PERF_REPORT_EVERY");
        uint64_t v = 200;
        if (env != nullptr && env[0] != '\0') {
            long long tmp = atoll(env);
            if (tmp > 0) v = (uint64_t) tmp;
        }
        cached = v;
    }
    return cached;
}

#define LLAMA_MOE_PERF_ON (llama_moe_perf::enabled())

using clock_type = std::chrono::steady_clock;
using time_point = clock_type::time_point;

inline time_point now() { return clock_type::now(); }

inline double elapsed_us(time_point start) {
    const auto dt = clock_type::now() - start;
    return std::chrono::duration<double, std::micro>(dt).count();
}

struct stat_accum {
    const char *          name;
    std::atomic<uint64_t> count   {0};
    std::atomic<uint64_t> total_us{0};
    std::atomic<uint64_t> max_us  {0};
    std::atomic<uint64_t> sum_val {0};
    std::atomic<uint64_t> max_val {0};

    explicit stat_accum(const char * n) : name(n) {}

    void add(double us) {
        const auto u = (uint64_t) (us + 0.5);
        count.fetch_add(1, std::memory_order_relaxed);
        total_us.fetch_add(u, std::memory_order_relaxed);
        uint64_t cur = max_us.load(std::memory_order_relaxed);
        while (u > cur && !max_us.compare_exchange_weak(
                              cur, u, std::memory_order_relaxed)) {
        }
    }

    void add_value(uint64_t v) {
        sum_val.fetch_add(v, std::memory_order_relaxed);
        uint64_t cur = max_val.load(std::memory_order_relaxed);
        while (v > cur && !max_val.compare_exchange_weak(
                              cur, v, std::memory_order_relaxed)) {
        }
    }

    void reset_window() {
        count.store(0,    std::memory_order_relaxed);
        total_us.store(0, std::memory_order_relaxed);
        max_us.store(0,   std::memory_order_relaxed);
        sum_val.store(0,  std::memory_order_relaxed);
        max_val.store(0,  std::memory_order_relaxed);
    }
};

// Fused cold kernel (per-call; measured by thread 0).
inline stat_accum stat_fc_total      {"fused_cold.total"};
inline stat_accum stat_fc_setup      {"fused_cold.phase0_setup"};
inline stat_accum stat_fc_quant_cur  {"fused_cold.quant_cur+memset"};
inline stat_accum stat_fc_phase1     {"fused_cold.phase1_up_gate_swiglu"};
inline stat_accum stat_fc_quant_act  {"fused_cold.quant_act"};
inline stat_accum stat_fc_phase2     {"fused_cold.phase2_down"};
inline stat_accum stat_fc_n_active   {"fused_cold.n_active_experts"};
inline stat_accum stat_fc_total_slots{"fused_cold.total_slots"};

// Hot cache post_decode + promote + swap + rebalance.
inline stat_accum stat_pd_total       {"post_decode.total"};
inline stat_accum stat_pd_snapshot    {"post_decode.snapshot_ids_async"};
inline stat_accum stat_pd_accumulate  {"post_decode.window_accumulate"};
inline stat_accum stat_pd_promote     {"post_decode.promote_calls_total"};
inline stat_accum stat_pd_rebalance   {"post_decode.rebalance_total"};
inline stat_accum stat_pd_idx_build   {"post_decode.graph_index_build"};
inline stat_accum stat_pd_idx_reuse   {"post_decode.graph_index_reuse"};
inline stat_accum stat_pd_ids_overflow{"post_decode.pinned_overflow_layers"};
inline stat_accum stat_promote_stage  {"promote_layer.stage+sync"};
inline stat_accum stat_swap_total     {"swap_layer.stage+sync"};
inline stat_accum stat_swap_pairs     {"swap_layer.n_pairs"};
inline stat_accum stat_rb_sort        {"rebalance.partial_sort_all_layers"};
inline stat_accum stat_rb_entropy     {"rebalance.entropy_all_layers"};

static inline void report_one(stat_accum & s) {
    const uint64_t c  = s.count.load(std::memory_order_relaxed);
    const uint64_t t  = s.total_us.load(std::memory_order_relaxed);
    const uint64_t m  = s.max_us.load(std::memory_order_relaxed);
    const uint64_t sv = s.sum_val.load(std::memory_order_relaxed);
    const uint64_t mv = s.max_val.load(std::memory_order_relaxed);
    if (c == 0 && sv == 0) {
        return;
    }
    const double avg_us = c > 0 ? (double) t / (double) c : 0.0;
    const double avg_v  = c > 0 ? (double) sv / (double) c : 0.0;
    if (sv > 0 || mv > 0) {
        // WARN-prefix per the llm-proxy stderr-filter convention — this
        // is diagnostic, not a warning, but LLAMA_LOG_INFO would be
        // filtered to DEBUG and silently dropped.
        fprintf(stderr,
            "WARN [moe-perf]   %-38s  n=%-8llu  avg=%7.1fus  max=%6lluus  "
            "val_avg=%.1f  val_max=%llu\n",
            s.name,
            (unsigned long long) c,
            avg_us,
            (unsigned long long) m,
            avg_v,
            (unsigned long long) mv);
    } else {
        fprintf(stderr,
            "WARN [moe-perf]   %-38s  n=%-8llu  avg=%7.1fus  max=%6lluus  "
            "sum=%.2fms\n",
            s.name,
            (unsigned long long) c,
            avg_us,
            (unsigned long long) m,
            (double) t / 1000.0);
    }
    fflush(stderr);
    s.reset_window();
}

inline void maybe_report() {
    if (!enabled()) return;
    const uint64_t c  = stat_pd_total.count.load(std::memory_order_relaxed);
    const uint64_t re = report_every();
    if (c == 0 || (c % re) != 0) return;

    fprintf(stderr,
        "WARN [moe-perf] window summary (last %llu post_decode calls)\n",
        (unsigned long long) re);
    fflush(stderr);
    report_one(stat_pd_total);
    report_one(stat_pd_snapshot);
    report_one(stat_pd_accumulate);
    report_one(stat_pd_ids_overflow);
    report_one(stat_pd_idx_build);
    report_one(stat_pd_idx_reuse);
    report_one(stat_pd_promote);
    report_one(stat_promote_stage);
    report_one(stat_pd_rebalance);
    report_one(stat_rb_sort);
    report_one(stat_rb_entropy);
    report_one(stat_swap_total);
    report_one(stat_swap_pairs);
    report_one(stat_fc_total);
    report_one(stat_fc_setup);
    report_one(stat_fc_quant_cur);
    report_one(stat_fc_phase1);
    report_one(stat_fc_quant_act);
    report_one(stat_fc_phase2);
    report_one(stat_fc_n_active);
    report_one(stat_fc_total_slots);
}

}  // namespace llama_moe_perf
