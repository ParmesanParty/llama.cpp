// ggml-moe-diag.h — env-gated wall-time diagnostic instrumentation for the
// MUL_MAT_ID compute path and the backend scheduler's split loop.
//
// Purpose: characterize where time is spent when the MoE hot-cache dual-path
// is engaged at PP-sized ubatches, so the April 16 PP regression
// investigation can validate (or falsify) its hypotheses about memset tax,
// groupby, and cross-backend sync. See docs/superpowers/handoffs/
// 2026-04-16-pp-dualpath-investigation-handoff.md.
//
// Design:
//   - Env-gated (LLAMA_MMID_DIAG=1). Zero-overhead when off.
//   - Thread-0-only accumulators. No atomics: the MMID forward is called
//     within a compute-thread pool that never runs two ops concurrently,
//     and the scheduler's compute_splits is single-threaded.
//   - Auto-report every LLAMA_MMID_DIAG_REPORT_EVERY MMID ops (default 1000).
//   - fprintf(stderr, "WARN [mmid-diag] ...") — promoted to INFO by the
//     llm-proxy stderr filter so it lands in journalctl without raising
//     global log verbosity.
//
// Pure C so ggml-cpu.c (C) and ggml-backend.cpp (C++) can share the header.
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int ggml_moe_diag_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * e = getenv("LLAMA_MMID_DIAG");
        cached = (e != NULL && e[0] != '\0' && e[0] != '0') ? 1 : 0;
        if (cached) {
            fprintf(stderr,
                "WARN [mmid-diag] ENABLED (LLAMA_MMID_DIAG=%s). Reports every "
                "LLAMA_MMID_DIAG_REPORT_EVERY MMID ops (default 1000).\n", e);
            fflush(stderr);
        }
    }
    return cached;
}

static inline uint64_t ggml_moe_diag_report_every(void) {
    static uint64_t cached = 0;
    if (cached == 0) {
        const char * e = getenv("LLAMA_MMID_DIAG_REPORT_EVERY");
        long long v = (e != NULL && e[0] != '\0') ? atoll(e) : 0;
        cached = (v > 0) ? (uint64_t)v : 1000;
    }
    return cached;
}

// Separate cadence for sched-level reports. MMID fires once per op (~144/ubatch
// in qwen3.5-122b at 48 MoE layers × 3 MMIDs), so aggregating is necessary.
// compute_splits fires once per graph compute (one per ubatch). Defaulting to
// 1 gives a summary line per ubatch — exactly the grain the PP investigation
// wants. For sustained decode this will spam; tune via env when needed.
static inline uint64_t ggml_moe_diag_sched_report_every(void) {
    static uint64_t cached = 0;
    if (cached == 0) {
        const char * e = getenv("LLAMA_SCHED_DIAG_REPORT_EVERY");
        long long v = (e != NULL && e[0] != '\0') ? atoll(e) : 0;
        cached = (v > 0) ? (uint64_t)v : 1;
    }
    return cached;
}

static inline uint64_t ggml_moe_diag_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

struct ggml_moe_diag_stat {
    const char * name;
    uint64_t     count;
    uint64_t     total_ns;
    uint64_t     max_ns;
    uint64_t     total_bytes;    // optional (memset / copy bandwidth)
};

static inline void ggml_moe_diag_add(
        struct ggml_moe_diag_stat * s, uint64_t ns, uint64_t bytes) {
    s->count      += 1;
    s->total_ns   += ns;
    s->total_bytes+= bytes;
    if (ns > s->max_ns) s->max_ns = ns;
}

static inline void ggml_moe_diag_report(const struct ggml_moe_diag_stat * s) {
    if (s->count == 0) return;
    const double total_ms = (double)s->total_ns / 1e6;
    const double avg_us   = (double)s->total_ns / ((double)s->count * 1000.0);
    const double max_us   = (double)s->max_ns / 1000.0;
    if (s->total_bytes > 0) {
        const double gb_per_s =
            ((double)s->total_bytes / (1024.0*1024.0*1024.0))
            / ((double)s->total_ns / 1e9);
        fprintf(stderr,
            "WARN [mmid-diag]   %-32s  n=%-8llu  total=%9.2fms  avg=%8.1fus  "
            "max=%8.1fus  bw=%6.2fGB/s\n",
            s->name, (unsigned long long)s->count,
            total_ms, avg_us, max_us, gb_per_s);
    } else {
        fprintf(stderr,
            "WARN [mmid-diag]   %-32s  n=%-8llu  total=%9.2fms  avg=%8.1fus  "
            "max=%8.1fus\n",
            s->name, (unsigned long long)s->count,
            total_ms, avg_us, max_us);
    }
    fflush(stderr);
}

static inline void ggml_moe_diag_reset(struct ggml_moe_diag_stat * s) {
    s->count = 0;
    s->total_ns = 0;
    s->max_ns = 0;
    s->total_bytes = 0;
}

#ifdef __cplusplus
}
#endif
