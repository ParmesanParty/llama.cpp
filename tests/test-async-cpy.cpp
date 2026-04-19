// test-async-cpy: regression test for ggml_backend_cuda_cpy_tensor_async's
// new CUDA↔pinned-host fast paths (D2H Case 2 and H2D Case 3), validated
// via ggml_backend_wait_input_ready and (for H2D) a CUDA round-trip.
//
// Two scenarios:
//
// 1) D2H — allocate a CUDA src tensor, write a known pattern to it via
//    ggml_backend_tensor_set (sync H2D), allocate a pinned-host dst tensor,
//    call cuda_backend->iface.cpy_tensor_async DIRECTLY (bypassing the
//    public wrapper to ensure we hit the D2H code path), call
//    ggml_backend_wait_input_ready, and verify dst matches src byte-for-byte.
//
// 2) H2D — allocate a pinned-host src tensor with a known pattern, allocate
//    a CUDA dst tensor, call cuda_backend->iface.cpy_tensor_async DIRECTLY
//    (the H2D path issues on the compute stream so no event coordination).
//    Read the data back via ggml_backend_tensor_get (sync D2H) and verify.
//
// Calling iface.cpy_tensor_async DIRECTLY is intentional. The public wrapper
// ggml_backend_tensor_copy_async tries dst-side iface first; for the D2H
// (cuda→cpu) case, dst is CPU and has no async impl, so the wrapper falls
// through to the sync path even after the Task 3 Step 4 fix (which adds
// src-side fallback to the wrapper). The fix makes the wrapper "correct"
// but for testing we want to be sure we're hitting the new code path
// without ambiguity.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
// Internal header — same convention as test-moe-hot-cache.cpp's relative
// path includes; ggml-backend-impl.h gives us direct access to the iface
// struct so we can call cpy_tensor_async without going through the public
// wrapper.
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "ASSERTION FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        std::exit(1); \
    } \
} while (0)

static void test_d2h(ggml_backend_t cuda_backend) {
    // Allocate a CUDA tensor (src).
    constexpr int N = 1024;  // 4 KB of float — small enough to be fast, large enough that DMA matters
    ggml_backend_buffer_type_t cuda_buft = ggml_backend_cuda_buffer_type(0);
    ggml_init_params src_params = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * src_ctx = ggml_init(src_params);
    ggml_tensor * src = ggml_new_tensor_1d(src_ctx, GGML_TYPE_F32, N);
    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors_from_buft(src_ctx, cuda_buft);
    TEST_ASSERT(src_buf != nullptr);

    // Initialize src with a known pattern: src[i] = i * 1.5f
    std::vector<float> src_host(N);
    for (int i = 0; i < N; ++i) src_host[i] = i * 1.5f;
    ggml_backend_tensor_set(src, src_host.data(), 0, N * sizeof(float));

    // Allocate a pinned-host tensor (dst) on cuda_host_buffer_type.
    ggml_backend_buffer_type_t pinned_buft = ggml_backend_cuda_host_buffer_type();
    TEST_ASSERT(ggml_backend_buft_is_cuda_host(pinned_buft));

    ggml_init_params dst_params = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * dst_ctx = ggml_init(dst_params);
    ggml_tensor * dst = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_F32, N);
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors_from_buft(dst_ctx, pinned_buft);
    TEST_ASSERT(dst_buf != nullptr);

    // Initialize dst to a poison pattern so we can detect a missed copy.
    std::memset(dst->data, 0xCC, N * sizeof(float));

    // Call the CUDA backend's iface DIRECTLY (bypass the public wrapper).
    // Pass cuda_backend as both src and dst — the iface impl figures out the
    // direction from the buffer types.
    bool handled = cuda_backend->iface.cpy_tensor_async(cuda_backend, /* unused for src-side */ cuda_backend, src, dst);
    TEST_ASSERT(handled && "D2H fast path should accept this CUDA→pinned-host copy");

    // **Idempotency check:** call cpy_tensor_async AGAIN on the same dst.
    // This simulates the prefetch + inline-issue pattern from the scheduler.
    // The CUDA backend's D2H Case 2 branch should detect the existing entry
    // in pending_d2h_events (keyed by dst->data) and early-return true
    // WITHOUT issuing a new copy. If this returns false, or if it issues a
    // new copy and creates a new event, the prefetch design is broken — the
    // second event would capture additional compute stream state and the
    // consumer's drain would wait for it, defeating the parallelism. This
    // test is the regression guard for that bug class.
    bool handled_2nd = cuda_backend->iface.cpy_tensor_async(cuda_backend, cuda_backend, src, dst);
    TEST_ASSERT(handled_2nd && "Second cpy_tensor_async on same dst should idempotently return true");

    // Drain pending events. The CUDA backend's wait_input_ready will look up
    // dst by data pointer and cudaEventSynchronize.
    ggml_backend_wait_input_ready(cuda_backend, dst);

    // Verify dst matches src exactly.
    const float * dst_data = (const float *) dst->data;
    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(dst_data[i] == src_host[i]);
    }
    fprintf(stderr, "test-async-cpy [D2H + idempotent re-issue]: %d floats verified\n", N);

    // Cleanup (buffers first, then contexts).
    ggml_backend_buffer_free(src_buf);
    ggml_backend_buffer_free(dst_buf);
    ggml_free(src_ctx);
    ggml_free(dst_ctx);
}

static void test_h2d(ggml_backend_t cuda_backend) {
    // Allocate a pinned-host src tensor.
    constexpr int N = 1024;
    ggml_backend_buffer_type_t pinned_buft = ggml_backend_cuda_host_buffer_type();
    ggml_init_params src_params = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * src_ctx = ggml_init(src_params);
    ggml_tensor * src = ggml_new_tensor_1d(src_ctx, GGML_TYPE_F32, N);
    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors_from_buft(src_ctx, pinned_buft);
    TEST_ASSERT(src_buf != nullptr);

    // Pinned src is host-accessible — write directly.
    float * src_data = (float *) src->data;
    for (int i = 0; i < N; ++i) src_data[i] = i * 0.25f;

    // Allocate a CUDA dst tensor.
    ggml_backend_buffer_type_t cuda_buft = ggml_backend_cuda_buffer_type(0);
    ggml_init_params dst_params = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * dst_ctx = ggml_init(dst_params);
    ggml_tensor * dst = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_F32, N);
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors_from_buft(dst_ctx, cuda_buft);
    TEST_ASSERT(dst_buf != nullptr);

    // Call the CUDA backend's iface DIRECTLY for the H2D path.
    // The H2D path is detected when src is on a cuda_host_buffer_type and
    // dst is on a regular cuda_buffer_type. It issues cudaMemcpyAsync on
    // the compute stream (FIFO with subsequent compute_async ops).
    bool handled = cuda_backend->iface.cpy_tensor_async(cuda_backend, cuda_backend, src, dst);
    TEST_ASSERT(handled && "H2D fast path should accept this pinned-host→CUDA copy");

    // The H2D doesn't need an event drain because it's on the compute stream
    // FIFO-ordered with the next op. To verify the data made it onto the
    // GPU, read it back via ggml_backend_tensor_get (sync D2H) and check.
    // The sync D2H synchronizes the compute stream first, so the H2D will
    // have completed by the time we read.
    std::vector<float> dst_host(N);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(dst_host[i] == i * 0.25f);
    }
    fprintf(stderr, "test-async-cpy [H2D]: %d floats verified\n", N);

    // Cleanup.
    ggml_backend_buffer_free(src_buf);
    ggml_backend_buffer_free(dst_buf);
    ggml_free(src_ctx);
    ggml_free(dst_ctx);
}

int main() {
    // Initialize the CUDA backend. If CUDA isn't available, skip with a
    // friendly message rather than failing.
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (cuda_backend == nullptr) {
        fprintf(stderr, "test-async-cpy: SKIP (CUDA backend not available)\n");
        return 0;
    }

    test_d2h(cuda_backend);
    test_h2d(cuda_backend);

    ggml_backend_free(cuda_backend);

    fprintf(stderr, "test-async-cpy: PASS\n");
    return 0;
}
