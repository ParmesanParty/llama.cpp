#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

// Returns true if the buffer type is the CUDA host (pinned) buffer type.
// Used by ggml_backend_cuda_cpy_tensor_async to detect the pinned-host case
// for the async D2H/H2D fast path.
GGML_BACKEND_API bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// Parmesan: expose the default compute stream for a CUDA backend as an
// opaque `void *`, so external code (e.g., the MoE hot cache manager) can
// record/wait on CUDA events against it without having to include
// cuda_runtime.h from the consuming TU. Returns nullptr if the backend is
// not a CUDA backend. Callers cast the result to `cudaStream_t`.
//
// NOTE: the return type is intentionally `void *`, not `cudaStream_t`,
// because the native CUDA backend build does not define `GGML_USE_CUDA`
// (that macro is only set for MUSA/HIP translation builds) and ggml-cuda.h
// is transitively included by many non-CUDA TUs that must not depend on
// cuda_runtime.h.
GGML_BACKEND_API void * ggml_backend_cuda_get_stream(ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
