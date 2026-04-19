# llama.cpp — parmesan fork

> **Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).** For the original project — build system, backends, model support, quantization, and full documentation — visit the upstream repository. This README describes only what this fork adds on top of upstream `master`.

A customization of llama.cpp focused on running large sparse mixture-of-experts models on a single consumer GPU, plus a set of server and web-UI additions for long-running hosted inference.

## Headline addition: MoE hot expert compute cache

Running a model like Qwen3.5-122B-A10B on a single 24GB GPU requires offloading the expert FFN weights to system RAM (`-ot exps=CPU`). Decode then becomes CPU-bound — the GPU sits idle ~86% of the time pulling 8-of-256 expert rows across the PCIe boundary per layer.

This fork adds a **per-layer hot expert cache** that pins the top-K most-activated experts in VRAM and runs them on the GPU in parallel with the remaining experts on CPU. At K=42 on Qwen3.5-122B-A10B-Q4_K_L, this moves decode from CPU-bound toward balanced-parallel, with measurable throughput gains that scale with the cache hit rate.

How it works:

- The graph emits **two MUL_MAT_ID ops per projection per MoE layer** — a hot branch on GPU and a cold branch on CPU — with expert ids anchored so each branch masks out the other's experts via a sentinel-skip flag (`GGML_MUL_MAT_ID_FLAG_SENTINEL`).
- The cold branch is fused into a single custom CPU op (`llama_moe_fused_cold_compute`) that combines up + gate + SwiGLU + down into one dispatch with 3 internal OpenMP barriers instead of the 4-op graph's ~8.
- Async cross-backend copies on a dedicated CUDA stream, with look-ahead prefetch, let the scheduler overlap GPU hot compute with CPU cold compute rather than serializing behind cross-backend input transfers.
- Hot set is chosen by a per-layer tumbling-window counter over routing frequencies, rebalanced every N decodes (default 60) with hysteresis to suppress thrashing.
- FILLING mode eagerly bootstraps the cache over the first ~8 decodes so the initial prefill doesn't have to wait for a full rebalance interval.

Config vars (optional; feature is off by default):

| Env | Default | Purpose |
|-----|---------|---------|
| `LLAMA_ARG_MOE_HOT_K` | `0` (disabled) | Hot experts pinned per layer. Typical: 32-56. |
| `LLAMA_ARG_MOE_HOT_REBALANCE_INTERVAL` | `40` | Decode steps between rebalance ticks. |
| `LLAMA_MOE_HOT_PP_BYPASS_N_TOKENS` | `0` (disabled) | Skip the dual-path for ubatches with `n_tokens > N`. Needed because the fused cold kernel is optimized for decode (its inner loop is per-row `vec_dot`, not batched matmul); at prefill ubatch sizes the dual-path incurs a ~3.5x PP throughput slowdown vs `K=0`. Setting to 64 keeps decode (including batched/speculative up to 64) on the dual-path while PP falls through to the standard single-path. |

Feature engages only when:
- A partial-offload MoE preset is active (`-ot exps=CPU` or equivalent).
- `LLAMA_ARG_MOE_HOT_K > 0`.
- The model's MoE forward pass has been wired for it (currently `qwen35moe.cpp` is the only target).

Source: `src/llama-moe-hot-cache.{h,cpp}`, `src/llama-moe-fused-cold.{h,cpp}`, dual-path emission in `src/llama-graph.cpp`, MUL_MAT_ID sentinel-skip in `ggml/src/ggml-cpu/ggml-cpu.c` and `ggml/src/ggml-cuda/mmid.cu`, async cross-backend copy infrastructure under `ggml/src/ggml-backend*` and `ggml/src/ggml-cuda/`.

### Requirements and caveats

- **CUDA only** for the hot branch. The cross-backend async copy infrastructure is stubbed as NULL on non-CUDA backends (Metal, Vulkan, SYCL, etc.), so the cache refuses to engage without CUDA.
- **OpenMP** is required to build `llama-moe-fused-cold.cpp` (linked into the `llama` target via `src/CMakeLists.txt` when `GGML_OPENMP=ON`, which is the default).
- **One model family wired** — Qwen3.5 MoE (`qwen35moe.cpp`). Mixtral, DeepSeek, Qwen3MoE, Llama4 are all compatible with the infrastructure but their forward passes haven't been opted in.
- **Routing telemetry** writes JSONL records when `LLAMA_MOE_HOT_TELEMETRY=<dir>` is set. Off by default; enable only when diagnosing cache behavior — the per-token record path adds measurable overhead.

## Other server additions

### `POST /sleep` and `POST /keepalive`

Explicit lifecycle endpoints for a reverse-proxy-driven sleep/wake loop. `/sleep` triggers an immediate model destroy (blocks until complete, drops RSS to ~400MB). `/keepalive` wakes a sleeping model (blocks until loaded). Used in deployments where the GPU is cooperatively shared with other services and the model needs to yield VRAM without tearing down the process.

### `enable_thinking` as a top-level request parameter

`/v1/chat/completions` accepts `enable_thinking: true|false` at the top level of the request body. The server merges it into `chat_template_kwargs` before template rendering, so clients can toggle thinking mode per-request without managing template kwargs themselves. Upstream supports this only via the more verbose `chat_template_kwargs` path.

### System prompt injection with hot-reload

`LLAMA_ARG_SYSTEM_PROMPT_FILE=<path>` makes the server prepend a system message to any completion request that didn't supply one. The file is `stat`'d on each request and re-read on mtime change, so edits to the file take effect without a restart. The current date is injected for temporal grounding.

### Model alias from GGUF metadata

`/v1/models` returns a model name derived from `general.name` in the loaded GGUF's metadata (falling back to the resolved filename stem). Upstream returns the raw file path, which leaks local filesystem structure and is unstable across symlink swaps.

## Web UI additions (Parmesan Chat)

The bundled `tools/server/webui` SvelteKit app has been customized:

- **Branding**: renamed to Parmesan Chat with a custom favicon.
- **Thinking toggle** per chat, persisted in IndexedDB.
- **Condensed rendering for intermediate tool-call steps** so multi-step tool sessions don't flood the message list.
- **Context compaction banner** for client-side rendering of proxy-driven summarization.
- **Retraction markers** for mid-stream message revisions (e.g. config-verification nudges).
- **Streaming citation resolver** that incrementally rewrites `[N]` citations to markdown links as tokens arrive.
- **Model switcher** for proxy-managed preset switching, driven by SSE push notifications (`GET /api/events`).
- **Mobile fixes**: sidebar tap-to-navigate, iOS Safari keyboard-open viewport bug.
- **Thinking-mode sampling overrides** in settings, synced with the server via `/props`.
- **Cached-prompt-token count** in chat statistics.

All client-side only; upstream-compatible with the core `/v1/chat/completions` SSE contract when the client doesn't opt into typed events (`X-Stream-Features` header).

## Building

The fork builds with the upstream build system — no extra steps required for the kernel work. OpenMP is picked up automatically when `GGML_OPENMP=ON` (the default).

```
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<your-arch>  # e.g. 89 for Ada Lovelace
cmake --build build --config Release -j
```

For `llama-server` with the web UI bundled, build the UI first:

```
cd tools/server/webui && npm install && npm run build && cd -
cmake --build build --target llama-server -j
```

See upstream's [build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for backend-specific details.

## Running with the hot cache

Minimal example for a partial-offload MoE model:

```
llama-server \
  --model /path/to/model.gguf \
  --n-gpu-layers 99 \
  --override-tensor exps=CPU \
  --ctx-size 131072 \
  --ubatch-size 3072 \
  --batch-size 3072 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on
```

Plus these environment variables to engage the hot cache:

```
LLAMA_ARG_MOE_HOT_K=42
LLAMA_ARG_MOE_HOT_REBALANCE_INTERVAL=60
LLAMA_MOE_HOT_PP_BYPASS_N_TOKENS=64
```

K sizing is a VRAM tradeoff against KV cache. K=42 for Qwen3.5-122B-A10B-Q4_K_L uses ~9.9 GB of the 24GB budget; the remainder goes to KV (q8_0 recommended) and the ubatch activation buffer.

## Tests

Fork-specific tests:

```
ctest --test-dir build -R 'test-moe-hot-cache|test-async-cpy'
build/bin/test-backend-ops -o MUL_MAT_ID   # covers sentinel-skip
```

Standard upstream tests are unaffected.

## Staying in sync with upstream

This branch is designed to be rebased onto `origin/master` regularly:

```
git fetch origin
git rebase origin/master
```

Fork-specific commits are semantically partitioned so they rebase cleanly in the common case:

- `ggml: MoE hot cache infrastructure ops` — backend ifaces, CUDA async copies, MUL_MAT_ID sentinel skip
- `llama: MoE hot cache core and routing telemetry`
- `llama-graph: dual-path MUL_MAT_ID and fused cold op emission`
- `llama: fused cold CPU MoE FFN kernel`
- `models/qwen35moe: wire MoE hot cache into forward pass`
- `tests: MoE hot cache, async cross-backend copy, MUL_MAT_ID sentinel skip`
- `server: POST /sleep and /keepalive`
- `server: enable_thinking top-level parameter`
- `server: model alias from GGUF metadata`
- `server: system prompt injection with hot-reload`
- `webui: …` (Parmesan Chat customizations)

Conflicts when rebasing are most likely in `ggml/src/ggml-backend.cpp`, `src/llama-graph.cpp`, and `src/llama-context.cpp` when upstream refactors those areas. Scheduler changes upstream are the common source — our dual-path relies on the scheduler's split behavior.

## Issues and contributing

For issues or features that are specific to the additions in this fork, open an issue on this repo. For core llama.cpp questions, please use the [upstream repo](https://github.com/ggml-org/llama.cpp).

If you build on this fork — porting the hot cache to other MoE model families, adding rebalance strategies, extending the telemetry, improving the fused cold kernel's PP path — please file a PR.

## License and attribution

This fork inherits the upstream MIT license. All original work is copyright the upstream contributors; the additions documented here are released under the same MIT license. See `LICENSE`.

llama.cpp is created and maintained by the [ggml-org](https://github.com/ggml-org) community. This fork would not exist without their work.
