/**
 * engine/include/distribox/worker_engine.h — Worker compute engine C ABI
 *
 * This is the boundary between Go (worker agent) and C/C++ (compute kernels).
 * Every backend (OpenCL, CUDA, CPU) implements this interface.
 *
 * The engine manages a local OpenCL context, compiles kernels, and
 * executes NDRange commands on behalf of the worker agent.
 */

#ifndef DISTRIBOX_WORKER_ENGINE_H
#define DISTRIBOX_WORKER_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Opaque handles ─────────────────────────────────────

typedef struct engine_ctx engine_ctx_t;
typedef struct engine_buffer engine_buffer_t;
typedef struct engine_program engine_program_t;
typedef struct engine_kernel engine_kernel_t;

// ── Engine lifecycle ───────────────────────────────────

/**
 * Initialize the compute engine.
 * @param preferred_backend  "opencl", "cuda", or "cpu" (NULL = auto-detect)
 * @return engine context, or NULL on failure
 */
engine_ctx_t *engine_init(const char *preferred_backend);

/**
 * Shut down the engine and free all resources.
 */
void engine_destroy(engine_ctx_t *ctx);

/**
 * Get engine backend name ("OpenCL", "CUDA", "CPU")
 */
const char *engine_backend_name(engine_ctx_t *ctx);

/**
 * Get device info string for capability reporting.
 * Returns JSON with: vendor, model, vram_mb, compute_units, clock_mhz,
 * opencl_version, benchmark_gflops.
 */
const char *engine_get_device_info(engine_ctx_t *ctx);

// ── Buffer management ──────────────────────────────────

/**
 * Allocate a buffer on the local device.
 * @param size  Buffer size in bytes
 * @param flags  cl_mem_flags (CL_MEM_READ_ONLY, etc.)
 * @param data   Initial data (can be NULL)
 * @return buffer handle, or NULL on failure
 */
engine_buffer_t *engine_buffer_create(engine_ctx_t *ctx, uint64_t size,
                                       uint32_t flags, const void *data);

/**
 * Write data to a buffer.
 */
int engine_buffer_write(engine_ctx_t *ctx, engine_buffer_t *buf,
                         uint64_t offset, uint64_t size, const void *data);

/**
 * Read data from a buffer.
 */
int engine_buffer_read(engine_ctx_t *ctx, engine_buffer_t *buf,
                        uint64_t offset, uint64_t size, void *data);

/**
 * Release a buffer.
 */
void engine_buffer_release(engine_ctx_t *ctx, engine_buffer_t *buf);

/**
 * Get buffer size.
 */
uint64_t engine_buffer_get_size(engine_buffer_t *buf);

// ── Program compilation ────────────────────────────────

/**
 * Create a program from OpenCL C source code.
 * @param source       OpenCL C kernel source
 * @param source_len   Length of source string
 * @param options      Compile options (e.g., "-cl-mad-enable -cl-fast-relaxed-math")
 * @return program handle, or NULL on failure
 */
engine_program_t *engine_program_create_from_source(engine_ctx_t *ctx,
                                                     const char *source,
                                                     uint64_t source_len,
                                                     const char *options);

/**
 * Create a program from cached binary.
 * @param binary       Pre-compiled binary
 * @param binary_len   Length of binary data
 */
engine_program_t *engine_program_create_from_binary(engine_ctx_t *ctx,
                                                     const uint8_t *binary,
                                                     uint64_t binary_len);

/**
 * Build (compile) a program for the local device.
 * @param prog     Program to build
 * @param options  Compile options
 * @param build_log  Output: compilation log (caller frees)
 * @return 0 on success, -1 on failure
 */
int engine_program_build(engine_ctx_t *ctx, engine_program_t *prog,
                          const char *options, char **build_log);

/**
 * Get compiled binary for caching.
 * @param out_len  Output: length of binary data
 * @return binary data (caller frees)
 */
uint8_t *engine_program_get_binary(engine_ctx_t *ctx, engine_program_t *prog,
                                    uint64_t *out_len);

/**
 * Release a program.
 */
void engine_program_release(engine_ctx_t *ctx, engine_program_t *prog);

// ── Kernel management ──────────────────────────────────

/**
 * Create a kernel object from a program.
 * @param prog         Compiled program
 * @param kernel_name  Name of the kernel function
 * @return kernel handle, or NULL on failure
 */
engine_kernel_t *engine_kernel_create(engine_ctx_t *ctx,
                                       engine_program_t *prog,
                                       const char *kernel_name);

/**
 * Set a kernel argument.
 * @param index     Argument index
 * @param size      Size of argument in bytes
 * @param value     Pointer to argument data (for scalars) or buffer handle (for buffers)
 * @param is_buffer true if value points to an engine_buffer_t*
 */
int engine_kernel_set_arg(engine_ctx_t *ctx, engine_kernel_t *kernel,
                           uint32_t index, uint64_t size,
                           const void *value, bool is_buffer);

/**
 * Release a kernel.
 */
void engine_kernel_release(engine_ctx_t *ctx, engine_kernel_t *kernel);

// ── Execution ──────────────────────────────────────────

/**
 * Execute an NDRange kernel on the local device.
 *
 * @param ctx            Engine context
 * @param kernel         Kernel to execute
 * @param work_dim       Number of dimensions (1-3)
 * @param global_size    Global work size [work_dim]
 * @param global_offset  Global offset [work_dim] (can be NULL)
 * @param local_size     Local work group size [work_dim] (can be NULL)
 * @param output_buffers Array of buffer handles to read back after execution
 * @param num_outputs    Number of output buffers
 * @return 0 on success, -1 on failure
 *
 * This is the core function. It executes the kernel on this worker's
 * portion of the NDRange and stores results in the output buffers.
 */
int engine_execute_ndrange(engine_ctx_t *ctx,
                            engine_kernel_t *kernel,
                            uint32_t work_dim,
                            const uint64_t *global_size,
                            const uint64_t *global_offset,
                            const uint64_t *local_size,
                            engine_buffer_t **output_buffers,
                            uint32_t num_outputs);

/**
 * Wait for all pending operations on the device to complete.
 */
int engine_finish(engine_ctx_t *ctx);

// ── Micro-benchmark ────────────────────────────────────

/**
 * Run a quick compute benchmark to measure empirical GFLOPS.
 * @return measured GFLOPS (single precision)
 */
double engine_run_micro_benchmark(engine_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // DISTRIBOX_WORKER_ENGINE_H
