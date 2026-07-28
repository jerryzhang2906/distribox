/**
 * engine/src/engine_opencl.c — OpenCL compute backend
 *
 * Implements the worker engine interface using OpenCL.
 * This is the primary backend: it uses the local GPU via OpenCL if available,
 * or falls back to CPU-based OpenCL (PoCL, Intel OpenCL CPU runtime).
 */

#include "distribox/worker_engine.h"

#ifdef HAS_OPENCL
#include <CL/cl.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ── Internal structures ────────────────────────────────

struct engine_ctx {
#ifdef HAS_OPENCL
    cl_platform_id platform;
    cl_device_id   device;
    cl_context     context;
    cl_command_queue queue;
    bool           has_gpu;
#endif
    char           backend_name[32];
    char           device_json[1024];
};

struct engine_buffer {
    uint64_t size;
    uint32_t flags;
#ifdef HAS_OPENCL
    cl_mem   cl_buffer;
#endif
    void    *host_ptr;   // For CPU fallback
    bool     is_opencl;
};

struct engine_program {
#ifdef HAS_OPENCL
    cl_program cl_program;
#endif
    char   *source;
    uint64_t source_len;
    bool    compiled;
};

struct engine_kernel {
#ifdef HAS_OPENCL
    cl_kernel cl_kernel;
#endif
    char   name[256];
};

// ── Init / Destroy ─────────────────────────────────────

engine_ctx_t *engine_init(const char *preferred_backend) {
    engine_ctx_t *ctx = (engine_ctx_t *)calloc(1, sizeof(engine_ctx_t));
    if (!ctx) return NULL;

    bool use_opencl = (preferred_backend == NULL ||
                       strcmp(preferred_backend, "opencl") == 0 ||
                       strcmp(preferred_backend, "auto") == 0);

#ifdef HAS_OPENCL
    if (use_opencl) {
        cl_int err;
        cl_uint num_platforms;

        err = clGetPlatformIDs(1, &ctx->platform, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            // No OpenCL platform found — fall through to CPU
            use_opencl = false;
        } else {
            // Get GPU device if available, otherwise CPU
            err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_GPU, 1,
                                 &ctx->device, NULL);
            if (err == CL_SUCCESS) {
                ctx->has_gpu = true;
            } else {
                // Fall back to CPU OpenCL device
                err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_CPU, 1,
                                     &ctx->device, NULL);
                if (err != CL_SUCCESS) {
                    use_opencl = false;
                } else {
                    ctx->has_gpu = false;
                }
            }

            if (use_opencl) {
                ctx->context = clCreateContext(NULL, 1, &ctx->device,
                                                NULL, NULL, &err);
                if (err == CL_SUCCESS) {
                    ctx->queue = clCreateCommandQueue(ctx->context, ctx->device,
                                                      0, &err);
                    if (err == CL_SUCCESS) {
                        strncpy(ctx->backend_name, "OpenCL", sizeof(ctx->backend_name)-1);

                        // Build device info JSON
                        char dname[256], dvendor[128], dversion[64];
                        cl_ulong vram;
                        cl_uint cu, freq;
                        clGetDeviceInfo(ctx->device, CL_DEVICE_NAME,
                                        sizeof(dname), dname, NULL);
                        clGetDeviceInfo(ctx->device, CL_DEVICE_VENDOR,
                                        sizeof(dvendor), dvendor, NULL);
                        clGetDeviceInfo(ctx->device, CL_DEVICE_GLOBAL_MEM_SIZE,
                                        sizeof(vram), &vram, NULL);
                        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                        sizeof(cu), &cu, NULL);
                        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                        sizeof(freq), &freq, NULL);
                        clGetDeviceInfo(ctx->device, CL_DEVICE_VERSION,
                                        sizeof(dversion), dversion, NULL);

                        snprintf(ctx->device_json, sizeof(ctx->device_json),
                            "{\"vendor\":\"%s\",\"model\":\"%s\","
                            "\"vram_mb\":%llu,\"compute_units\":%u,"
                            "\"clock_mhz\":%u,\"opencl_version\":\"%s\","
                            "\"type\":\"%s\"}",
                            dvendor, dname,
                            (unsigned long long)(vram/(1024*1024)), cu, freq,
                            dversion,
                            ctx->has_gpu ? "GPU" : "CPU");

                        return ctx;
                    }
                }
            }
        }
    }
#endif

    // ── CPU fallback ────────────────────────────────────
    strncpy(ctx->backend_name, "CPU", sizeof(ctx->backend_name)-1);
    snprintf(ctx->device_json, sizeof(ctx->device_json),
        "{\"vendor\":\"Generic\",\"model\":\"CPU Backend\","
        "\"vram_mb\":0,\"compute_units\":%d,\"clock_mhz\":0,"
        "\"opencl_version\":\"\",\"type\":\"CPU\"}", 4);
    return ctx;
}

void engine_destroy(engine_ctx_t *ctx) {
    if (!ctx) return;
#ifdef HAS_OPENCL
    if (ctx->queue) clReleaseCommandQueue(ctx->queue);
    if (ctx->context) clReleaseContext(ctx->context);
#endif
    free(ctx);
}

const char *engine_backend_name(engine_ctx_t *ctx) {
    return ctx ? ctx->backend_name : "unknown";
}

const char *engine_get_device_info(engine_ctx_t *ctx) {
    return ctx ? ctx->device_json : "{}";
}

// ── Buffer management ──────────────────────────────────

engine_buffer_t *engine_buffer_create(engine_ctx_t *ctx, uint64_t size,
                                       uint32_t flags, const void *data) {
    if (!ctx) return NULL;

    engine_buffer_t *buf = (engine_buffer_t *)calloc(1, sizeof(engine_buffer_t));
    if (!buf) return NULL;

    buf->size = size;
    buf->flags = flags;

#ifdef HAS_OPENCL
    if (ctx->context) {
        cl_int err;
        buf->cl_buffer = clCreateBuffer(ctx->context, (cl_mem_flags)flags,
                                         (size_t)size, NULL, &err);
        if (err == CL_SUCCESS) {
            buf->is_opencl = true;

            // If initial data provided, write it
            if (data) {
                clEnqueueWriteBuffer(ctx->queue, buf->cl_buffer, CL_TRUE,
                                     0, (size_t)size, data, 0, NULL, NULL);
            }
            return buf;
        }
    }
#endif

    // CPU fallback
    buf->is_opencl = false;
    buf->host_ptr = malloc((size_t)size);
    if (buf->host_ptr && data) {
        memcpy(buf->host_ptr, data, (size_t)size);
    }

    return buf;
}

int engine_buffer_write(engine_ctx_t *ctx, engine_buffer_t *buf,
                         uint64_t offset, uint64_t size, const void *data) {
    if (!ctx || !buf || !data) return -1;

#ifdef HAS_OPENCL
    if (buf->is_opencl) {
        return clEnqueueWriteBuffer(ctx->queue, buf->cl_buffer, CL_TRUE,
                                     (size_t)offset, (size_t)size, data,
                                     0, NULL, NULL) == CL_SUCCESS ? 0 : -1;
    }
#endif

    if (buf->host_ptr) {
        memcpy((char*)buf->host_ptr + offset, data, (size_t)size);
        return 0;
    }
    return -1;
}

int engine_buffer_read(engine_ctx_t *ctx, engine_buffer_t *buf,
                        uint64_t offset, uint64_t size, void *data) {
    if (!ctx || !buf || !data) return -1;

#ifdef HAS_OPENCL
    if (buf->is_opencl) {
        return clEnqueueReadBuffer(ctx->queue, buf->cl_buffer, CL_TRUE,
                                    (size_t)offset, (size_t)size, data,
                                    0, NULL, NULL) == CL_SUCCESS ? 0 : -1;
    }
#endif

    if (buf->host_ptr) {
        memcpy(data, (char*)buf->host_ptr + offset, (size_t)size);
        return 0;
    }
    return -1;
}

void engine_buffer_release(engine_ctx_t *ctx, engine_buffer_t *buf) {
    if (!buf) return;
    (void)ctx;
#ifdef HAS_OPENCL
    if (buf->is_opencl && buf->cl_buffer) {
        clReleaseMemObject(buf->cl_buffer);
    }
#endif
    if (buf->host_ptr) free(buf->host_ptr);
    free(buf);
}

uint64_t engine_buffer_get_size(engine_buffer_t *buf) {
    return buf ? buf->size : 0;
}

// ── Program ────────────────────────────────────────────

engine_program_t *engine_program_create_from_source(engine_ctx_t *ctx,
                                                     const char *source,
                                                     uint64_t source_len,
                                                     const char *options) {
    if (!ctx || !source) return NULL;
    (void)options;

    engine_program_t *prog = (engine_program_t *)calloc(1, sizeof(engine_program_t));
    if (!prog) return NULL;

    prog->source = (char *)malloc(source_len + 1);
    if (!prog->source) { free(prog); return NULL; }
    memcpy(prog->source, source, source_len);
    prog->source[source_len] = '\0';
    prog->source_len = source_len;
    prog->compiled = false;

#ifdef HAS_OPENCL
    if (ctx->context) {
        cl_int err;
        const char *src_ptr = prog->source;
        size_t src_len = (size_t)source_len;
        prog->cl_program = clCreateProgramWithSource(ctx->context, 1,
                                                      &src_ptr, &src_len, &err);
        if (err != CL_SUCCESS) {
            free(prog->source);
            free(prog);
            return NULL;
        }
    }
#endif

    return prog;
}

engine_program_t *engine_program_create_from_binary(engine_ctx_t *ctx,
                                                     const uint8_t *binary,
                                                     uint64_t binary_len) {
    if (!ctx || !binary) return NULL;

    engine_program_t *prog = (engine_program_t *)calloc(1, sizeof(engine_program_t));
    if (!prog) return NULL;

#ifdef HAS_OPENCL
    if (ctx->context) {
        cl_int err, binary_status;
        const unsigned char *bin_ptr = binary;
        size_t bin_len = (size_t)binary_len;
        prog->cl_program = clCreateProgramWithBinary(ctx->context, 1,
                                                      &ctx->device, &bin_len,
                                                      &bin_ptr, &binary_status, &err);
        if (err != CL_SUCCESS || binary_status != CL_SUCCESS) {
            free(prog);
            return NULL;
        }
        prog->compiled = true;
    }
#else
    (void)binary_len;
    prog->compiled = true;
#endif

    return prog;
}

int engine_program_build(engine_ctx_t *ctx, engine_program_t *prog,
                          const char *options, char **build_log) {
    if (!ctx || !prog) return -1;

#ifdef HAS_OPENCL
    if (prog->cl_program && !prog->compiled) {
        cl_int err = clBuildProgram(prog->cl_program, 1, &ctx->device,
                                     options, NULL, NULL);

        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(prog->cl_program, ctx->device,
                              CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (build_log && log_size > 1) {
            *build_log = (char *)malloc(log_size);
            clGetProgramBuildInfo(prog->cl_program, ctx->device,
                                  CL_PROGRAM_BUILD_LOG, log_size, *build_log, NULL);
        }

        if (err != CL_SUCCESS) {
            prog->compiled = false;
            return -1;
        }
    }
#endif

    prog->compiled = true;
    return 0;
}

void engine_program_release(engine_ctx_t *ctx, engine_program_t *prog) {
    if (!prog) return;
    (void)ctx;
#ifdef HAS_OPENCL
    if (prog->cl_program) clReleaseProgram(prog->cl_program);
#endif
    if (prog->source) free(prog->source);
    free(prog);
}

// ── Kernel ─────────────────────────────────────────────

engine_kernel_t *engine_kernel_create(engine_ctx_t *ctx,
                                       engine_program_t *prog,
                                       const char *kernel_name) {
    if (!ctx || !prog || !kernel_name) return NULL;

    engine_kernel_t *kern = (engine_kernel_t *)calloc(1, sizeof(engine_kernel_t));
    if (!kern) return NULL;

    strncpy(kern->name, kernel_name, sizeof(kern->name) - 1);

#ifdef HAS_OPENCL
    if (prog->cl_program && prog->compiled) {
        cl_int err;
        kern->cl_kernel = clCreateKernel(prog->cl_program, kernel_name, &err);
        if (err != CL_SUCCESS) {
            free(kern);
            return NULL;
        }
    }
#endif

    return kern;
}

int engine_kernel_set_arg(engine_ctx_t *ctx, engine_kernel_t *kernel,
                           uint32_t index, uint64_t size,
                           const void *value, bool is_buffer) {
    if (!ctx || !kernel) return -1;

#ifdef HAS_OPENCL
    if (kernel->cl_kernel) {
        if (is_buffer && value) {
            engine_buffer_t *buf = (engine_buffer_t *)value;
            cl_mem cl_buf = buf->cl_buffer;
            return clSetKernelArg(kernel->cl_kernel, index, sizeof(cl_mem),
                                  &cl_buf) == CL_SUCCESS ? 0 : -1;
        } else {
            return clSetKernelArg(kernel->cl_kernel, index, (size_t)size,
                                  value) == CL_SUCCESS ? 0 : -1;
        }
    }
#else
    (void)index; (void)size; (void)value; (void)is_buffer;
#endif
    return 0;
}

void engine_kernel_release(engine_ctx_t *ctx, engine_kernel_t *kernel) {
    if (!kernel) return;
    (void)ctx;
#ifdef HAS_OPENCL
    if (kernel->cl_kernel) clReleaseKernel(kernel->cl_kernel);
#endif
    free(kernel);
}

// ── Execute NDRange ────────────────────────────────────

int engine_execute_ndrange(engine_ctx_t *ctx,
                            engine_kernel_t *kernel,
                            uint32_t work_dim,
                            const uint64_t *global_size,
                            const uint64_t *global_offset,
                            const uint64_t *local_size,
                            engine_buffer_t **output_buffers,
                            uint32_t num_outputs) {
    if (!ctx || !kernel) return -1;

    // Convert uint64_t arrays to size_t arrays for OpenCL
    size_t g_size[3] = {1, 1, 1};
    size_t g_offset[3] = {0, 0, 0};
    size_t l_size[3] = {0, 0, 0};

    for (uint32_t d = 0; d < work_dim && d < 3; d++) {
        g_size[d] = (size_t)(global_size ? global_size[d] : 1);
        g_offset[d] = (size_t)(global_offset ? global_offset[d] : 0);
        l_size[d] = (size_t)(local_size ? local_size[d] : 0);
    }

#ifdef HAS_OPENCL
    if (kernel->cl_kernel && ctx->queue) {
        cl_int err = clEnqueueNDRangeKernel(
            ctx->queue, kernel->cl_kernel, (cl_uint)work_dim,
            global_offset ? g_offset : NULL,
            g_size, local_size ? l_size : NULL,
            0, NULL, NULL);

        if (err != CL_SUCCESS) return -1;

        // Read back output buffers
        for (uint32_t i = 0; i < num_outputs; i++) {
            if (output_buffers[i] && output_buffers[i]->is_opencl) {
                // Output data stays on device until explicitly read
                // The buffer read will be triggered later by engine_buffer_read
            }
        }

        return 0;
    }
#else
    (void)g_size; (void)g_offset; (void)l_size; (void)output_buffers; (void)num_outputs;
#endif
    return -1;
}

int engine_finish(engine_ctx_t *ctx) {
    if (!ctx) return -1;
#ifdef HAS_OPENCL
    if (ctx->queue) {
        clFinish(ctx->queue);
    }
#endif
    return 0;
}

// ── Micro-benchmark ────────────────────────────────────

double engine_run_micro_benchmark(engine_ctx_t *ctx) {
    // Simple matrix multiply benchmark to measure empirical GFLOPS
    // TODO: Run 1024x1024 matmul on the local device, measure time
    // Return GFLOPS = 2*M*N*K / time_seconds / 1e9
    (void)ctx;
    return 0.0; // Not yet implemented
}
