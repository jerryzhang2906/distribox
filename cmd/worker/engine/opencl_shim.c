/*
 * cmd/worker/engine/opencl_shim.c — Minimal CGO shim to call local OpenCL GPU
 *
 * Avoids the zig-compiled engine_opencl.c relocation issues.
 * Directly linked against IntelOpenCL64.dll (or system OpenCL).
 * Only compiled when CGO is enabled.
 *
 * Build: CGO_ENABLED=1 CGO_LDFLAGS="-lOpenCL" go build
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

// ── Global OpenCL state ──────────────────────────────

static cl_platform_id   g_platform    = NULL;
static cl_device_id     g_device      = NULL;
static cl_context       g_context     = NULL;
static cl_command_queue g_queue       = NULL;
static int              g_initialized = 0;

// ── Init ──────────────────────────────────────────────

int ocl_init(void) {
    if (g_initialized) return 0;

    cl_int err;
    cl_uint num;

    // Get platform (prefer GPU)
    err = clGetPlatformIDs(1, &g_platform, &num);
    if (err != CL_SUCCESS || num == 0) {
        fprintf(stderr, "OpenCL: no platform found (err=%d)\n", err);
        return -1;
    }

    // Get GPU device
    err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_GPU, 1, &g_device, NULL);
    if (err != CL_SUCCESS) {
        // Fall back to CPU
        err = clGetDeviceIDs(g_platform, CL_DEVICE_TYPE_CPU, 1, &g_device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL: no device found (err=%d)\n", err);
            return -1;
        }
    }

    // Create context
    g_context = clCreateContext(NULL, 1, &g_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: context create failed (err=%d)\n", err);
        return -1;
    }

    // Create command queue (OpenCL 2.0)
    g_queue = clCreateCommandQueueWithProperties(g_context, g_device, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: queue create failed (err=%d)\n", err);
        clReleaseContext(g_context);
        g_context = NULL;
        return -1;
    }

    g_initialized = 1;

    // Print device info
    char name[256];
    clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    fprintf(stderr, "OpenCL GPU: %s\n", name);

    return 0;
}

void ocl_close(void) {
    if (g_queue)   { clReleaseCommandQueue(g_queue); g_queue = NULL; }
    if (g_context) { clReleaseContext(g_context); g_context = NULL; }
    g_initialized = 0;
}

const char* ocl_device_name(void) {
    static char name[256] = "OpenCL GPU";
    if (g_device) {
        clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    }
    return name;
}

// ── Buffer ops ────────────────────────────────────────

void* ocl_create_buffer(size_t size, const void* data) {
    cl_int err;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    if (data) flags |= CL_MEM_COPY_HOST_PTR;
    cl_mem buf = clCreateBuffer(g_context, flags, size, (void*)data, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: buffer create failed (err=%d)\n", err);
        return NULL;
    }
    return (void*)buf;
}

int ocl_write_buffer(void* buf, size_t offset, size_t size, const void* data) {
    cl_int err = clEnqueueWriteBuffer(g_queue, (cl_mem)buf, CL_TRUE,
        offset, size, data, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

int ocl_read_buffer(void* buf, size_t offset, size_t size, void* data) {
    cl_int err = clEnqueueReadBuffer(g_queue, (cl_mem)buf, CL_TRUE,
        offset, size, data, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

void ocl_release_buffer(void* buf) {
    if (buf) clReleaseMemObject((cl_mem)buf);
}

// ── Kernel execution ──────────────────────────────────

typedef struct {
    cl_kernel kernel;
    char name[256];
} ocl_kernel_t;

void* ocl_create_kernel(const char* source, const char* kernel_name) {
    cl_int err;
    size_t src_len = strlen(source);

    // Create program
    cl_program prog = clCreateProgramWithSource(g_context, 1, &source, &src_len, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: program create failed (err=%d)\n", err);
        return NULL;
    }

    // Build program
    err = clBuildProgram(prog, 1, &g_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(prog, g_device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "OpenCL: build failed:\n%s\n", log);
        clReleaseProgram(prog);
        return NULL;
    }

    // Create kernel
    cl_kernel kern = clCreateKernel(prog, kernel_name, &err);
    clReleaseProgram(prog); // kernel holds ref

    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: kernel '%s' create failed (err=%d)\n", kernel_name, err);
        return NULL;
    }

    ocl_kernel_t* k = (ocl_kernel_t*)malloc(sizeof(ocl_kernel_t));
    k->kernel = kern;
    strncpy(k->name, kernel_name, sizeof(k->name)-1);
    return (void*)k;
}

int ocl_set_kernel_arg(void* kernel, int index, size_t size, const void* value, int is_buffer) {
    ocl_kernel_t* k = (ocl_kernel_t*)kernel;
    if (is_buffer) {
        cl_mem buf = *(cl_mem*)value;
        return clSetKernelArg(k->kernel, (cl_uint)index, sizeof(cl_mem), &buf);
    }
    return clSetKernelArg(k->kernel, (cl_uint)index, size, value);
}

int ocl_execute_ndrange(void* kernel, int work_dim,
                        const size_t* global_size,
                        const size_t* global_offset,
                        const size_t* local_size) {
    ocl_kernel_t* k = (ocl_kernel_t*)kernel;
    cl_int err = clEnqueueNDRangeKernel(g_queue, k->kernel,
        (cl_uint)work_dim, global_offset, global_size, local_size,
        0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: NDRange failed for '%s' (err=%d)\n", k->name, err);
        return -1;
    }
    return 0;
}

int ocl_finish(void) {
    return clFinish(g_queue);
}

void ocl_release_kernel(void* kernel) {
    if (kernel) {
        ocl_kernel_t* k = (ocl_kernel_t*)kernel;
        clReleaseKernel(k->kernel);
        free(k);
    }
}
