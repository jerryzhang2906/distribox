/**
 * test/icd_test.c — Simple smoke test for the OpenCL ICD
 *
 * Compile and run to verify the ICD works:
 *   cmake -B build -S . -DBUILD_TESTS=ON
 *   cmake --build build
 *   ./build/icd_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

static void check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "FAIL: %s (error %d)\n", msg, err);
        exit(1);
    }
    printf("  PASS: %s\n", msg);
}

int main(void) {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer_a, buffer_b, buffer_c;
    cl_int err;
    cl_uint num;

    printf("DistriBox ICD Smoke Test\n");
    printf("========================\n\n");

    // 1. Platform
    err = clGetPlatformIDs(1, &platform, &num);
    check(err, "clGetPlatformIDs");
    printf("    Platforms found: %u\n", num);

    // 2. Platform info
    char name[256];
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, NULL);
    check(err, "clGetPlatformInfo");
    printf("    Platform name: %s\n", name);

    // 3. Device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num);
    check(err, "clGetDeviceIDs");
    printf("    Devices found: %u\n", num);

    // 4. Device info
    char dname[256];
    cl_ulong vram;
    cl_uint cu;
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
    check(err, "clGetDeviceInfo(NAME)");
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram), &vram, NULL);
    check(err, "clGetDeviceInfo(VRAM)");
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    check(err, "clGetDeviceInfo(CU)");
    printf("    Device: %s, VRAM: %llu MB, CUs: %u\n",
           dname, (unsigned long long)(vram / (1024*1024)), cu);

    // 5. Context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "clCreateContext");

    // 6. Command Queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    check(err, "clCreateCommandQueue");

    // 7. Program (simple vector add kernel)
    const char *kernel_source =
        "__kernel void vector_add(__global const float *a,\n"
        "                        __global const float *b,\n"
        "                        __global float *c,\n"
        "                        const int n) {\n"
        "    int i = get_global_id(0);\n"
        "    if (i < n) c[i] = a[i] + b[i];\n"
        "}\n";

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    check(err, "clBuildProgram");

    // 8. Kernel
    kernel = clCreateKernel(program, "vector_add", &err);
    check(err, "clCreateKernel");

    // 9. Buffers
    const int N = 1024;
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(i * 2); }

    buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              N * sizeof(float), a, &err);
    check(err, "clCreateBuffer(A)");
    buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              N * sizeof(float), b, &err);
    check(err, "clCreateBuffer(B)");
    buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                              N * sizeof(float), NULL, &err);
    check(err, "clCreateBuffer(C)");

    // 10. Set kernel args
    int n = N;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
    check(err, "clSetKernelArg(0)");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
    check(err, "clSetKernelArg(1)");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_c);
    check(err, "clSetKernelArg(2)");
    err = clSetKernelArg(kernel, 3, sizeof(int), &n);
    check(err, "clSetKernelArg(3)");

    // 11. Execute NDRange
    size_t global = N;
    size_t local = 64;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                                  0, NULL, NULL);
    check(err, "clEnqueueNDRangeKernel");

    // 12. Read results
    err = clEnqueueReadBuffer(queue, buffer_c, CL_TRUE, 0,
                              N * sizeof(float), c, 0, NULL, NULL);
    check(err, "clEnqueueReadBuffer");

    // 13. Verify results
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            errors++;
            if (errors <= 3) {
                printf("    MISMATCH at %d: expected %f, got %f\n",
                       i, a[i] + b[i], c[i]);
            }
        }
    }
    if (errors == 0) {
        printf("  PASS: Results verified (all %d correct)\n", N);
    } else {
        printf("  FAIL: %d / %d results incorrect\n", errors, N);
    }

    // 14. Cleanup
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("\n========================\n");
    printf("All tests passed!\n");
    return 0;
}
