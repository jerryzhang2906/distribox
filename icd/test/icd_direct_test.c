/**
 * test/icd_direct_test.c — Direct ICD test (bypasses ICD loader)
 *
 * Calls the distribox* internal functions directly to test the full
 * ICD → IPC → VGPU Core pipeline without needing the Khronos ICD loader.
 *
 * Build:
 *   zig cc -I .. -I ../../third_party/include icd_direct_test.c
 *          -L ../../build/icd -ldistribox_icd -o icd_direct_test.exe
 *
 * Run:
 *   Start VGPU Core first: distribox-vgpu.exe
 *   Then: icd_direct_test.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Include ICD internals directly
#include "../icd.h"
#include "../icd_dispatch.h"

// External globals from icd.c
extern distri_platform_t *g_platform;
extern distri_device_t *g_device;

static int g_tests = 0;
static int g_passed = 0;

static void check(cl_int err, const char *msg) {
    g_tests++;
    if (err == CL_SUCCESS) {
        g_passed++;
        printf("  PASS: %s\n", msg);
    } else {
        printf("  FAIL: %s (error %d)\n", msg, err);
    }
}

int main(int argc, char **argv) {
    printf("DistriBox ICD Direct Test (IPC wired)\n");
    printf("=====================================\n\n");

    // 1. Initialize ICD dispatch (creates platform singleton)
    icd_init_dispatch();
    check(g_platform != NULL ? CL_SUCCESS : CL_INVALID_PLATFORM, "icd_init_dispatch");
    printf("    Platform: %s\n", g_platform->name);

    // 2. Get platform IDs
    cl_uint num_platforms;
    cl_platform_id platform;
    cl_int err = distriboxGetPlatformIDs(1, &platform, &num_platforms);
    check(err, "distriboxGetPlatformIDs");
    printf("    Platforms: %u\n", num_platforms);

    // 3. Get device IDs (triggers device creation)
    cl_device_id device;
    cl_uint num_devices;
    err = distriboxGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    check(err, "distriboxGetDeviceIDs");
    printf("    Devices: %u\n", num_devices);

    // 4. Device info
    char dname[256];
    cl_ulong vram;
    cl_uint cu;
    distriboxGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, NULL);
    distriboxGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram), &vram, NULL);
    distriboxGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    printf("    Device: %s, VRAM: %llu MB, CUs: %u\n",
           dname, (unsigned long long)(vram / (1024*1024)), cu);

    // 5. Create context
    cl_context context = distriboxCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "distriboxCreateContext");

    // 6. Create command queue
    cl_command_queue queue = distriboxCreateCommandQueueWithProperties(
        context, device, NULL, &err);
    check(err, "distriboxCreateCommandQueue");

    // 7. Create program
    const char *kernel_source =
        "__kernel void vector_add(__global const float *a,\n"
        "                        __global const float *b,\n"
        "                        __global float *c,\n"
        "                        const int n) {\n"
        "    int i = get_global_id(0);\n"
        "    if (i < n) c[i] = a[i] + b[i];\n"
        "}\n";

    cl_program program = distriboxCreateProgramWithSource(
        context, 1, &kernel_source, NULL, &err);
    check(err, "distriboxCreateProgramWithSource");

    // 8. Build program (sends to VGPU Core via IPC)
    err = distriboxBuildProgram(program, 1, &device, NULL, NULL, NULL);
    check(err, "distriboxBuildProgram");

    // 9. Create kernel
    cl_kernel kernel = distriboxCreateKernel(program, "vector_add", &err);
    check(err, "distriboxCreateKernel(vector_add)");

    // 10. Create buffers with test data
    const int N = 64;
    float a[N], b[N], c_init[N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
        c_init[i] = -1.0f; // Initialize output to invalid value
    }

    cl_mem buffer_a = distriboxCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            N * sizeof(float), a, &err);
    check(err, "distriboxCreateBuffer(A)");

    cl_mem buffer_b = distriboxCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            N * sizeof(float), b, &err);
    check(err, "distriboxCreateBuffer(B)");

    cl_mem buffer_c = distriboxCreateBuffer(context, CL_MEM_READ_WRITE,
                                            N * sizeof(float), NULL, &err);
    check(err, "distriboxCreateBuffer(C)");

    // 11. Set kernel args
    int n = N;
    err = distriboxSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
    check(err, "distriboxSetKernelArg(0)");
    err = distriboxSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
    check(err, "distriboxSetKernelArg(1)");
    err = distriboxSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_c);
    check(err, "distriboxSetKernelArg(2)");
    err = distriboxSetKernelArg(kernel, 3, sizeof(int), &n);
    check(err, "distriboxSetKernelArg(3)");

    // 12. Execute NDRange — 🔑 SENT VIA IPC TO VGPU CORE
    printf("\n  >>> Executing vector_add NDRange via IPC...\n");
    size_t global = N;
    size_t local = 64;
    cl_event ndr_event = NULL;
    err = distriboxEnqueueNDRangeKernel(queue, kernel, 1,
        NULL, &global, &local, 0, NULL, &ndr_event);
    check(err, "distriboxEnqueueNDRangeKernel (IPC dispatch)");

    // 13. Wait for and read result
    if (ndr_event) {
        distriboxWaitForEvents(1, (const cl_event *)&ndr_event);
    }

    float c[N];
    memset(c, 0, sizeof(c));
    err = distriboxEnqueueReadBuffer(queue, buffer_c, CL_TRUE, 0,
                                     N * sizeof(float), c, 0, NULL, NULL);
    check(err, "distriboxEnqueueReadBuffer");

    // 14. Verify results
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = a[i] + b[i];
        if (c[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("    MISMATCH at %d: expected %.1f, got %.1f\n",
                       i, expected, c[i]);
            }
        }
    }
    if (errors == 0) {
        check(CL_SUCCESS, "result verification");
        printf("    All %d results correct!\n", N);
    } else {
        check(CL_INVALID_VALUE, "result verification");
        printf("    %d/%d mismatches\n", errors, N);
    }

    // 15. Cleanup
    distriboxReleaseMemObject(buffer_a);
    distriboxReleaseMemObject(buffer_b);
    distriboxReleaseMemObject(buffer_c);
    distriboxReleaseKernel(kernel);
    distriboxReleaseProgram(program);
    distriboxReleaseCommandQueue(queue);
    distriboxReleaseContext(context);

    if (ndr_event) distriboxReleaseEvent(ndr_event);

    printf("\n=====================================\n");
    printf("Results: %d/%d tests passed\n", g_passed, g_tests);

    return (g_passed == g_tests) ? 0 : 1;
}
