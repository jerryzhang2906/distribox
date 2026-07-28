/*
 * cuda_proxy/nvcuda_proxy.c — CUDA Runtime API proxy DLL
 *
 * Intercepts CUDA calls by replacing nvcuda.dll (NVIDIA's CUDA driver).
 * Forwards all calls to the real driver while capturing:
 *   - cuLaunchKernel (GPU kernel launches → distribute to workers)
 *   - cuMemAlloc / cuMemFree (track GPU memory)
 *   - cuMemcpyHtoD / cuMemcpyDtoH (data transfer tracking)
 *
 * Installation (requires admin):
 *   1. Rename C:\Windows\System32\nvcuda.dll to nvcuda_orig.dll
 *   2. Copy this built DLL to C:\Windows\System32\nvcuda.dll
 *
 * Build: zig cc -shared -O2 cuda_proxy/nvcuda_proxy.c -o build/cuda/nvcuda.dll
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdint.h>

/* ── Logging ──────────────────────────────────────────── */
static FILE *g_log = NULL;

static void log_init(void) {
    if (!g_log) {
        g_log = fopen("C:\\distribox_cuda.log", "a");
    }
}

#define CUDA_LOG(fmt, ...) do { \
    log_init(); \
    if (g_log) fprintf(g_log, "[CUDA] " fmt "\n", ##__VA_ARGS__); \
} while(0)

/* ── Real driver function pointers ────────────────────── */
static HMODULE g_real_cuda = NULL;

typedef int (*cuInit_t)(unsigned int);
typedef int (*cuDeviceGetCount_t)(int *);
typedef int (*cuDeviceGet_t)(int *, int);
typedef int (*cuDeviceGetName_t)(char *, int, int);
typedef int (*cuCtxCreate_t)(void *, unsigned int, int);
typedef int (*cuCtxDestroy_t)(void *);
typedef int (*cuMemAlloc_t)(void *, size_t);
typedef int (*cuMemFree_t)(void *);
typedef int (*cuMemcpyHtoD_t)(void *, const void *, size_t);
typedef int (*cuMemcpyDtoH_t)(void *, const void *, size_t);
typedef int (*cuLaunchKernel_t)(void *, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int, void *, void **, void **);
typedef int (*cuCtxSynchronize_t)(void);
typedef int (*cuModuleLoadData_t)(void *, const void *);
typedef int (*cuModuleGetFunction_t)(void *, void *, const char *);
typedef int (*cuGetErrorString_t)(int, const char **);

static cuInit_t               real_cuInit = NULL;
static cuDeviceGetCount_t     real_cuDeviceGetCount = NULL;
static cuDeviceGet_t          real_cuDeviceGet = NULL;
static cuDeviceGetName_t      real_cuDeviceGetName = NULL;
static cuCtxCreate_t          real_cuCtxCreate = NULL;
static cuCtxDestroy_t         real_cuCtxDestroy = NULL;
static cuMemAlloc_t           real_cuMemAlloc = NULL;
static cuMemFree_t            real_cuMemFree = NULL;
static cuMemcpyHtoD_t         real_cuMemcpyHtoD = NULL;
static cuMemcpyDtoH_t         real_cuMemcpyDtoH = NULL;
static cuLaunchKernel_t       real_cuLaunchKernel = NULL;
static cuCtxSynchronize_t     real_cuCtxSynchronize = NULL;
static cuModuleLoadData_t     real_cuModuleLoadData = NULL;
static cuModuleGetFunction_t  real_cuModuleGetFunction = NULL;
static cuGetErrorString_t     real_cuGetErrorString = NULL;

/* Tracked state */
static int g_cuda_available = 0;
static int g_device_count = 0;
static char g_device_name[256] = {0};
static uint64_t g_total_allocated = 0;

/* ── Initialization ───────────────────────────────────── */

static int load_real_cuda(void) {
    if (g_real_cuda) return 0;

    /* Try multiple locations for the real driver */
    const char *paths[] = {
        "nvcuda_orig.dll",
        "C:\\Windows\\System32\\nvcuda_orig.dll",
        "C:\\Windows\\System32\\DriverStore\\FileRepository\\nv_dispi.inf_amd64_*\\nvcuda.dll",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        g_real_cuda = LoadLibraryA(paths[i]);
        if (g_real_cuda) {
            CUDA_LOG("Loaded real CUDA driver: %s", paths[i]);
            break;
        }
    }

    if (!g_real_cuda) {
        CUDA_LOG("No NVIDIA CUDA driver found — CUDA proxy inactive");
        return 1;
    }

    /* Resolve all function pointers */
    #define LOAD(fn) real_##fn = (fn##_t)GetProcAddress(g_real_cuda, #fn)
    LOAD(cuInit);
    LOAD(cuDeviceGetCount);
    LOAD(cuDeviceGet);
    LOAD(cuDeviceGetName);
    LOAD(cuCtxCreate);
    LOAD(cuCtxDestroy);
    LOAD(cuMemAlloc);
    LOAD(cuMemFree);
    LOAD(cuMemcpyHtoD);
    LOAD(cuMemcpyDtoH);
    LOAD(cuLaunchKernel);
    LOAD(cuCtxSynchronize);
    LOAD(cuModuleLoadData);
    LOAD(cuModuleGetFunction);
    LOAD(cuGetErrorString);
    #undef LOAD

    return 0;
}

/* ── Intercepted API functions ────────────────────────── */

__declspec(dllexport) int cuInit(unsigned int flags) {
    CUDA_LOG("cuInit(flags=%u)", flags);

    if (load_real_cuda() != 0) {
        return 999; /* CUDA_ERROR_UNKNOWN */
    }

    int ret = real_cuInit(flags);
    if (ret == 0) {
        g_cuda_available = 1;
        real_cuDeviceGetCount(&g_device_count);
        if (g_device_count > 0) {
            int dev;
            real_cuDeviceGet(&dev, 0);
            real_cuDeviceGetName(g_device_name, sizeof(g_device_name), dev);
            CUDA_LOG("CUDA available: %d device(s), primary: %s", g_device_count, g_device_name);
        }
    }
    return ret;
}

__declspec(dllexport) int cuDeviceGetCount(int *count) {
    if (!g_cuda_available) return 999;
    return real_cuDeviceGetCount(count);
}

__declspec(dllexport) int cuDeviceGet(int *device, int ordinal) {
    if (!g_cuda_available) return 999;
    return real_cuDeviceGet(device, ordinal);
}

__declspec(dllexport) int cuDeviceGetName(char *name, int len, int dev) {
    if (!g_cuda_available) return 999;
    return real_cuDeviceGetName(name, len, dev);
}

__declspec(dllexport) int cuCtxCreate(void *pctx, unsigned int flags, int dev) {
    CUDA_LOG("cuCtxCreate(dev=%d, flags=%u)", dev, flags);
    if (!g_cuda_available) return 999;
    return real_cuCtxCreate(pctx, flags, dev);
}

__declspec(dllexport) int cuCtxDestroy(void *ctx) {
    CUDA_LOG("cuCtxDestroy()");
    if (!g_cuda_available) return 999;
    return real_cuCtxDestroy(ctx);
}

__declspec(dllexport) int cuMemAlloc(void *dptr, size_t bytesize) {
    CUDA_LOG("cuMemAlloc(%zu bytes)", bytesize);
    if (!g_cuda_available) return 999;
    int ret = real_cuMemAlloc(dptr, bytesize);
    if (ret == 0) {
        g_total_allocated += bytesize;
        CUDA_LOG("  -> total allocated: %llu bytes", (unsigned long long)g_total_allocated);
    }
    return ret;
}

__declspec(dllexport) int cuMemFree(void *dptr) {
    CUDA_LOG("cuMemFree()");
    if (!g_cuda_available) return 999;
    return real_cuMemFree(dptr);
}

__declspec(dllexport) int cuMemcpyHtoD(void *dst, const void *src, size_t bytes) {
    /* Track but don't intercept — just log large transfers */
    if (bytes > 1048576) {
        CUDA_LOG("cuMemcpyHtoD(%zu bytes — %.1f MB)", bytes, (double)bytes / 1048576.0);
    }
    if (!g_cuda_available) return 999;
    return real_cuMemcpyHtoD(dst, src, bytes);
}

__declspec(dllexport) int cuMemcpyDtoH(void *dst, const void *src, size_t bytes) {
    if (bytes > 1048576) {
        CUDA_LOG("cuMemcpyDtoH(%zu bytes — %.1f MB)", bytes, (double)bytes / 1048576.0);
    }
    if (!g_cuda_available) return 999;
    return real_cuMemcpyDtoH(dst, src, bytes);
}

__declspec(dllexport) int cuLaunchKernel(
    void *f, unsigned int gdx, unsigned int gdy, unsigned int gdz,
    unsigned int bdx, unsigned int bdy, unsigned int bdz,
    unsigned int sharedMem, void *stream, void **params, void **extra)
{
    CUDA_LOG("cuLaunchKernel(grid=(%u,%u,%u), block=(%u,%u,%u), shared=%u bytes)",
        gdx, gdy, gdz, bdx, bdy, bdz, sharedMem);

    /* TODO: DistriBox integration — intercept here for distributed GPU compute
     *
     * 1. Serialize kernel params + grid/block dims
     * 2. Send to VGPU Core via TCP
     * 3. VGPU Core dispatches to Workers
     * 4. Workers execute locally (real GPU or CPU fallback)
     * 5. Collect results, copy back to device memory
     *
     * For now: passthrough to real CUDA driver.
     */
    if (!g_cuda_available) return 999;
    return real_cuLaunchKernel(f, gdx, gdy, gdz, bdx, bdy, bdz, sharedMem, stream, params, extra);
}

__declspec(dllexport) int cuCtxSynchronize(void) {
    if (!g_cuda_available) return 999;
    return real_cuCtxSynchronize();
}

__declspec(dllexport) int cuModuleLoadData(void *module, const void *image) {
    CUDA_LOG("cuModuleLoadData()");
    if (!g_cuda_available) return 999;
    return real_cuModuleLoadData(module, image);
}

__declspec(dllexport) int cuModuleGetFunction(void *hfunc, void *module, const char *name) {
    CUDA_LOG("cuModuleGetFunction(%s)", name);
    if (!g_cuda_available) return 999;
    return real_cuModuleGetFunction(hfunc, module, name);
}

__declspec(dllexport) int cuGetErrorString(int error, const char **pStr) {
    if (!g_cuda_available) return 999;
    return real_cuGetErrorString(error, pStr);
}

/* ── DLL Entry Point ──────────────────────────────────── */

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    (void)hinstDLL;
    (void)lpvReserved;

    switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
        log_init();
        CUDA_LOG("=== DistriBox CUDA Proxy loaded ===");
        load_real_cuda(); /* Try to load real driver early */
        break;
    case DLL_PROCESS_DETACH:
        CUDA_LOG("=== DistriBox CUDA Proxy unloaded (total allocated: %llu bytes) ===",
            (unsigned long long)g_total_allocated);
        if (g_log) fclose(g_log);
        break;
    }
    return TRUE;
}
