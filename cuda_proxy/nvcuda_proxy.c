/*
 * cuda_proxy/nvcuda_proxy.c — CUDA Driver API proxy DLL
 *
 * Replaces nvcuda.dll to intercept CUDA kernel launches and redirect
 * them to the DistriBox VGPU Core for distributed execution.
 *
 * Features:
 *   - Kernel function name tracking (cuModuleGetFunction → function pointer mapping)
 *   - cuLaunchKernel interception with JSON serialization
 *   - IPC client to VGPU Core (127.0.0.1:9876)
 *   - Memory allocation tracking (cuMemAlloc/cuMemFree/cuMemcpy)
 *   - Graceful fallback to real CUDA driver when VGPU Core unavailable
 *
 * Installation (requires admin):
 *   1. Rename C:\Windows\System32\nvcuda.dll to nvcuda_orig.dll
 *   2. Copy this built DLL to C:\Windows\System32\nvcuda.dll
 *
 * Build: zig cc -shared -O2 cuda_proxy/nvcuda_proxy.c -lws2_32 -o build/cuda/nvcuda.dll
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "ws2_32.lib")

/* ── IPC client (Winsock TCP to VGPU Core) ──────────── */
#define IPC_HOST "127.0.0.1"
#define IPC_PORT 9876
#define IPC_BUFFER_SIZE 65536
#define IPC_RECONNECT_MS 5000

typedef struct {
    SOCKET    sock;
    int       connected;
    CRITICAL_SECTION lock;
    DWORD     lastConnectAttempt;
    char      recvBuf[IPC_BUFFER_SIZE];
} IPCClient;

static IPCClient g_ipc;
static int g_wsaInited = 0;

static void ipcInit(void) {
    if (g_wsaInited) return;
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);
    InitializeCriticalSection(&g_ipc.lock);
    g_ipc.sock = INVALID_SOCKET;
    g_ipc.connected = 0;
    g_wsaInited = 1;
}

static int ipcConnect(void) {
    if (!g_wsaInited) ipcInit();
    EnterCriticalSection(&g_ipc.lock);

    if (g_ipc.connected && g_ipc.sock != INVALID_SOCKET) {
        LeaveCriticalSection(&g_ipc.lock);
        return 1;
    }

    DWORD now = GetTickCount();
    if (now - g_ipc.lastConnectAttempt < IPC_RECONNECT_MS) {
        LeaveCriticalSection(&g_ipc.lock);
        return 0;
    }
    g_ipc.lastConnectAttempt = now;

    g_ipc.sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (g_ipc.sock == INVALID_SOCKET) {
        LeaveCriticalSection(&g_ipc.lock);
        return 0;
    }

    u_long mode = 1;
    ioctlsocket(g_ipc.sock, FIONBIO, &mode);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(IPC_PORT);
    addr.sin_addr.s_addr = inet_addr(IPC_HOST);

    connect(g_ipc.sock, (struct sockaddr*)&addr, sizeof(addr));

    fd_set fdset;
    FD_ZERO(&fdset);
    FD_SET(g_ipc.sock, &fdset);
    struct timeval tv = {0, 100000};
    if (select(0, NULL, &fdset, NULL, &tv) <= 0) {
        closesocket(g_ipc.sock);
        g_ipc.sock = INVALID_SOCKET;
        LeaveCriticalSection(&g_ipc.lock);
        return 0;
    }

    mode = 0;
    ioctlsocket(g_ipc.sock, FIONBIO, &mode);

    /* Send hello */
    const char* hello = "{\"type\":\"cuda_hello\",\"protocol\":\"1.0\"}\n";
    send(g_ipc.sock, hello, (int)strlen(hello), 0);

    g_ipc.connected = 1;
    LeaveCriticalSection(&g_ipc.lock);
    return 1;
}

static void ipcDisconnect(void) {
    if (!g_wsaInited) return;
    EnterCriticalSection(&g_ipc.lock);
    if (g_ipc.sock != INVALID_SOCKET) {
        closesocket(g_ipc.sock);
        g_ipc.sock = INVALID_SOCKET;
    }
    g_ipc.connected = 0;
    LeaveCriticalSection(&g_ipc.lock);
}

static int ipcSend(const char* json, int len) {
    if (!g_ipc.connected) return 0;
    EnterCriticalSection(&g_ipc.lock);
    if (!g_ipc.connected || g_ipc.sock == INVALID_SOCKET) {
        LeaveCriticalSection(&g_ipc.lock);
        return 0;
    }
    int sent = send(g_ipc.sock, json, len, 0);
    LeaveCriticalSection(&g_ipc.lock);
    if (sent <= 0) {
        ipcDisconnect();
        return 0;
    }
    return 1;
}

static const char* ipcRecv(int timeoutMs) {
    if (!g_ipc.connected) return NULL;
    EnterCriticalSection(&g_ipc.lock);
    if (!g_ipc.connected || g_ipc.sock == INVALID_SOCKET) {
        LeaveCriticalSection(&g_ipc.lock);
        return NULL;
    }

    u_long mode = 1;
    ioctlsocket(g_ipc.sock, FIONBIO, &mode);

    fd_set fdset;
    FD_ZERO(&fdset);
    FD_SET(g_ipc.sock, &fdset);
    struct timeval tv = {timeoutMs / 1000, (timeoutMs % 1000) * 1000};

    int ready = select(0, &fdset, NULL, NULL, &tv);

    mode = 0;
    ioctlsocket(g_ipc.sock, FIONBIO, &mode);

    if (ready <= 0) {
        LeaveCriticalSection(&g_ipc.lock);
        return NULL;
    }

    int n = recv(g_ipc.sock, g_ipc.recvBuf, IPC_BUFFER_SIZE - 1, 0);
    if (n <= 0) {
        ipcDisconnect();
        LeaveCriticalSection(&g_ipc.lock);
        return NULL;
    }
    g_ipc.recvBuf[n] = '\0';
    LeaveCriticalSection(&g_ipc.lock);
    return g_ipc.recvBuf;
}

/* ── Kernel function name tracking ───────────────────── */
#define MAX_KERNEL_FUNCTIONS 128

typedef struct {
    void*       funcPtr;
    char        name[256];
    uint64_t    gridDim[3];
    uint64_t    totalGridSize;
    int         callCount;
} KernelFunction;

static KernelFunction g_kernels[MAX_KERNEL_FUNCTIONS];
static int g_kernelCount = 0;
static CRITICAL_SECTION g_kernelLock;

static const char* getKernelName(void* f) {
    EnterCriticalSection(&g_kernelLock);
    for (int i = 0; i < g_kernelCount; i++) {
        if (g_kernels[i].funcPtr == f) {
            const char* name = g_kernels[i].name;
            g_kernels[i].callCount++;
            g_kernels[i].totalGridSize += g_kernels[i].gridDim[0] * g_kernels[i].gridDim[1] * g_kernels[i].gridDim[2];
            LeaveCriticalSection(&g_kernelLock);
            return name;
        }
    }
    LeaveCriticalSection(&g_kernelLock);
    return "unknown_kernel";
}

static void registerKernelFunction(void* hfunc, const char* name) {
    EnterCriticalSection(&g_kernelLock);
    if (g_kernelCount < MAX_KERNEL_FUNCTIONS) {
        g_kernels[g_kernelCount].funcPtr = hfunc;
        strncpy(g_kernels[g_kernelCount].name, name, sizeof(g_kernels[g_kernelCount].name) - 1);
        g_kernels[g_kernelCount].name[sizeof(g_kernels[g_kernelCount].name) - 1] = '\0';
        g_kernels[g_kernelCount].callCount = 0;
        g_kernels[g_kernelCount].totalGridSize = 0;
        g_kernelCount++;
    }
    LeaveCriticalSection(&g_kernelLock);
}

/* ── Memory tracking table ────────────────────────────── */
#define MAX_TRACKED_BUFFERS 256

typedef struct {
    void*  devicePtr;
    size_t size;
    char*  hostData;    /* Copy of host data (captured during HtoD) */
    int    hasData;
} TrackedBuffer;

static TrackedBuffer g_trackedBufs[MAX_TRACKED_BUFFERS];
static int g_trackedCount = 0;
static CRITICAL_SECTION g_trackLock;

static TrackedBuffer* findTrackedBuffer(void* devicePtr) {
    for (int i = 0; i < g_trackedCount; i++) {
        if (g_trackedBufs[i].devicePtr == devicePtr) {
            return &g_trackedBufs[i];
        }
    }
    return NULL;
}

static void trackBuffer(void* devicePtr, size_t size) {
    EnterCriticalSection(&g_trackLock);
    if (g_trackedCount < MAX_TRACKED_BUFFERS) {
        g_trackedBufs[g_trackedCount].devicePtr = devicePtr;
        g_trackedBufs[g_trackedCount].size = size;
        g_trackedBufs[g_trackedCount].hostData = NULL;
        g_trackedBufs[g_trackedCount].hasData = 0;
        g_trackedCount++;
    }
    LeaveCriticalSection(&g_trackLock);
}

static void setBufferData(void* devicePtr, const void* data, size_t size) {
    EnterCriticalSection(&g_trackLock);
    TrackedBuffer* buf = findTrackedBuffer(devicePtr);
    if (buf) {
        if (buf->hostData) free(buf->hostData);
        buf->hostData = (char*)malloc(size);
        if (buf->hostData) {
            memcpy(buf->hostData, data, size);
            buf->size = size;
            buf->hasData = 1;
        }
    }
    LeaveCriticalSection(&g_trackLock);
}

/* Base64 encoding for binary data in JSON */
static void base64Encode(const unsigned char* data, int len, char* out) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int i, j;
    for (i = 0, j = 0; i < len; i += 3) {
        uint32_t val = (uint32_t)(data[i]) << 16;
        if (i + 1 < len) val |= (uint32_t)(data[i+1]) << 8;
        if (i + 2 < len) val |= (uint32_t)(data[i+2]);
        out[j++] = table[(val >> 18) & 0x3F];
        out[j++] = table[(val >> 12) & 0x3F];
        out[j++] = (i + 1 < len) ? table[(val >> 6) & 0x3F] : '=';
        out[j++] = (i + 2 < len) ? table[val & 0x3F] : '=';
    }
    out[j] = '\0';
}

/* Generate a unique buffer ID for JSON */
static void genBufID(void* devicePtr, int index, char* out, int outSize) {
    snprintf(out, outSize, "cuda-buf-%p-%d", devicePtr, index);
}

/* ── Logging ──────────────────────────────────────────── */
static FILE *g_log = NULL;
static CRITICAL_SECTION g_logLock;

static void logWrite(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    EnterCriticalSection(&g_logLock);
    if (!g_log) {
        g_log = fopen("C:\\distribox_cuda.log", "a");
        if (!g_log) {
            LeaveCriticalSection(&g_logLock);
            va_end(args);
            return;
        }
    }
    fprintf(g_log, "[CUDA] ");
    vfprintf(g_log, fmt, args);
    fprintf(g_log, "\n");
    fflush(g_log);
    LeaveCriticalSection(&g_logLock);
    va_end(args);
}

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
static volatile LONG g_kernelCalls = 0;
static volatile LONG g_redirectedCalls = 0;

/* ── Initialization ───────────────────────────────────── */

static int load_real_cuda(void) {
    if (g_real_cuda) return 0;

    InitializeCriticalSection(&g_kernelLock);
    InitializeCriticalSection(&g_logLock);
    InitializeCriticalSection(&g_trackLock);

    const char *paths[] = {
        "nvcuda_orig.dll",
        "C:\\Windows\\System32\\nvcuda_orig.dll",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        g_real_cuda = LoadLibraryA(paths[i]);
        if (g_real_cuda) {
            logWrite("Loaded real CUDA driver: %s", paths[i]);
            break;
        }
    }

    if (!g_real_cuda) {
        logWrite("No NVIDIA CUDA driver found — proxy inactive (passthrough mode)");
        return 1;
    }

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
    logWrite("cuInit(flags=%u)", flags);

    if (load_real_cuda() != 0) return 999;

    int ret = real_cuInit(flags);
    if (ret == 0) {
        g_cuda_available = 1;
        real_cuDeviceGetCount(&g_device_count);
        if (g_device_count > 0) {
            int dev;
            real_cuDeviceGet(&dev, 0);
            real_cuDeviceGetName(g_device_name, sizeof(g_device_name), dev);
            logWrite("CUDA available: %d device(s), primary: %s", g_device_count, g_device_name);
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
    logWrite("cuCtxCreate(dev=%d)", dev);
    if (!g_cuda_available) return 999;
    return real_cuCtxCreate(pctx, flags, dev);
}

__declspec(dllexport) int cuCtxDestroy(void *ctx) {
    logWrite("cuCtxDestroy()");
    if (!g_cuda_available) return 999;
    return real_cuCtxDestroy(ctx);
}

__declspec(dllexport) int cuMemAlloc(void *dptr, size_t bytesize) {
    if (!g_cuda_available) return 999;
    int ret = real_cuMemAlloc(dptr, bytesize);
    if (ret == 0) {
        g_total_allocated += bytesize;
        /* Track for later buffer data capture */
        void* ptr = NULL;
        memcpy(&ptr, dptr, sizeof(void*));
        trackBuffer(ptr, bytesize);
    }
    return ret;
}

__declspec(dllexport) int cuMemFree(void *dptr) {
    if (!g_cuda_available) return 999;
    return real_cuMemFree(dptr);
}

__declspec(dllexport) int cuMemcpyHtoD(void *dst, const void *src, size_t bytes) {
    if (!g_cuda_available) return 999;
    /* Capture data being sent to GPU for IPC transmission */
    void* ptr = NULL;
    memcpy(&ptr, dst, sizeof(void*));
    setBufferData(ptr, src, bytes);
    return real_cuMemcpyHtoD(dst, src, bytes);
}

__declspec(dllexport) int cuMemcpyDtoH(void *dst, const void *src, size_t bytes) {
    if (!g_cuda_available) return 999;
    return real_cuMemcpyDtoH(dst, src, bytes);
}

/* ── ★ MAIN INTERCEPTION: cuLaunchKernel ────────────── */
__declspec(dllexport) int cuLaunchKernel(
    void *f, unsigned int gdx, unsigned int gdy, unsigned int gdz,
    unsigned int bdx, unsigned int bdy, unsigned int bdz,
    unsigned int sharedMem, void *stream, void **params, void **extra)
{
    InterlockedIncrement(&g_kernelCalls);

    const char* name = getKernelName(f);

    /* Try to redirect to distributed workers */
    int redirected = 0;
    if (ipcConnect()) {
        /* Build JSON with buffer data */
        char json[65536];
        char bufSection[32768];
        int bufOffset = 0;
        bufSection[0] = '\0';

        /* Include tracked buffer data */
        EnterCriticalSection(&g_trackLock);
        int bufCount = 0;
        for (int i = 0; i < g_trackedCount && bufCount < 16; i++) {
            if (g_trackedBufs[i].hasData && g_trackedBufs[i].hostData) {
                char bufID[128];
                genBufID(g_trackedBufs[i].devicePtr, bufCount, bufID, sizeof(bufID));
                char b64[65536];
                base64Encode((unsigned char*)g_trackedBufs[i].hostData,
                    (int)g_trackedBufs[i].size, b64);

                bufOffset += snprintf(bufSection + bufOffset, sizeof(bufSection) - bufOffset,
                    "%s{\"id\":\"%s\",\"size\":%llu,\"data_b64\":\"%s\",\"direction\":\"in\"}",
                    bufCount > 0 ? "," : "",
                    bufID,
                    (unsigned long long)g_trackedBufs[i].size,
                    b64);
                bufCount++;
            }
        }
        LeaveCriticalSection(&g_trackLock);

        int len = snprintf(json, sizeof(json),
            "{\"type\":\"cuda_launch\","
            "\"kernel_name\":\"%s\","
            "\"grid\":[%u,%u,%u],"
            "\"block\":[%u,%u,%u],"
            "\"shared_mem\":%u,"
            "\"buffers\":[%s]}\n",
            name,
            (unsigned)gdx, (unsigned)gdy, (unsigned)gdz,
            (unsigned)bdx, (unsigned)bdy, (unsigned)bdz,
            (unsigned)sharedMem,
            bufSection);

        if (len > 0 && len < (int)sizeof(json)) {
            if (ipcSend(json, len)) {
                const char* resp = ipcRecv(1000);
                if (resp && strstr(resp, "\"success\":true")) {
                    redirected = 1;
                    InterlockedIncrement(&g_redirectedCalls);
                    logWrite("REDIRECTED cuLaunchKernel: %s grid=(%u,%u,%u) block=(%u,%u,%u) bufs=%d",
                        name, (unsigned)gdx, (unsigned)gdy, (unsigned)gdz,
                        (unsigned)bdx, (unsigned)bdy, (unsigned)bdz, bufCount);
                    return 0; /* Success! Skip local execution */
                }
            }
        }
    }

    /* Fall through to real CUDA driver */
    if (!g_cuda_available) return 999;

    if (strcmp(name, "unknown_kernel") != 0) {
        logWrite("cuLaunchKernel: %s grid=(%u,%u,%u) block=(%u,%u,%u) [local]",
            name, (unsigned)gdx, (unsigned)gdy, (unsigned)gdz,
            (unsigned)bdx, (unsigned)bdy, (unsigned)bdz);
    }

    (void)redirected;
    return real_cuLaunchKernel(f, gdx, gdy, gdz, bdx, bdy, bdz, sharedMem, stream, params, extra);
}

__declspec(dllexport) int cuCtxSynchronize(void) {
    if (!g_cuda_available) return 999;
    return real_cuCtxSynchronize();
}

__declspec(dllexport) int cuModuleLoadData(void *module, const void *image) {
    logWrite("cuModuleLoadData()");
    if (!g_cuda_available) return 999;
    return real_cuModuleLoadData(module, image);
}

__declspec(dllexport) int cuModuleGetFunction(void *hfunc, void *module, const char *name) {
    logWrite("cuModuleGetFunction(%s)", name);
    if (!g_cuda_available) return 999;
    int ret = real_cuModuleGetFunction(hfunc, module, name);
    if (ret == 0 && name) {
        /* Register kernel function name for tracking */
        void* funcPtr = NULL;
        memcpy(&funcPtr, hfunc, sizeof(void*));
        registerKernelFunction(funcPtr, name);
    }
    return ret;
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
        DisableThreadLibraryCalls(hinstDLL);
        ipcInit();
        logWrite("=== DistriBox CUDA Proxy v0.3 (kernel redirect) loaded ===");
        load_real_cuda();
        break;
    case DLL_PROCESS_DETACH:
        logWrite("=== DistriBox CUDA Proxy unloaded (total calls=%d, redirected=%d, "
            "memory=%llu bytes) ===",
            g_kernelCalls, g_redirectedCalls, (unsigned long long)g_total_allocated);
        ipcDisconnect();
        DeleteCriticalSection(&g_ipc.lock);
        DeleteCriticalSection(&g_kernelLock);
        if (g_log) fclose(g_log);
        break;
    }
    return TRUE;
}
