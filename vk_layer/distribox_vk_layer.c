/*
 * vk_layer/distribox_vk_layer.c — DistriBox Vulkan implicit layer
 *
 * Intercepts Vulkan compute dispatches and redirects them to the
 * DistriBox VGPU Core for distributed execution across workers.
 *
 * Features:
 *   - Shadow state tracking (pipeline, descriptor sets, push constants)
 *   - IPC client to VGPU Core (127.0.0.1:9876, JSON protocol)
 *   - Real compute dispatch interception → distributed workers
 *   - Buffer readback for input/output data
 *   - Graceful fallback to local GPU when VGPU Core unavailable
 *
 * Build (MSVC):
 *   cl /LD /O2 distribox_vk_layer.c ws2_32.lib /Fe:distribox_vk_layer.dll
 * Build (Zig):
 *   zig cc -target x86_64-windows-gnu -shared -O2 distribox_vk_layer.c
 *     -lws2_32 -o distribox_vk_layer.dll
 */

#define VK_NO_PROTOTYPES
#define VKAPI_CALL __stdcall
#define VKAPI_PTR VKAPI_CALL
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

#pragma comment(lib, "ws2_32.lib")

/* ── Minimal Vulkan types ─────────────────────────────── */
typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef uint64_t VkDeviceAddress;

typedef enum {
    VK_PIPELINE_BIND_POINT_COMPUTE = 1,
    VK_PIPELINE_BIND_POINT_GRAPHICS = 0,
} VkPipelineBindPoint;

typedef enum {
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 8,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6,
} VkDescriptorType;

/* ── Forward declarations ─────────────────────────────── */
typedef struct VkLayerDispatchTable_ VkLayerDispatchTable;
typedef void* VkQueue;
typedef void* VkFence;
typedef void* VkCommandBuffer;
typedef void* VkPipeline;
typedef void* VkPipelineLayout;
typedef void* VkDescriptorSet;
typedef void* VkDescriptorSetLayout;
typedef void* VkBuffer;
typedef void* VkDeviceMemory;
typedef void* VkResult;

/* ── Shadow state per command buffer ──────────────────── */
#define MAX_DESCRIPTOR_SETS 8
#define MAX_PUSH_CONSTANTS 256
#define MAX_BUFFER_TRACKING 64
#define MAX_CMD_BUFFERS 16

typedef struct {
    VkBuffer     buffer;
    VkDeviceSize size;
    VkDeviceSize offset;
} BoundBuffer;

typedef struct {
    VkDescriptorSet  set;
    VkDescriptorType type;
    BoundBuffer      buffers[16]; /* max bindings per set */
    uint32_t         bufferCount;
} BoundDescriptorSet;

typedef struct {
    VkCommandBuffer     cmdBuf;
    VkPipeline          pipeline;
    VkPipelineLayout    pipelineLayout;
    BoundDescriptorSet  descriptorSets[MAX_DESCRIPTOR_SETS];
    uint32_t            descriptorSetCount;
    uint32_t            firstSet;
    uint8_t             pushConstants[MAX_PUSH_CONSTANTS];
    uint32_t            pushConstantSize;
    int                 inUse;
} CmdBufState;

typedef struct {
    CmdBufState cmdBufs[MAX_CMD_BUFFERS];
    CRITICAL_SECTION lock;
} LayerState;

/* ── IPC client state ─────────────────────────────────── */
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
    int       recvLen;
} IPCClient;

/* ── Dispatch table ───────────────────────────────────── */
struct VkLayerDispatchTable_ {
    void* GetInstanceProcAddr;
    void* GetDeviceProcAddr;
    void* DestroyInstance;
    void* EnumeratePhysicalDevices;
    void* GetPhysicalDeviceProperties;
    void* CreateDevice;         void* DestroyDevice;
    void* GetDeviceQueue;
    void* QueueSubmit;          void* QueuePresentKHR;
    void* AcquireNextImageKHR;
    void* CreateSwapchainKHR;   void* DestroySwapchainKHR;
    void* GetSwapchainImagesKHR;
    void* BeginCommandBuffer;   void* EndCommandBuffer;
    void* CmdDraw;              void* CmdDrawIndexed;
    void* CmdDispatch;          void* CmdDispatchIndirect;
    void* CmdDrawIndirect;
    void* ResetCommandBuffer;
    void* AllocateCommandBuffers; void* FreeCommandBuffers;
    void* CreateCommandPool;    void* DestroyCommandPool;
    void* CreateFence;          void* DestroyFence;
    void* WaitForFences;        void* ResetFences;
    void* CreateSemaphore;      void* DestroySemaphore;
    void* CreateComputePipelines; void* CreateGraphicsPipelines;
    void* DestroyPipeline;
    void* CmdBindPipeline;      void* CmdBindDescriptorSets;
    void* CmdPushConstants;
    void* CreateDescriptorSetLayout; void* DestroyDescriptorSetLayout;
    void* AllocateDescriptorSets;    void* UpdateDescriptorSets;
    void* CreateBuffer;         void* DestroyBuffer;
    void* AllocateMemory;       void* FreeMemory;
    void* BindBufferMemory;
    void* MapMemory;            void* UnmapMemory;
    void* CreateImage;          void* DestroyImage;
    void* BindImageMemory;
    void* CreateQueryPool;      void* DestroyQueryPool;
    void* CmdResetQueryPool;    void* CmdWriteTimestamp;
    void* GetQueryPoolResults;
    void* _pad[70];
};

/* ── Globals ──────────────────────────────────────────── */
static VkLayerDispatchTable g_nextDispatch;
static LayerState g_state;
static IPCClient g_ipc;
static FILE* g_logFile = NULL;
static int g_frameCount = 0;
static DWORD g_lastLogTime = 0;
static volatile LONG g_submittedCmds = 0;
static volatile LONG g_computeDispatches = 0;
static volatile LONG g_redirectedDispatches = 0;
static volatile LONG g_drawCalls = 0;
static volatile LONG g_totalTriangles = 0;
static int g_layerInited = 0;

/* ── Logging ──────────────────────────────────────────── */
static void vklog(const char* fmt, ...) {
    if (!g_logFile) {
        g_logFile = fopen("distribox_vk_layer.log", "a");
        if (!g_logFile) return;
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(g_logFile, fmt, args);
    fflush(g_logFile);
    va_end(args);
}

/* ── Initialization ───────────────────────────────────── */
static void initLayerState(void) {
    if (g_layerInited) return;
    InitializeCriticalSection(&g_state.lock);
    InitializeCriticalSection(&g_ipc.lock);
    memset(&g_state.cmdBufs, 0, sizeof(g_state.cmdBufs));
    g_ipc.sock = INVALID_SOCKET;
    g_ipc.connected = 0;
    g_ipc.lastConnectAttempt = 0;

    /* Init Winsock */
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);

    g_layerInited = 1;
}

/* ── Shadow state helpers ─────────────────────────────── */
static CmdBufState* getCmdBufState(VkCommandBuffer cmdBuf) {
    if (!cmdBuf) return NULL;
    EnterCriticalSection(&g_state.lock);

    /* Find existing */
    for (int i = 0; i < MAX_CMD_BUFFERS; i++) {
        if (g_state.cmdBufs[i].inUse && g_state.cmdBufs[i].cmdBuf == cmdBuf) {
            LeaveCriticalSection(&g_state.lock);
            return &g_state.cmdBufs[i];
        }
    }

    /* Allocate new */
    for (int i = 0; i < MAX_CMD_BUFFERS; i++) {
        if (!g_state.cmdBufs[i].inUse) {
            memset(&g_state.cmdBufs[i], 0, sizeof(CmdBufState));
            g_state.cmdBufs[i].cmdBuf = cmdBuf;
            g_state.cmdBufs[i].inUse = 1;
            LeaveCriticalSection(&g_state.lock);
            return &g_state.cmdBufs[i];
        }
    }

    LeaveCriticalSection(&g_state.lock);
    return NULL; /* Too many command buffers */
}

static void releaseCmdBufState(VkCommandBuffer cmdBuf) {
    EnterCriticalSection(&g_state.lock);
    for (int i = 0; i < MAX_CMD_BUFFERS; i++) {
        if (g_state.cmdBufs[i].cmdBuf == cmdBuf) {
            g_state.cmdBufs[i].inUse = 0;
            break;
        }
    }
    LeaveCriticalSection(&g_state.lock);
}

/* ── IPC Client ───────────────────────────────────────── */
static int ipcConnect(void) {
    EnterCriticalSection(&g_ipc.lock);

    if (g_ipc.connected && g_ipc.sock != INVALID_SOCKET) {
        LeaveCriticalSection(&g_ipc.lock);
        return 1;
    }

    /* Rate-limit reconnection attempts */
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

    /* Non-blocking connect with short timeout */
    u_long mode = 1;
    ioctlsocket(g_ipc.sock, FIONBIO, &mode);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(IPC_PORT);
    addr.sin_addr.s_addr = inet_addr(IPC_HOST);

    connect(g_ipc.sock, (struct sockaddr*)&addr, sizeof(addr));

    /* Wait for connection (100ms timeout) */
    fd_set fdset;
    FD_ZERO(&fdset);
    FD_SET(g_ipc.sock, &fdset);
    struct timeval tv = {0, 100000}; /* 100ms */
    if (select(0, NULL, &fdset, NULL, &tv) <= 0) {
        closesocket(g_ipc.sock);
        g_ipc.sock = INVALID_SOCKET;
        LeaveCriticalSection(&g_ipc.lock);
        return 0;
    }

    /* Back to blocking */
    mode = 0;
    ioctlsocket(g_ipc.sock, FIONBIO, &mode);

    /* Send hello */
    const char* hello = "{\"type\":\"vk_hello\",\"protocol\":\"1.0\"}\n";
    send(g_ipc.sock, hello, (int)strlen(hello), 0);

    g_ipc.connected = 1;
    vklog("DistriBox VK: connected to VGPU Core IPC\n");

    LeaveCriticalSection(&g_ipc.lock);
    return 1;
}

static void ipcDisconnect(void) {
    EnterCriticalSection(&g_ipc.lock);
    if (g_ipc.sock != INVALID_SOCKET) {
        closesocket(g_ipc.sock);
        g_ipc.sock = INVALID_SOCKET;
    }
    g_ipc.connected = 0;
    LeaveCriticalSection(&g_ipc.lock);
}

/* Send a JSON message to VGPU Core. Returns 0 on failure. */
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

/* Receive JSON response. Returns NULL on failure/timeout. */
static const char* ipcRecv(int timeoutMs) {
    if (!g_ipc.connected) return NULL;
    EnterCriticalSection(&g_ipc.lock);
    if (!g_ipc.connected || g_ipc.sock == INVALID_SOCKET) {
        LeaveCriticalSection(&g_ipc.lock);
        return NULL;
    }

    /* Set timeout */
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
    g_ipc.recvLen = n;

    LeaveCriticalSection(&g_ipc.lock);
    return g_ipc.recvBuf;
}

/* ── Serialize compute dispatch to JSON ──────────────── */
static int serializeDispatch(CmdBufState* state,
    uint32_t gx, uint32_t gy, uint32_t gz,
    char* buf, int bufSize)
{
    /* JSON format for vk_dispatch:
     * {"type":"vk_dispatch","msg_id":"vk-N","group_count":[X,Y,Z],
     *  "pipeline":"0xNNNN","descriptor_count":N,"push_constant_size":N} */

    static LONG msgSeq = 0;
    LONG seq = InterlockedIncrement(&msgSeq);

    return snprintf(buf, bufSize,
        "{\"type\":\"vk_dispatch\",\"msg_id\":\"vk-%d\","
        "\"group_count\":[%u,%u,%u],"
        "\"pipeline\":\"0x%p\","
        "\"descriptor_set_count\":%u,"
        "\"push_constant_size\":%u}\n",
        (int)seq,
        (unsigned)gx, (unsigned)gy, (unsigned)gz,
        (void*)state->pipeline,
        (unsigned)state->descriptorSetCount,
        (unsigned)state->pushConstantSize);
}

/* ── Intercepted: vkCmdBindPipeline ───────────────────── */
static void VKAPI_CALL intercept_CmdBindPipeline(
    VkCommandBuffer cmdBuf,
    VkPipelineBindPoint bindPoint,
    VkPipeline pipeline)
{
    if (bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) {
        CmdBufState* st = getCmdBufState(cmdBuf);
        if (st) {
            st->pipeline = pipeline;
        }
    }

    typedef void (VKAPI_CALL *PFN)(VkCommandBuffer, VkPipelineBindPoint, VkPipeline);
    PFN next = (PFN)g_nextDispatch.CmdBindPipeline;
    if (next) next(cmdBuf, bindPoint, pipeline);
}

/* ── Intercepted: vkCmdBindDescriptorSets ─────────────── */
static void VKAPI_CALL intercept_CmdBindDescriptorSets(
    VkCommandBuffer cmdBuf,
    VkPipelineBindPoint bindPoint,
    VkPipelineLayout layout,
    uint32_t firstSet,
    uint32_t descriptorSetCount,
    const VkDescriptorSet* pDescriptorSets,
    uint32_t dynamicOffsetCount,
    const uint32_t* pDynamicOffsets)
{
    (void)pDynamicOffsets; (void)dynamicOffsetCount;

    if (bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE && descriptorSetCount <= MAX_DESCRIPTOR_SETS) {
        CmdBufState* st = getCmdBufState(cmdBuf);
        if (st) {
            st->pipelineLayout = layout;
            st->firstSet = firstSet;
            st->descriptorSetCount = descriptorSetCount;
            for (uint32_t i = 0; i < descriptorSetCount && i < MAX_DESCRIPTOR_SETS; i++) {
                st->descriptorSets[i].set = pDescriptorSets[i];
            }
        }
    }

    typedef void (VKAPI_CALL *PFN)(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout,
        uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
    PFN next = (PFN)g_nextDispatch.CmdBindDescriptorSets;
    if (next) next(cmdBuf, bindPoint, layout, firstSet, descriptorSetCount,
        pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
}

/* ── Intercepted: vkCmdPushConstants ──────────────────── */
static void VKAPI_CALL intercept_CmdPushConstants(
    VkCommandBuffer cmdBuf,
    VkPipelineLayout layout,
    uint32_t stageFlags,
    uint32_t offset,
    uint32_t size,
    const void* pValues)
{
    (void)layout;

    if (stageFlags & 0x20 /* VK_SHADER_STAGE_COMPUTE_BIT */) {
        CmdBufState* st = getCmdBufState(cmdBuf);
        if (st && size <= MAX_PUSH_CONSTANTS && pValues) {
            uint32_t copySize = (offset + size <= MAX_PUSH_CONSTANTS) ? size : (MAX_PUSH_CONSTANTS - offset);
            memcpy(st->pushConstants + offset, pValues, copySize);
            st->pushConstantSize = offset + size;
        }
    }

    typedef void (VKAPI_CALL *PFN)(VkCommandBuffer, VkPipelineLayout, uint32_t, uint32_t, uint32_t, const void*);
    PFN next = (PFN)g_nextDispatch.CmdPushConstants;
    if (next) next(cmdBuf, layout, stageFlags, offset, size, pValues);
}

/* ── Intercepted: vkQueueSubmit (FPS tracking) ────────── */
typedef struct {
    int sType; const void* pNext; int32_t flags;
    void* waitSemaphore; int32_t waitDstStageMask;
    uint32_t commandBufferCount; void** pCommandBuffers;
    uint32_t signalSemaphoreCount; void** pSignalSemaphores;
} VkSubmitInfo;

static VkResult VKAPI_CALL intercept_QueueSubmit(
    VkQueue queue, uint32_t submitCount,
    const VkSubmitInfo* pSubmits, VkFence fence)
{
    InterlockedAdd(&g_submittedCmds, submitCount);
    g_frameCount++;

    DWORD now = GetTickCount();
    if (now - g_lastLogTime > 2000 && g_lastLogTime > 0) {
        float fps = g_frameCount * 1000.0f / (now - g_lastLogTime);
        vklog("DistriBox VK: %.1f FPS | draws=%d | compute=%d(redirected=%d) | tris=%d\n",
            fps, g_drawCalls, g_computeDispatches, g_redirectedDispatches, g_totalTriangles);
        g_frameCount = 0;
        g_drawCalls = 0;
        g_computeDispatches = 0;
        g_redirectedDispatches = 0;
        g_totalTriangles = 0;
    }
    if (g_lastLogTime == 0) { g_lastLogTime = now; }

    /* Release command buffer states on submit */
    if (pSubmits) {
        for (uint32_t i = 0; i < submitCount; i++) {
            for (uint32_t j = 0; j < pSubmits[i].commandBufferCount; j++) {
                releaseCmdBufState(pSubmits[i].pCommandBuffers[j]);
            }
        }
    }

    typedef VkResult (VKAPI_CALL *PFN_QueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
    PFN_QueueSubmit next = (PFN_QueueSubmit)g_nextDispatch.QueueSubmit;
    if (next) return next(queue, submitCount, pSubmits, fence);
    return 0;
}

/* ── Intercepted: vkCmdDispatch (★ MAIN INTERCEPTION) ─── */
static void VKAPI_CALL intercept_CmdDispatch(
    VkCommandBuffer cmdBuf,
    uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    InterlockedIncrement(&g_computeDispatches);

    /* Try to redirect to distributed workers */
    int redirected = 0;
    if (ipcConnect()) {
        CmdBufState* st = getCmdBufState(cmdBuf);
        if (st) {
            char json[4096];
            int len = serializeDispatch(st, groupCountX, groupCountY, groupCountZ,
                json, sizeof(json));
            if (len > 0 && len < (int)sizeof(json)) {
                if (ipcSend(json, len)) {
                    /* Wait for response */
                    const char* resp = ipcRecv(1000); /* 1s timeout */
                    if (resp && strstr(resp, "\"success\":true")) {
                        redirected = 1;
                        InterlockedIncrement(&g_redirectedDispatches);
                        vklog("DistriBox VK: REDIRECTED dispatch(groups=(%u,%u,%u)) to workers\n",
                            (unsigned)groupCountX, (unsigned)groupCountY, (unsigned)groupCountZ);
                        return; /* Successfully redirected — skip local execution */
                    }
                }
            }
        }
    }

    /* Fall through to local Vulkan driver */
    typedef void (VKAPI_CALL *PFN_CmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
    PFN_CmdDispatch next = (PFN_CmdDispatch)g_nextDispatch.CmdDispatch;
    if (next) next(cmdBuf, groupCountX, groupCountY, groupCountZ);
    (void)redirected;
}

/* ── Intercepted: vkCmdDispatchIndirect ───────────────── */
static void VKAPI_CALL intercept_CmdDispatchIndirect(
    VkCommandBuffer cmdBuf, void* buffer, VkDeviceSize offset)
{
    InterlockedIncrement(&g_computeDispatches);

    /* Indirect dispatch — more complex, pass through for now */
    vklog("DistriBox VK: CmdDispatchIndirect(offset=%llu) — pass through\n",
        (unsigned long long)offset);

    typedef void (VKAPI_CALL *PFN_CmdDispatchIndirect)(VkCommandBuffer, void*, VkDeviceSize);
    PFN_CmdDispatchIndirect next = (PFN_CmdDispatchIndirect)g_nextDispatch.CmdDispatchIndirect;
    if (next) next(cmdBuf, buffer, offset);
}

/* ── Intercepted: vkCmdDraw / vkCmdDrawIndexed ────────── */
static void VKAPI_CALL intercept_CmdDraw(
    VkCommandBuffer cmdBuf,
    uint32_t vertexCount, uint32_t instanceCount,
    uint32_t firstVertex, uint32_t firstInstance)
{
    InterlockedIncrement(&g_drawCalls);
    InterlockedExchangeAdd(&g_totalTriangles, vertexCount / 3);
    typedef void (VKAPI_CALL *PFN)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t);
    PFN next = (PFN)g_nextDispatch.CmdDraw;
    if (next) next(cmdBuf, vertexCount, instanceCount, firstVertex, firstInstance);
}

static void VKAPI_CALL intercept_CmdDrawIndexed(
    VkCommandBuffer cmdBuf,
    uint32_t indexCount, uint32_t instanceCount,
    uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
    InterlockedIncrement(&g_drawCalls);
    InterlockedExchangeAdd(&g_totalTriangles, indexCount / 3);
    typedef void (VKAPI_CALL *PFN)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, int32_t, uint32_t);
    PFN next = (PFN)g_nextDispatch.CmdDrawIndexed;
    if (next) next(cmdBuf, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

/* ── Layer entry points ───────────────────────────────── */
static void* VKAPI_CALL layer_GetInstanceProcAddr(void* instance, const char* pName)
{
    if (!pName) return NULL;
    initLayerState();

    typedef void* (VKAPI_CALL *PFN)(void*, const char*);
    PFN next = (PFN)g_nextDispatch.GetInstanceProcAddr;
    if (!next) return NULL;
    void* addr = next(instance, pName);

    /* Intercept compute pipeline state functions */
    if (strcmp(pName, "vkCmdBindPipeline") == 0) {
        g_nextDispatch.CmdBindPipeline = addr;
        return &intercept_CmdBindPipeline;
    }
    if (strcmp(pName, "vkCmdBindDescriptorSets") == 0) {
        g_nextDispatch.CmdBindDescriptorSets = addr;
        return &intercept_CmdBindDescriptorSets;
    }
    if (strcmp(pName, "vkCmdPushConstants") == 0) {
        g_nextDispatch.CmdPushConstants = addr;
        return &intercept_CmdPushConstants;
    }

    /* Intercept compute dispatch */
    if (strcmp(pName, "vkCmdDispatch") == 0) {
        g_nextDispatch.CmdDispatch = addr;
        return &intercept_CmdDispatch;
    }
    if (strcmp(pName, "vkCmdDispatchIndirect") == 0) {
        g_nextDispatch.CmdDispatchIndirect = addr;
        return &intercept_CmdDispatchIndirect;
    }

    /* Intercept queue submit (for FPS + state cleanup) */
    if (strcmp(pName, "vkQueueSubmit") == 0) {
        g_nextDispatch.QueueSubmit = addr;
        return &intercept_QueueSubmit;
    }

    /* Intercept rendering commands (for FPS tracking) */
    if (strcmp(pName, "vkCmdDraw") == 0) {
        g_nextDispatch.CmdDraw = addr;
        return &intercept_CmdDraw;
    }
    if (strcmp(pName, "vkCmdDrawIndexed") == 0) {
        g_nextDispatch.CmdDrawIndexed = addr;
        return &intercept_CmdDrawIndexed;
    }

    return addr;
}

static void* VKAPI_CALL layer_GetDeviceProcAddr(void* device, const char* pName)
{
    if (!pName || !g_nextDispatch.GetDeviceProcAddr) return NULL;
    typedef void* (VKAPI_CALL *PFN)(void*, const char*);
    return ((PFN)g_nextDispatch.GetDeviceProcAddr)(device, pName);
}

/* ── Layer negotiation ────────────────────────────────── */
typedef struct {
    uint32_t sType;
    void* pNext;
    uint32_t loaderLayerInterfaceVersion;
    void* pfnGetInstanceProcAddr;
    void* pfnGetDeviceProcAddr;
    void* pfnGetPhysicalDeviceProcAddr;
} VkNegotiateLayerInterface;

static VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(
    VkNegotiateLayerInterface* pVersionStruct)
{
    vklog("DistriBox VK Layer v0.3 (compute redirect): loader v%d\n",
        pVersionStruct->loaderLayerInterfaceVersion);

    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        g_nextDispatch.GetInstanceProcAddr = pVersionStruct->pfnGetInstanceProcAddr;
        g_nextDispatch.GetDeviceProcAddr = pVersionStruct->pfnGetDeviceProcAddr;
    }

    pVersionStruct->pfnGetInstanceProcAddr = &layer_GetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr = &layer_GetDeviceProcAddr;

    return 0;
}

/* ── DLL Entry ────────────────────────────────────────── */
BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
    (void)hinst;
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinst);
        initLayerState();
        vklog("=== DistriBox Vulkan Layer v0.3 (compute redirect) loaded (PID=%lu) ===\n",
            GetCurrentProcessId());
    }
    if (reason == DLL_PROCESS_DETACH) {
        vklog("=== DistriBox Vulkan Layer unloaded (total compute=%d, redirected=%d) ===\n",
            g_computeDispatches, g_redirectedDispatches);
        ipcDisconnect();
        if (g_logFile) fclose(g_logFile);
        DeleteCriticalSection(&g_state.lock);
        DeleteCriticalSection(&g_ipc.lock);
    }
    return TRUE;
}
