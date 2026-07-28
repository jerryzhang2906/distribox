/*
 * vk_layer/distribox_vk_layer.c — DistriBox Vulkan implicit layer
 *
 * Intercepts Vulkan API calls for:
 *   - FPS tracking (vkQueueSubmit)
 *   - Compute shader dispatch (vkCmdDispatch — AI/ML workloads)
 *   - Draw call tracking (vkCmdDraw, vkCmdDrawIndexed — rendering)
 *
 * Phase 1: FPS overlay + VGPU Core integration
 * Phase 2: Command buffer offloading to workers
 * Phase 3: Distributed frame rendering
 *
 * Build: zig cc -shared -O2 distribox_vk_layer.c -o distribox_vk_layer.dll
 */

#define VK_NO_PROTOTYPES
#define VKAPI_CALL __stdcall
#define VKAPI_PTR VKAPI_CALL
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>

/* ── Minimal Vulkan types ─────────────────────────────── */
typedef uint32_t VkFlags;
typedef uint64_t VkDeviceSize;
typedef uint32_t VkBool32;

typedef struct VkLayerDispatchTable_ {
    /* Instance-level */
    void* GetInstanceProcAddr;
    void* GetDeviceProcAddr;
    void* DestroyInstance;
    void* EnumeratePhysicalDevices;
    void* GetPhysicalDeviceProperties;
    /* Device-level */
    void* CreateDevice;
    void* DestroyDevice;
    void* GetDeviceQueue;
    void* QueueSubmit;
    void* QueuePresentKHR;
    void* AcquireNextImageKHR;
    void* CreateSwapchainKHR;
    void* DestroySwapchainKHR;
    void* GetSwapchainImagesKHR;
    /* Command buffer */
    void* BeginCommandBuffer;
    void* EndCommandBuffer;
    void* CmdDraw;
    void* CmdDrawIndexed;
    void* CmdDispatch;            /* Compute dispatch */
    void* CmdDispatchIndirect;    /* Indirect compute dispatch */
    void* CmdDrawIndirect;
    void* ResetCommandBuffer;
    void* AllocateCommandBuffers;
    void* FreeCommandBuffers;
    void* CreateCommandPool;
    void* DestroyCommandPool;
    /* Sync */
    void* CreateFence;
    void* DestroyFence;
    void* WaitForFences;
    void* ResetFences;
    void* CreateSemaphore;
    void* DestroySemaphore;
    /* Pipeline */
    void* CreateComputePipelines;
    void* CreateGraphicsPipelines;
    void* DestroyPipeline;
    void* CmdBindPipeline;
    void* CmdBindDescriptorSets;
    void* CmdPushConstants;
    /* Descriptor */
    void* CreateDescriptorSetLayout;
    void* DestroyDescriptorSetLayout;
    void* AllocateDescriptorSets;
    void* UpdateDescriptorSets;
    /* Buffer/Image */
    void* CreateBuffer;
    void* DestroyBuffer;
    void* AllocateMemory;
    void* FreeMemory;
    void* BindBufferMemory;
    void* MapMemory;
    void* UnmapMemory;
    void* CreateImage;
    void* DestroyImage;
    void* BindImageMemory;
    /* Query */
    void* CreateQueryPool;
    void* DestroyQueryPool;
    void* CmdResetQueryPool;
    void* CmdWriteTimestamp;
    void* GetQueryPoolResults;
    void* _pad[80];
} VkLayerDispatchTable;

typedef struct {
    VkLayerDispatchTable* pLayerDispatch;
    void* pNextChain;
    void* pLoaderData;
} VkLayerInstanceDispatchChain;

/* ── Globals ──────────────────────────────────────────── */
static VkLayerDispatchTable g_nextDispatch;
static FILE* g_logFile = NULL;
static int g_frameCount = 0;
static DWORD g_lastLogTime = 0;
static volatile LONG g_submittedCmds = 0;
static volatile LONG g_computeDispatches = 0;
static volatile LONG g_drawCalls = 0;
static volatile LONG g_totalTriangles = 0;

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

/* ── Intercepted: vkQueueSubmit (FPS tracking) ────────── */
typedef void* VkQueue;
typedef void* VkFence;
typedef void* VkCommandBuffer;
typedef void* VkResult;

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
        vklog("DistriBox VK: %.1f FPS | draws=%d | compute=%d | tris=%d\n",
            fps, g_drawCalls, g_computeDispatches, g_totalTriangles);
        g_frameCount = 0;
        g_drawCalls = 0;
        g_computeDispatches = 0;
        g_totalTriangles = 0;
    }
    if (g_lastLogTime == 0) g_lastLogTime = now;

    typedef VkResult (VKAPI_CALL *PFN_QueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
    PFN_QueueSubmit next = (PFN_QueueSubmit)g_nextDispatch.QueueSubmit;
    if (next) return next(queue, submitCount, pSubmits, fence);
    return 0;
}

/* ── Intercepted: vkCmdDispatch (compute workloads) ───── */
/* This is the key entry point for AI inference via Vulkan.
 * llama.cpp Vulkan backend calls this for each matrix multiply. */

static void VKAPI_CALL intercept_CmdDispatch(
    VkCommandBuffer cmdBuf,
    uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    InterlockedIncrement(&g_computeDispatches);

    vklog("DistriBox VK: CmdDispatch(groups=(%u,%u,%u))\n",
        groupCountX, groupCountY, groupCountZ);

    /* TODO: Distributed compute interception
     *
     * When DistriBox VGPU Core is running:
     * 1. Capture the compute pipeline state (bound descriptors, push constants)
     * 2. Send dispatch parameters to VGPU Core
     * 3. VGPU Core splits across workers
     * 4. Workers execute locally (real GPU)
     * 5. Results written back to device memory
     */

    typedef void (VKAPI_CALL *PFN_CmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
    PFN_CmdDispatch next = (PFN_CmdDispatch)g_nextDispatch.CmdDispatch;
    if (next) next(cmdBuf, groupCountX, groupCountY, groupCountZ);
}

/* ── Intercepted: vkCmdDispatchIndirect ───────────────── */
static void VKAPI_CALL intercept_CmdDispatchIndirect(
    VkCommandBuffer cmdBuf, void* buffer, VkDeviceSize offset)
{
    InterlockedIncrement(&g_computeDispatches);
    vklog("DistriBox VK: CmdDispatchIndirect(offset=%llu)\n",
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

    typedef void (VKAPI_CALL *PFN_CmdDraw)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t);
    PFN_CmdDraw next = (PFN_CmdDraw)g_nextDispatch.CmdDraw;
    if (next) next(cmdBuf, vertexCount, instanceCount, firstVertex, firstInstance);
}

static void VKAPI_CALL intercept_CmdDrawIndexed(
    VkCommandBuffer cmdBuf,
    uint32_t indexCount, uint32_t instanceCount,
    uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
    InterlockedIncrement(&g_drawCalls);
    InterlockedExchangeAdd(&g_totalTriangles, indexCount / 3);

    typedef void (VKAPI_CALL *PFN_CmdDrawIndexed)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, int32_t, uint32_t);
    PFN_CmdDrawIndexed next = (PFN_CmdDrawIndexed)g_nextDispatch.CmdDrawIndexed;
    if (next) next(cmdBuf, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

/* ── Layer entry points ───────────────────────────────── */
static void* VKAPI_CALL layer_GetInstanceProcAddr(void* instance, const char* pName)
{
    if (!pName) return NULL;

    typedef void* (VKAPI_CALL *PFN)(void*, const char*);
    PFN next = (PFN)g_nextDispatch.GetInstanceProcAddr;
    if (!next) return NULL;

    void* addr = next(instance, pName);

    /* Intercept key functions */
    if (strcmp(pName, "vkQueueSubmit") == 0) {
        g_nextDispatch.QueueSubmit = addr;
        return &intercept_QueueSubmit;
    }
    if (strcmp(pName, "vkCmdDispatch") == 0) {
        g_nextDispatch.CmdDispatch = addr;
        return &intercept_CmdDispatch;
    }
    if (strcmp(pName, "vkCmdDispatchIndirect") == 0) {
        g_nextDispatch.CmdDispatchIndirect = addr;
        return &intercept_CmdDispatchIndirect;
    }
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
    vklog("DistriBox VK Layer: loader negotiated (v%d)\n",
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
        vklog("=== DistriBox Vulkan Layer v0.2 (compute+trace) loaded (PID=%lu) ===\n",
            GetCurrentProcessId());
    }
    if (reason == DLL_PROCESS_DETACH) {
        vklog("=== DistriBox Vulkan Layer unloaded (total compute=%d, draws=%d) ===\n",
            g_computeDispatches, g_drawCalls);
        if (g_logFile) fclose(g_logFile);
    }
    return TRUE;
}
