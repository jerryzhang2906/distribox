/**
 * icd/proxy/opencl_proxy.c — OpenCL.dll proxy/wrapper
 *
 * Instead of relying on the Khronos ICD Loader (which may be too old),
 * we build our own OpenCL.dll that:
 *   1. Loads both Intel's OpenCL and DistriBox ICD
 *   2. Merges platform/device lists
 *   3. Routes API calls to the appropriate backend
 *
 * This makes the DistriBox Virtual GPU visible to ALL OpenCL applications
 * without requiring ICD loader compatibility.
 *
 * Build:
 *   zig cc -shared -O2 -I ../.. -I ../../third_party/include
 *          opencl_proxy.c -o OpenCL.dll
 *          -L $WINDIR/System32 -lOpenCL
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

// ── Original OpenCL functions (forwarded from Intel) ────

static HMODULE g_real_opencl = NULL;

#define FORWARD_FUNC(ret, name, ...) \
    static ret (CL_API_CALL *pfn_##name)(__VA_ARGS__) = NULL;

// We only need to intercept clGetPlatformIDs and clGetDeviceIDs
// Everything else goes through the dispatch table

// ── DistriBox ICD ───────────────────────────────────────

typedef cl_int (CL_API_CALL *PFN_clIcdGetPlatformIDsKHR)(cl_uint, cl_platform_id*, cl_uint*);

static HMODULE g_distribox_icd = NULL;
static PFN_clIcdGetPlatformIDsKHR g_pfnDistriGetPlatforms = NULL;
static cl_platform_id g_distri_platform = NULL;

// ── Init ────────────────────────────────────────────────

static void init_proxy(void) {
    static int initialized = 0;
    if (initialized) return;
    initialized = 1;

    // Load real Intel OpenCL (renamed to OpenCL_orig.dll or from absolute path)
    g_real_opencl = LoadLibraryA("OpenCL_orig.dll");
    if (!g_real_opencl) {
        // Try loading the one from System32 with a different name
        g_real_opencl = LoadLibraryA("C:\\Windows\\System32\\OpenCL_orig.dll");
    }
    if (!g_real_opencl) {
        // Last resort: load the real Intel ICD directly
        g_real_opencl = LoadLibraryA("IntelOpenCL64.dll");
    }
    if (g_real_opencl) {
        OutputDebugStringA("DistriBox: Loaded real OpenCL (Intel)\n");
    } else {
        OutputDebugStringA("DistriBox: No real OpenCL found — virtual GPU only\n");
    }

    // Load DistriBox ICD — try multiple locations
    g_distribox_icd = LoadLibraryA("distribox_icd.dll");
    if (!g_distribox_icd) {
        g_distribox_icd = LoadLibraryA("C:\\ProgramData\\DistriBox\\distribox_icd.dll");
    }
    if (!g_distribox_icd) {
        // Fallback: check LOCALAPPDATA
        char localPath[MAX_PATH];
        if (GetEnvironmentVariableA("LOCALAPPDATA", localPath, sizeof(localPath)) > 0) {
            char fullPath[MAX_PATH];
            snprintf(fullPath, sizeof(fullPath), "%s\\DistriBox\\distribox_icd.dll", localPath);
            g_distribox_icd = LoadLibraryA(fullPath);
        }
    }
    if (g_distribox_icd) {
        g_pfnDistriGetPlatforms = (PFN_clIcdGetPlatformIDsKHR)
            GetProcAddress(g_distribox_icd, "clIcdGetPlatformIDsKHR");
        if (g_pfnDistriGetPlatforms) {
            cl_uint num = 0;
            cl_int ret = g_pfnDistriGetPlatforms(1, &g_distri_platform, &num);
            if (ret == CL_SUCCESS && num > 0) {
                char buf[256];
                OutputDebugStringA("DistriBox: Virtual GPU platform loaded\n");
            } else {
                OutputDebugStringA("DistriBox: ICD returned no platforms\n");
            }
        }
    }
    if (!g_distribox_icd) {
        OutputDebugStringA("DistriBox: distribox_icd.dll not found\n");
    }
}

// ── clGetPlatformIDs (THE KEY INTERCEPTION) ─────────────

CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint num_entries,
                 cl_platform_id *platforms,
                 cl_uint *num_platforms) CL_API_SUFFIX__VERSION_1_0
{
    init_proxy();

    cl_uint total = 0;
    cl_platform_id intel_plat[4] = {NULL};
    cl_uint intel_count = 0;

    // Get Intel platforms
    if (g_real_opencl) {
        typedef cl_int (CL_API_CALL *PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
        PFN_clGetPlatformIDs pfn = (PFN_clGetPlatformIDs)
            GetProcAddress(g_real_opencl, "clGetPlatformIDs");
        if (pfn) {
            pfn(4, intel_plat, &intel_count);
        }
    }

    // Get DistriBox platform
    cl_platform_id distri_plat[2] = {NULL};
    cl_uint distri_count = 0;
    if (g_pfnDistriGetPlatforms) {
        g_pfnDistriGetPlatforms(2, distri_plat, &distri_count);
    }

    total = intel_count + distri_count;
    if (num_platforms) *num_platforms = total;

    if (platforms && num_entries > 0) {
        cl_uint idx = 0;
        // Intel platforms first
        for (cl_uint i = 0; i < intel_count && idx < num_entries; i++) {
            platforms[idx++] = intel_plat[i];
        }
        // DistriBox platform
        for (cl_uint i = 0; i < distri_count && idx < num_entries; i++) {
            platforms[idx++] = distri_plat[i];
        }
    }

    return CL_SUCCESS;
}

// ── clGetPlatformInfo ───────────────────────────────────

CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id platform,
                  cl_platform_info param_name,
                  size_t param_value_size,
                  void *param_value,
                  size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    init_proxy();

    // Route through dispatch table: first field of every ICD object
    // is a pointer to the dispatch table
    struct _cl_icd_dispatch **disp = (struct _cl_icd_dispatch **)platform;
    if (disp && *disp && (*disp)->clGetPlatformInfo) {
        return (*disp)->clGetPlatformInfo(platform, param_name,
            param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

// ── DLL Entry point ─────────────────────────────────────

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    (void)hinstDLL; (void)lpvReserved;
    if (fdwReason == DLL_PROCESS_ATTACH) {
        // Don't init here — LoadLibrary during DllMain can deadlock
        DisableThreadLibraryCalls(hinstDLL);
    }
    return TRUE;
}

// ── Export clIcdGetPlatformIDsKHR ────────────────────────
// (So the Khronos loader can also load us as an ICD)

CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint num_entries,
                       cl_platform_id *platforms,
                       cl_uint *num_platforms) CL_API_SUFFIX__VERSION_1_0
{
    init_proxy();
    if (g_pfnDistriGetPlatforms) {
        return g_pfnDistriGetPlatforms(num_entries, platforms, num_platforms);
    }
    if (num_platforms) *num_platforms = 0;
    return CL_SUCCESS;
}
