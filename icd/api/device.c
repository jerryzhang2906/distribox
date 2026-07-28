/**
 * api/device.c — clGetDeviceIDs, clGetDeviceInfo
 *
 * This is where the virtual GPU becomes "visible" to applications.
 * We create a single virtual device with user-configurable specs.
 * Defaults are set here; they can be updated from VGPU Core via IPC.
 */

#include "../icd.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Default virtual GPU specs — calibrated to ~GTX 1050 Ti class
// (updated from VGPU Core on connect with actual cluster capability)
#define DEFAULT_VRAM_SIZE        (4ULL * 1024 * 1024 * 1024)    // 4 GB
#define DEFAULT_COMPUTE_UNITS    24                              // ~GTX 1050 Ti class
#define DEFAULT_CLOCK_MHZ        1392                            // Boost clock
#define DEFAULT_MAX_WG_SIZE      1024

// ── Device initialization (lazy) ─────────────────────

static void ensure_device(void) {
    if (g_device != NULL) return;

    g_device = (distri_device_t *)calloc(1, sizeof(distri_device_t));
    if (g_device == NULL) return;

    g_device->dispatch = g_platform->dispatch;
    g_device->platform = g_platform;

    // Default virtual GPU specs — will be overwritten if VGPU Core provides config
    snprintf(g_device->name, sizeof(g_device->name),
             "DistriBox Virtual GPU (NVIDIA Compatible)");
    snprintf(g_device->vendor, sizeof(g_device->vendor),
             "DistriBox Technologies");
    g_device->global_mem_size = DEFAULT_VRAM_SIZE;
    g_device->max_compute_units = DEFAULT_COMPUTE_UNITS;
    g_device->max_clock_frequency = DEFAULT_CLOCK_MHZ;
    g_device->max_work_group_size = DEFAULT_MAX_WG_SIZE;
    g_device->max_work_item_sizes[0] = 1024;
    g_device->max_work_item_sizes[1] = 1024;
    g_device->max_work_item_sizes[2] = 64;

    // Try to connect to VGPU Core for updated specs
    g_device->ipc_connected = false;
    g_device->ipc_socket = -1;
}

// ── Device enumeration ───────────────────────────────

cl_int distriboxGetDeviceIDs(cl_platform_id platform,
                              cl_device_type device_type,
                              cl_uint num_entries,
                              cl_device_id *devices,
                              cl_uint *num_devices) {
    if (platform == NULL) {
        return CL_INVALID_PLATFORM;
    }

    ensure_device();
    if (g_device == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
    }

    // We only expose ourselves as a GPU device
    // This means apps looking for CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_ALL will find us
    bool matches = (device_type == CL_DEVICE_TYPE_GPU) ||
                   (device_type == CL_DEVICE_TYPE_ALL) ||
                   (device_type == CL_DEVICE_TYPE_DEFAULT) ||
                   (device_type == CL_DEVICE_TYPE_ACCELERATOR);

    if (!matches) {
        if (num_devices) *num_devices = 0;
        return CL_DEVICE_NOT_FOUND;
    }

    // Only connect to VGPU Core if we have actual workers registered
    // For now, we always present the device — with or without workers
    // If no workers are available, kernels just execute on local CPU

    if (num_devices) {
        *num_devices = 1;
    }
    if (devices && num_entries >= 1) {
        devices[0] = (cl_device_id)g_device;
    }

    return CL_SUCCESS;
}

// ── Device info — the "spec sheet" of our virtual GPU ─

cl_int distriboxGetDeviceInfo(cl_device_id device,
                               cl_device_info param_name,
                               size_t param_value_size,
                               void *param_value,
                               size_t *param_value_size_ret) {
    if (device == NULL) {
        return CL_INVALID_DEVICE;
    }

    distri_device_t *d = (distri_device_t *)device;

#define RETURN_STRING(s) do { \
    size_t slen = strlen(s) + 1; \
    if (param_value_size_ret) *param_value_size_ret = slen; \
    if (param_value && param_value_size >= slen) memcpy(param_value, s, slen); \
    return CL_SUCCESS; \
} while(0)

#define RETURN_VAL(val, type) do { \
    if (param_value_size_ret) *param_value_size_ret = sizeof(type); \
    if (param_value && param_value_size >= sizeof(type)) memcpy(param_value, &(val), sizeof(type)); \
    return CL_SUCCESS; \
} while(0)

    switch (param_name) {
    // ── Identification ───────────────────────────
    case CL_DEVICE_NAME:
        RETURN_STRING(d->name);
    case CL_DEVICE_VENDOR:
        RETURN_STRING(d->vendor);
    case CL_DEVICE_VENDOR_ID:
        { cl_uint vid = 0x44425858; RETURN_VAL(vid, cl_uint); }  // "DBXX"
    case CL_DRIVER_VERSION:
        RETURN_STRING("1.0 DistriBox");
    case CL_DEVICE_VERSION:
        RETURN_STRING("OpenCL 2.0 DistriBox");

    // ── Type & capabilities ───────────────────────
    case CL_DEVICE_TYPE:
        { cl_device_type t = CL_DEVICE_TYPE_GPU; RETURN_VAL(t, cl_device_type); }
    case CL_DEVICE_AVAILABLE:
        { cl_bool b = CL_TRUE; RETURN_VAL(b, cl_bool); }
    case CL_DEVICE_COMPILER_AVAILABLE:
        { cl_bool b = CL_TRUE; RETURN_VAL(b, cl_bool); }
    case CL_DEVICE_LINKER_AVAILABLE:
        { cl_bool b = CL_TRUE; RETURN_VAL(b, cl_bool); }

    // ── Memory ────────────────────────────────────
    case CL_DEVICE_GLOBAL_MEM_SIZE:
        RETURN_VAL(d->global_mem_size, cl_ulong);
    case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
        { cl_ulong cache = d->global_mem_size / 16; RETURN_VAL(cache, cl_ulong); }
    case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
        { cl_device_mem_cache_type t = CL_READ_WRITE_CACHE; RETURN_VAL(t, cl_device_mem_cache_type); }
    case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
        { cl_uint cls = 64; RETURN_VAL(cls, cl_uint); }
    case CL_DEVICE_LOCAL_MEM_SIZE:
        { cl_ulong lms = 49152; RETURN_VAL(lms, cl_ulong); }  // 48 KB
    case CL_DEVICE_LOCAL_MEM_TYPE:
        { cl_device_local_mem_type t = CL_LOCAL; RETURN_VAL(t, cl_device_local_mem_type); }
    case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:
        { cl_ulong s = 65536; RETURN_VAL(s, cl_ulong); }
    case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
        { cl_ulong s = d->global_mem_size / 4; RETURN_VAL(s, cl_ulong); }
    case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
        { cl_uint a = 1024; RETURN_VAL(a, cl_uint); }  // 1 KB alignment
    case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
        { cl_uint a = 128; RETURN_VAL(a, cl_uint); }

    // ── Compute ────────────────────────────────────
    case CL_DEVICE_MAX_COMPUTE_UNITS:
        RETURN_VAL(d->max_compute_units, cl_uint);
    case CL_DEVICE_MAX_CLOCK_FREQUENCY:
        RETURN_VAL(d->max_clock_frequency, cl_uint);
    case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
        { cl_uint dim = 3; RETURN_VAL(dim, cl_uint); }
    case CL_DEVICE_MAX_WORK_ITEM_SIZES:
        { if (param_value_size_ret) *param_value_size_ret = 3 * sizeof(size_t);
          if (param_value && param_value_size >= 3 * sizeof(size_t)) {
              memcpy(param_value, d->max_work_item_sizes, 3 * sizeof(size_t));
          } return CL_SUCCESS; }
    case CL_DEVICE_MAX_WORK_GROUP_SIZE:
        RETURN_VAL(d->max_work_group_size, size_t);

    // ── Vector capabilities ───────────────────────
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
        { cl_uint w = 4; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
        { cl_uint w = 2; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
        { cl_uint w = 1; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
        { cl_uint w = 1; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
        { cl_uint w = 1; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
        { cl_uint w = 1; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
        { cl_uint w = 1; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
        { cl_uint w = 16; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
        { cl_uint w = 8; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
        { cl_uint w = 4; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
        { cl_uint w = 2; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
        { cl_uint w = 4; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
        { cl_uint w = 2; RETURN_VAL(w, cl_uint); }
    case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
        { cl_uint w = 8; RETURN_VAL(w, cl_uint); }

    // ── FP capabilities ────────────────────────────
    case CL_DEVICE_SINGLE_FP_CONFIG:
        { cl_device_fp_config fp = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO |
                CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN | CL_FP_DENORM; RETURN_VAL(fp, cl_device_fp_config); }
    case CL_DEVICE_DOUBLE_FP_CONFIG:
        { cl_device_fp_config fp = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO |
                CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN | CL_FP_DENORM; RETURN_VAL(fp, cl_device_fp_config); }

    // ── Execution ──────────────────────────────────
    case CL_DEVICE_QUEUE_ON_HOST_PROPERTIES:
        { cl_command_queue_properties p = CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
          RETURN_VAL(p, cl_command_queue_properties); }
    case CL_DEVICE_EXECUTION_CAPABILITIES:
        { cl_device_exec_capabilities e = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL; RETURN_VAL(e, cl_device_exec_capabilities); }

    // ── Extensions ─────────────────────────────────
    case CL_DEVICE_EXTENSIONS:
        RETURN_STRING("cl_khr_fp64 cl_khr_global_int32_base_atomics "
                       "cl_khr_global_int32_extended_atomics "
                       "cl_khr_local_int32_base_atomics "
                       "cl_khr_local_int32_extended_atomics "
                       "cl_khr_byte_addressable_store");
    case CL_DEVICE_OPENCL_C_VERSION:
        RETURN_STRING("OpenCL C 2.0");

    // ── Image support (we report none) ─────────────
    case CL_DEVICE_IMAGE_SUPPORT:
        { cl_bool b = CL_FALSE; RETURN_VAL(b, cl_bool); }
    case CL_DEVICE_MAX_READ_IMAGE_ARGS:
        { cl_uint n = 0; RETURN_VAL(n, cl_uint); }
    case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
        { cl_uint n = 0; RETURN_VAL(n, cl_uint); }
    case CL_DEVICE_IMAGE2D_MAX_WIDTH:
    case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
    case CL_DEVICE_IMAGE3D_MAX_WIDTH:
    case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
    case CL_DEVICE_IMAGE3D_MAX_DEPTH:
        { size_t n = 0; RETURN_VAL(n, size_t); }

    // ── Other defaults ─────────────────────────────
    case CL_DEVICE_MAX_SAMPLERS:
        { cl_uint n = 0; RETURN_VAL(n, cl_uint); }
    case CL_DEVICE_MAX_PARAMETER_SIZE:
        { size_t s = 4096; RETURN_VAL(s, size_t); }
    case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
        { size_t r = 1; RETURN_VAL(r, size_t); }  // nanosecond
    case CL_DEVICE_ADDRESS_BITS:
        { cl_uint b = 64; RETURN_VAL(b, cl_uint); }
    case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
        { cl_bool b = CL_FALSE; RETURN_VAL(b, cl_bool); }
    case CL_DEVICE_ENDIAN_LITTLE:
        { cl_bool b = CL_TRUE; RETURN_VAL(b, cl_bool); }
    case CL_DEVICE_HOST_UNIFIED_MEMORY:
        { cl_bool b = CL_TRUE; RETURN_VAL(b, cl_bool); }  // Virtual GPU always uses host memory

    default:
        return CL_INVALID_VALUE;
    }

#undef RETURN_STRING
#undef RETURN_VAL
}

// ── Sub-devices (not supported) ──────────────────────

cl_int distriboxCreateSubDevices(cl_device_id in_device,
                                  const cl_device_partition_property *properties,
                                  cl_uint num_devices,
                                  cl_device_id *out_devices,
                                  cl_uint *num_devices_ret) {
    if (num_devices_ret) *num_devices_ret = 0;
    return CL_INVALID_VALUE;
}

cl_int distriboxRetainDevice(cl_device_id device) {
    (void)device;
    return CL_SUCCESS;  // Singleton, no ref counting needed
}

cl_int distriboxReleaseDevice(cl_device_id device) {
    (void)device;
    return CL_SUCCESS;  // Singleton, never freed
}
