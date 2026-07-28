/**
 * icd.c — OpenCL ICD entry point and dispatch table
 *
 * This is the main entry point for the DistriBox OpenCL ICD.
 * The ICD loader (libOpenCL.so) discovers this library and calls
 * clIcdGetPlatformIDsKHR to enumerate platforms.
 *
 * Every OpenCL object we create has a dispatch table as its first field,
 * which the ICD loader uses to route API calls to our implementations.
 */

#include "icd.h"
#include "icd_dispatch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ─────────────────────────────────────────────────────────
// Global state
// ─────────────────────────────────────────────────────────

distri_platform_t *g_platform = NULL;
distri_device_t *g_device = NULL;
int g_ipc_fd = -1;
bool g_ipc_connected = false;

// ─────────────────────────────────────────────────────────
// API function declarations (implemented in api/*.c)
// ─────────────────────────────────────────────────────────

// platform.c
extern cl_int distriboxGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
extern cl_int distriboxGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

// device.c
extern cl_int distriboxGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);
extern cl_int distriboxGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
extern cl_int distriboxCreateSubDevices(cl_device_id in_device, const cl_device_partition_property *properties, cl_uint num_devices, cl_device_id *out_devices, cl_uint *num_devices_ret);
extern cl_int distriboxRetainDevice(cl_device_id device);
extern cl_int distriboxReleaseDevice(cl_device_id device);

// context.c
extern cl_context distriboxCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret);
extern cl_context distriboxCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type, void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret);
extern cl_int distriboxRetainContext(cl_context context);
extern cl_int distriboxReleaseContext(cl_context context);
extern cl_int distriboxGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

// commandqueue.c
extern cl_command_queue distriboxCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret);
extern cl_command_queue distriboxCreateCommandQueueWithProperties(cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret);
extern cl_int distriboxRetainCommandQueue(cl_command_queue command_queue);
extern cl_int distriboxReleaseCommandQueue(cl_command_queue command_queue);
extern cl_int distriboxGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
extern cl_int distriboxFinish(cl_command_queue command_queue);
extern cl_int distriboxFlush(cl_command_queue command_queue);

// memory.c
extern cl_mem distriboxCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);
extern cl_mem distriboxCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type create_type, const void *create_info, cl_int *errcode_ret);
extern cl_int distriboxRetainMemObject(cl_mem memobj);
extern cl_int distriboxReleaseMemObject(cl_mem memobj);
extern cl_int distriboxGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
extern cl_int distriboxEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
extern cl_int distriboxEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
extern cl_int distriboxEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void *pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
extern cl_int distriboxEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

// program.c
extern cl_program distriboxCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret);
extern cl_program distriboxCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *errcode_ret);
extern cl_int distriboxRetainProgram(cl_program program);
extern cl_int distriboxReleaseProgram(cl_program program);
extern cl_int distriboxBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void (*pfn_notify)(cl_program, void *), void *user_data);
extern cl_int distriboxGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
extern cl_int distriboxGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

// kernel.c
extern cl_kernel distriboxCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret);
extern cl_int distriboxRetainKernel(cl_kernel kernel);
extern cl_int distriboxReleaseKernel(cl_kernel kernel);
extern cl_int distriboxSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);
extern cl_int distriboxGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
extern cl_int distriboxGetKernelArgInfo(cl_kernel kernel, cl_uint arg_index, cl_kernel_arg_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
extern cl_int distriboxEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
extern cl_int distriboxEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

// event.c
extern cl_int distriboxWaitForEvents(cl_uint num_events, const cl_event *event_list);
extern cl_int distriboxGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
extern cl_int distriboxRetainEvent(cl_event event);
extern cl_int distriboxReleaseEvent(cl_event event);
extern cl_int distriboxGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
extern cl_event distriboxCreateUserEvent(cl_context context, cl_int *errcode_ret);
extern cl_int distriboxSetUserEventStatus(cl_event event, cl_int execution_status);

// ─────────────────────────────────────────────────────────
// Dispatch table — maps OpenCL API → our implementation
// ─────────────────────────────────────────────────────────

static struct _cl_icd_dispatch g_dispatch = {
    // Platform
    .clGetPlatformIDs = distriboxGetPlatformIDs,
    .clGetPlatformInfo = distriboxGetPlatformInfo,

    // Device
    .clGetDeviceIDs = distriboxGetDeviceIDs,
    .clGetDeviceInfo = distriboxGetDeviceInfo,
    .clCreateSubDevices = distriboxCreateSubDevices,
    .clRetainDevice = distriboxRetainDevice,
    .clReleaseDevice = distriboxReleaseDevice,

    // Context
    .clCreateContext = distriboxCreateContext,
    .clCreateContextFromType = distriboxCreateContextFromType,
    .clRetainContext = distriboxRetainContext,
    .clReleaseContext = distriboxReleaseContext,
    .clGetContextInfo = distriboxGetContextInfo,

    // Command Queue
    .clCreateCommandQueue = distriboxCreateCommandQueue,
    .clCreateCommandQueueWithProperties = distriboxCreateCommandQueueWithProperties,
    .clRetainCommandQueue = distriboxRetainCommandQueue,
    .clReleaseCommandQueue = distriboxReleaseCommandQueue,
    .clGetCommandQueueInfo = distriboxGetCommandQueueInfo,
    .clFinish = distriboxFinish,
    .clFlush = distriboxFlush,

    // Memory
    .clCreateBuffer = distriboxCreateBuffer,
    .clCreateSubBuffer = distriboxCreateSubBuffer,
    .clRetainMemObject = distriboxRetainMemObject,
    .clReleaseMemObject = distriboxReleaseMemObject,
    .clGetMemObjectInfo = distriboxGetMemObjectInfo,
    .clEnqueueReadBuffer = distriboxEnqueueReadBuffer,
    .clEnqueueWriteBuffer = distriboxEnqueueWriteBuffer,
    .clEnqueueFillBuffer = distriboxEnqueueFillBuffer,
    .clEnqueueCopyBuffer = distriboxEnqueueCopyBuffer,

    // Program
    .clCreateProgramWithSource = distriboxCreateProgramWithSource,
    .clCreateProgramWithBinary = distriboxCreateProgramWithBinary,
    .clRetainProgram = distriboxRetainProgram,
    .clReleaseProgram = distriboxReleaseProgram,
    .clBuildProgram = distriboxBuildProgram,
    .clGetProgramBuildInfo = distriboxGetProgramBuildInfo,
    .clGetProgramInfo = distriboxGetProgramInfo,

    // Kernel
    .clCreateKernel = distriboxCreateKernel,
    .clRetainKernel = distriboxRetainKernel,
    .clReleaseKernel = distriboxReleaseKernel,
    .clSetKernelArg = distriboxSetKernelArg,
    .clGetKernelInfo = distriboxGetKernelInfo,
    .clGetKernelArgInfo = distriboxGetKernelArgInfo,
    .clEnqueueNDRangeKernel = distriboxEnqueueNDRangeKernel,
    .clEnqueueTask = distriboxEnqueueTask,

    // Event
    .clWaitForEvents = distriboxWaitForEvents,
    .clGetEventInfo = distriboxGetEventInfo,
    .clRetainEvent = distriboxRetainEvent,
    .clReleaseEvent = distriboxReleaseEvent,
    .clGetEventProfilingInfo = distriboxGetEventProfilingInfo,
    .clCreateUserEvent = distriboxCreateUserEvent,
    .clSetUserEventStatus = distriboxSetUserEventStatus,

    // The rest are NULL (unsupported) — ICD loader will handle gracefully
};

// ─────────────────────────────────────────────────────────
// ICD initialization — called once on library load
// ─────────────────────────────────────────────────────────

void icd_init_dispatch(void) {
    // Platform is a singleton — created once
    if (g_platform == NULL) {
        g_platform = (distri_platform_t *)calloc(1, sizeof(distri_platform_t));
        if (g_platform) {
            g_platform->dispatch = &g_dispatch;
            strncpy(g_platform->name, "DistriBox Virtual GPU Platform", sizeof(g_platform->name) - 1);
            strncpy(g_platform->vendor, "DistriBox", sizeof(g_platform->vendor) - 1);
            strncpy(g_platform->version, "OpenCL 2.0 DistriBox", sizeof(g_platform->version) - 1);
            strncpy(g_platform->icd_suffix, "DISTRIBOX", sizeof(g_platform->icd_suffix) - 1);
        }
    }

    // Device — initialized lazily on first clGetDeviceIDs
    // The virtual device config will be read from VGPU Core via IPC
}

// ─────────────────────────────────────────────────────────
// ICD Loader entry point
// ─────────────────────────────────────────────────────────

// Explicit dllexport — CL_API_ENTRY is for import, but ICD builds need export
#ifdef _WIN32
__declspec(dllexport)
#endif
cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint num_entries,
                        cl_platform_id *platforms,
                        cl_uint *num_platforms) CL_API_SUFFIX__VERSION_1_0
{
    icd_init_dispatch();

    if (num_platforms) {
        *num_platforms = 1; // One virtual platform
    }

    if (platforms && num_entries > 0) {
        platforms[0] = (cl_platform_id)g_platform;
    }

    return CL_SUCCESS;
}

// ─────────────────────────────────────────────────────────
// Utility: generate unique IDs for objects
// ─────────────────────────────────────────────────────────

void generate_id(char *buf, size_t len, const char *prefix) {
    static uint64_t counter = 0;
    snprintf(buf, len, "%s-%llu", prefix, (unsigned long long)++counter);
}

void generate_shm_name(char *buf, size_t len, const char *prefix) {
    static uint64_t shm_counter = 0;
    snprintf(buf, len, "/distribox_%s_%llu", prefix, (unsigned long long)++shm_counter);
}
