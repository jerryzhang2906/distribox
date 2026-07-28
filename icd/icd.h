#ifndef DISTRIBOX_ICD_H
#define DISTRIBOX_ICD_H

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────
// ICD internal object types — wrap OpenCL opaque handles
// ─────────────────────────────────────────────────────────

// Forward declaration — full definition in icd_dispatch.h
struct _cl_icd_dispatch;

// Platform
typedef struct _distri_platform {
    struct _cl_icd_dispatch *dispatch;
    char name[256];
    char vendor[64];
    char version[64];
    char icd_suffix[64];         // OpenCL ICD extension suffix
} distri_platform_t;

// Device
typedef struct _distri_device {
    struct _cl_icd_dispatch *dispatch;
    distri_platform_t *platform;

    // Virtual device configuration (set by user / auto-computed)
    char name[256];
    char vendor[64];
    uint64_t global_mem_size;     // "VRAM" size
    uint32_t max_compute_units;
    uint32_t max_clock_frequency;
    uint64_t max_work_group_size;
    uint64_t max_work_item_sizes[3];

    // IPC connection state
    int ipc_socket;               // Unix socket or Named Pipe fd
    bool ipc_connected;
} distri_device_t;

// Context
typedef struct _distri_context {
    struct _cl_icd_dispatch *dispatch;
    distri_device_t *device;
    uint32_t ref_count;
} distri_context_t;

// Command Queue
typedef struct _distri_command_queue {
    struct _cl_icd_dispatch *dispatch;
    distri_context_t *context;
    cl_command_queue_properties properties;
    uint32_t ref_count;

    // Queue ID for IPC communication
    char queue_id[64];
} distri_command_queue_t;

// Memory object (Buffer)
typedef struct _distri_mem {
    struct _cl_icd_dispatch *dispatch;
    distri_context_t *context;
    cl_mem_object_type type;
    cl_mem_flags flags;
    uint64_t size;
    void *host_ptr;              // For CL_MEM_USE_HOST_PTR

    // Buffer ID for IPC
    char buffer_id[64];

    // Local staging memory (host-side copy of the buffer data)
    void *staging_data;
    bool staging_dirty;

    uint32_t ref_count;
} distri_mem_t;

// Program
typedef struct _distri_program {
    struct _cl_icd_dispatch *dispatch;
    distri_context_t *context;
    char *source;                // OpenCL C source
    uint64_t source_len;
    char *options;               // Compile options

    // Program ID for IPC
    char program_id[64];

    // Compilation state
    bool compiled;
    char build_log[4096];
    cl_build_status build_status;

    uint32_t ref_count;
} distri_program_t;

// Kernel
typedef struct _distri_kernel {
    struct _cl_icd_dispatch *dispatch;
    distri_program_t *program;
    char name[256];

    // Kernel ID for IPC
    char kernel_id[64];

    // Argument storage (needed for clSetKernelArg)
    // We store args locally and send them with NDRange commands
    struct {
        void *data;
        uint64_t size;
        bool is_buffer;          // true if arg is a cl_mem
        char buffer_id[64];      // if is_buffer, the mem object's buffer_id
    } args[32];                  // Reasonable max for AI kernels
    uint32_t num_args;

    uint32_t ref_count;
} distri_kernel_t;

// Event
typedef struct _distri_event {
    struct _cl_icd_dispatch *dispatch;
    distri_context_t *context;
    cl_int status;                // CL_QUEUED, CL_SUBMITTED, CL_RUNNING, CL_COMPLETE

    // Event ID for IPC
    char event_id[64];

    // Profiling timestamps
    cl_ulong queued_time;
    cl_ulong submit_time;
    cl_ulong start_time;
    cl_ulong end_time;

    uint32_t ref_count;
} distri_event_t;

// ─────────────────────────────────────────────────────────
// Global state
// ─────────────────────────────────────────────────────────

// A single virtual platform with a single virtual device
extern distri_platform_t *g_platform;
extern distri_device_t *g_device;

// IPC connection to Virtual GPU Core
extern int g_ipc_fd;
extern bool g_ipc_connected;

// ─────────────────────────────────────────────────────────
// ICD dispatch table initialization
// ─────────────────────────────────────────────────────────

void icd_init_dispatch(void);

// ─────────────────────────────────────────────────────────
// IPC helper functions
// ─────────────────────────────────────────────────────────

int ipc_connect(void);
int ipc_send_command(const char *json, uint64_t len);
int ipc_recv_response(char *buf, uint64_t max_len, int timeout_ms);
void ipc_disconnect(void);

// Buffer data transfer via shared memory
int ipc_shm_write(const char *shm_name, const void *data, uint64_t size);
int ipc_shm_read(const char *shm_name, void *buf, uint64_t size, uint64_t offset);
void ipc_shm_unlink(const char *shm_name);

// ─────────────────────────────────────────────────────────
// Utility: generate unique IDs
// ─────────────────────────────────────────────────────────

void generate_id(char *buf, size_t len, const char *prefix);
void generate_shm_name(char *buf, size_t len, const char *prefix);

// ─────────────────────────────────────────────────────────
// API function declarations (implemented in api/*.c)
// ─────────────────────────────────────────────────────────

// platform.c
cl_int distriboxGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
cl_int distriboxGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

// device.c
cl_int distriboxGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);
cl_int distriboxGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_int distriboxCreateSubDevices(cl_device_id in_device, const cl_device_partition_property *properties, cl_uint num_devices, cl_device_id *out_devices, cl_uint *num_devices_ret);
cl_int distriboxRetainDevice(cl_device_id device);
cl_int distriboxReleaseDevice(cl_device_id device);

// context.c
cl_context distriboxCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret);
cl_context distriboxCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type, void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret);
cl_int distriboxRetainContext(cl_context context);
cl_int distriboxReleaseContext(cl_context context);
cl_int distriboxGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

// commandqueue.c
cl_command_queue distriboxCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret);
cl_command_queue distriboxCreateCommandQueueWithProperties(cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret);
cl_int distriboxRetainCommandQueue(cl_command_queue command_queue);
cl_int distriboxReleaseCommandQueue(cl_command_queue command_queue);
cl_int distriboxGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_int distriboxFinish(cl_command_queue command_queue);
cl_int distriboxFlush(cl_command_queue command_queue);

// memory.c
cl_mem distriboxCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);
cl_mem distriboxCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type create_type, const void *create_info, cl_int *errcode_ret);
cl_int distriboxRetainMemObject(cl_mem memobj);
cl_int distriboxReleaseMemObject(cl_mem memobj);
cl_int distriboxGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_int distriboxEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
cl_int distriboxEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
cl_int distriboxEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void *pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
cl_int distriboxEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

// program.c
cl_program distriboxCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret);
cl_program distriboxCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *errcode_ret);
cl_int distriboxRetainProgram(cl_program program);
cl_int distriboxReleaseProgram(cl_program program);
cl_int distriboxBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void (*pfn_notify)(cl_program, void *), void *user_data);
cl_int distriboxGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_int distriboxGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

// kernel.c
cl_kernel distriboxCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret);
cl_int distriboxRetainKernel(cl_kernel kernel);
cl_int distriboxReleaseKernel(cl_kernel kernel);
cl_int distriboxSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);
cl_int distriboxGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_int distriboxGetKernelArgInfo(cl_kernel kernel, cl_uint arg_index, cl_kernel_arg_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_int distriboxEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);
cl_int distriboxEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event);

// event.c
cl_int distriboxWaitForEvents(cl_uint num_events, const cl_event *event_list);
cl_int distriboxGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_int distriboxRetainEvent(cl_event event);
cl_int distriboxReleaseEvent(cl_event event);
cl_int distriboxGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_event distriboxCreateUserEvent(cl_context context, cl_int *errcode_ret);
cl_int distriboxSetUserEventStatus(cl_event event, cl_int execution_status);

#ifdef __cplusplus
}
#endif

#endif // DISTRIBOX_ICD_H
