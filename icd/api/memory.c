/**
 * api/memory.c — Buffer creation, read, write, copy, fill
 *
 * This is where we manage "virtual VRAM". Buffers are tracked in host memory
 * and their contents are synchronized with workers via VGPU Core.
 */
#include "../icd.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// ── clCreateBuffer ───────────────────────────────────

cl_mem distriboxCreateBuffer(cl_context context,
                              cl_mem_flags flags,
                              size_t size,
                              void *host_ptr,
                              cl_int *errcode_ret) {
    if (context == NULL || size == 0) {
        if (errcode_ret) *errcode_ret = CL_INVALID_VALUE;
        return NULL;
    }

    distri_mem_t *mem = (distri_mem_t *)calloc(1, sizeof(distri_mem_t));
    if (mem == NULL) {
        if (errcode_ret) *errcode_ret = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }

    mem->dispatch = ((distri_context_t *)context)->dispatch;
    mem->context = (distri_context_t *)context;
    mem->type = CL_MEM_OBJECT_BUFFER;
    mem->flags = flags;
    mem->size = size;
    mem->ref_count = 1;

    generate_id(mem->buffer_id, sizeof(mem->buffer_id), "buf");

    // Allocate staging memory on host
    if (flags & CL_MEM_USE_HOST_PTR) {
        mem->host_ptr = host_ptr;      // Use application-provided memory
        mem->staging_data = NULL;       // No separate staging buffer
    } else {
        mem->host_ptr = NULL;
        mem->staging_data = calloc(1, size);  // Allocate staging
        if (mem->staging_data == NULL && size > 0) {
            free(mem);
            if (errcode_ret) *errcode_ret = CL_OUT_OF_HOST_MEMORY;
            return NULL;
        }

        // If CL_MEM_COPY_HOST_PTR, copy initial data
        if ((flags & CL_MEM_COPY_HOST_PTR) && host_ptr) {
            memcpy(mem->staging_data, host_ptr, size);
        }
    }

    // IPC: Notify VGPU Core about new buffer
    if (g_ipc_connected || ipc_connect() == 0) {
        char msg[512];
        const char *buf_type = "read_write";
        if (flags & CL_MEM_READ_ONLY) buf_type = "read_only";
        else if (flags & CL_MEM_WRITE_ONLY) buf_type = "write_only";
        int msg_len = snprintf(msg, sizeof(msg),
            "{\"type\":\"buffer_create\",\"msg_id\":\"%s\",\"buffer_id\":\"%s\","
            "\"size\":%llu,\"flags\":%lu,\"buffer_type\":\"%s\"}\n",
            mem->buffer_id, mem->buffer_id,
            (unsigned long long)size, (unsigned long)flags, buf_type);
        ipc_send_command(msg, (uint64_t)msg_len);
        char resp[256];
        ipc_recv_response(resp, sizeof(resp), 1000);

        // Mark as having local data that needs to be flushed to VGPU Core before kernel execution
        void *src = mem->staging_data ? mem->staging_data : mem->host_ptr;
        mem->staging_dirty = (src != NULL && size > 0);
    }

    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_mem)mem;
}

// ── clCreateSubBuffer ────────────────────────────────

cl_mem distriboxCreateSubBuffer(cl_mem buffer,
                                 cl_mem_flags flags,
                                 cl_buffer_create_type create_type,
                                 const void *create_info,
                                 cl_int *errcode_ret) {
    if (buffer == NULL || create_type != CL_BUFFER_CREATE_TYPE_REGION) {
        if (errcode_ret) *errcode_ret = CL_INVALID_VALUE;
        return NULL;
    }

    // For MVP, sub-buffers are not fully supported
    // Return unsupported for now
    if (errcode_ret) *errcode_ret = CL_INVALID_VALUE;
    return NULL;
}

// ── Retain/Release ───────────────────────────────────

cl_int distriboxRetainMemObject(cl_mem memobj) {
    if (memobj == NULL) return CL_INVALID_MEM_OBJECT;
    ((distri_mem_t *)memobj)->ref_count++;
    return CL_SUCCESS;
}

cl_int distriboxReleaseMemObject(cl_mem memobj) {
    if (memobj == NULL) return CL_INVALID_MEM_OBJECT;
    distri_mem_t *mem = (distri_mem_t *)memobj;
    mem->ref_count--;
    if (mem->ref_count == 0) {
        if (mem->staging_data && !(mem->flags & CL_MEM_USE_HOST_PTR)) {
            free(mem->staging_data);
        }
        free(mem);
    }
    return CL_SUCCESS;
}

// ── GetMemObjectInfo ─────────────────────────────────

cl_int distriboxGetMemObjectInfo(cl_mem memobj,
                                  cl_mem_info param_name,
                                  size_t param_value_size,
                                  void *param_value,
                                  size_t *param_value_size_ret) {
    if (memobj == NULL) return CL_INVALID_MEM_OBJECT;
    distri_mem_t *mem = (distri_mem_t *)memobj;

    switch (param_name) {
    case CL_MEM_TYPE: {
        cl_mem_object_type t = mem->type;
        if (param_value_size_ret) *param_value_size_ret = sizeof(t);
        if (param_value && param_value_size >= sizeof(t)) memcpy(param_value, &t, sizeof(t));
        return CL_SUCCESS;
    }
    case CL_MEM_FLAGS:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_mem_flags);
        if (param_value && param_value_size >= sizeof(cl_mem_flags))
            memcpy(param_value, &mem->flags, sizeof(cl_mem_flags));
        return CL_SUCCESS;
    case CL_MEM_SIZE:
        if (param_value_size_ret) *param_value_size_ret = sizeof(size_t);
        if (param_value && param_value_size >= sizeof(size_t))
            memcpy(param_value, &mem->size, sizeof(size_t));
        return CL_SUCCESS;
    case CL_MEM_HOST_PTR:
        if (param_value_size_ret) *param_value_size_ret = sizeof(void *);
        if (param_value && param_value_size >= sizeof(void *))
            memcpy(param_value, &mem->host_ptr, sizeof(void *));
        return CL_SUCCESS;
    case CL_MEM_REFERENCE_COUNT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
        if (param_value && param_value_size >= sizeof(cl_uint))
            memcpy(param_value, &mem->ref_count, sizeof(cl_uint));
        return CL_SUCCESS;
    case CL_MEM_CONTEXT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_context);
        if (param_value && param_value_size >= sizeof(cl_context))
            memcpy(param_value, &mem->context, sizeof(cl_context));
        return CL_SUCCESS;
    default:
        return CL_INVALID_VALUE;
    }
}

// ── clEnqueueReadBuffer ──────────────────────────────
// Read data from "virtual VRAM" back to host

cl_int distriboxEnqueueReadBuffer(cl_command_queue command_queue,
                                   cl_mem buffer,
                                   cl_bool blocking_read,
                                   size_t offset,
                                   size_t size,
                                   void *ptr,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event) {
    if (command_queue == NULL || buffer == NULL || ptr == NULL) {
        return CL_INVALID_VALUE;
    }

    distri_mem_t *mem = (distri_mem_t *)buffer;

    if (offset + size > mem->size) {
        return CL_INVALID_VALUE;
    }

    // IPC: Request data from VGPU Core (may trigger merge from workers)
    bool got_ipc_data = false;
    if (g_ipc_connected) {
        char msg[512];
        int msg_len = snprintf(msg, sizeof(msg),
            "{\"type\":\"buffer_read\",\"msg_id\":\"buf-rd-%s\",\"buffer_id\":\"%s\","
            "\"offset\":%llu,\"size\":%llu}\n",
            mem->buffer_id, mem->buffer_id,
            (unsigned long long)offset, (unsigned long long)size);
        ipc_send_command(msg, (uint64_t)msg_len);
        char resp[8192];
        int n = ipc_recv_response(resp, sizeof(resp), 5000);
        if (n > 0 && ptr) {
            // Parse hex-encoded "data_b64" field from JSON response
            char *data_start = strstr(resp, "\"data_b64\":\"");
            if (data_start) {
                data_start += 12; // Skip "data_b64":"
                char *data_end = strchr(data_start, '"');
                if (data_end) {
                    *data_end = '\0';
                    // Hex decode
                    size_t hex_len = strlen(data_start);
                    size_t decoded_len = hex_len / 2;
                    if (decoded_len <= size) {
                        for (size_t i = 0; i < decoded_len; i++) {
                            char hex_byte[3] = {data_start[i*2], data_start[i*2+1], '\0'};
                            ((unsigned char *)ptr)[i] = (unsigned char)strtol(hex_byte, NULL, 16);
                        }
                        got_ipc_data = true;
                    }
                }
            }
        }
    }

    // Fall back to local staging data if IPC didn't provide results
    if (!got_ipc_data) {
        void *src = mem->staging_data ? mem->staging_data : mem->host_ptr;
        if (src) {
            memcpy(ptr, (char *)src + offset, size);
        }
    }

    if (blocking_read == CL_TRUE) {
        // Data is already local (staging), return immediately
    }

    return CL_SUCCESS;
}

// ── clEnqueueWriteBuffer ─────────────────────────────
// Write data from host to "virtual VRAM"

cl_int distriboxEnqueueWriteBuffer(cl_command_queue command_queue,
                                    cl_mem buffer,
                                    cl_bool blocking_write,
                                    size_t offset,
                                    size_t size,
                                    const void *ptr,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event) {
    if (command_queue == NULL || buffer == NULL || ptr == NULL) {
        return CL_INVALID_VALUE;
    }

    distri_mem_t *mem = (distri_mem_t *)buffer;

    if (offset + size > mem->size) {
        return CL_INVALID_VALUE;
    }

    // Write to local staging
    void *dst = mem->staging_data ? mem->staging_data : mem->host_ptr;
    if (dst) {
        memcpy((char *)dst + offset, ptr, size);
    }

    // IPC: Notify VGPU Core to distribute updated buffer to workers
    if (g_ipc_connected) {
        char msg[1024];
        int msg_len = snprintf(msg, sizeof(msg),
            "{\"type\":\"buffer_write\",\"msg_id\":\"buf-wr-%s\",\"buffer_id\":\"%s\","
            "\"offset\":%llu,\"size\":%llu}\n",
            mem->buffer_id, mem->buffer_id,
            (unsigned long long)offset, (unsigned long long)size);
        ipc_send_command(msg, (uint64_t)msg_len);
        char resp[256];
        ipc_recv_response(resp, sizeof(resp), 2000);
    }

    return CL_SUCCESS;
}

// ── clEnqueueFillBuffer ──────────────────────────────

cl_int distriboxEnqueueFillBuffer(cl_command_queue command_queue,
                                   cl_mem buffer,
                                   const void *pattern,
                                   size_t pattern_size,
                                   size_t offset,
                                   size_t size,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event) {
    if (command_queue == NULL || buffer == NULL || pattern == NULL) {
        return CL_INVALID_VALUE;
    }

    distri_mem_t *mem = (distri_mem_t *)buffer;
    if (offset + size > mem->size) return CL_INVALID_VALUE;

    void *dst = mem->staging_data ? mem->staging_data : mem->host_ptr;
    if (dst) {
        // Fill with repeated pattern
        char *d = (char *)dst + offset;
        for (size_t i = 0; i < size; i += pattern_size) {
            size_t n = (i + pattern_size <= size) ? pattern_size : size - i;
            memcpy(d + i, pattern, n);
        }
    }

    return CL_SUCCESS;
}

// ── clEnqueueCopyBuffer ──────────────────────────────

cl_int distriboxEnqueueCopyBuffer(cl_command_queue command_queue,
                                   cl_mem src_buffer,
                                   cl_mem dst_buffer,
                                   size_t src_offset,
                                   size_t dst_offset,
                                   size_t size,
                                   cl_uint num_events_in_wait_list,
                                   const cl_event *event_wait_list,
                                   cl_event *event) {
    if (command_queue == NULL || src_buffer == NULL || dst_buffer == NULL) {
        return CL_INVALID_VALUE;
    }

    distri_mem_t *src = (distri_mem_t *)src_buffer;
    distri_mem_t *dst = (distri_mem_t *)dst_buffer;

    if (src_offset + size > src->size || dst_offset + size > dst->size) {
        return CL_INVALID_VALUE;
    }

    void *s = src->staging_data ? src->staging_data : src->host_ptr;
    void *d = dst->staging_data ? dst->staging_data : dst->host_ptr;
    if (s && d) {
        memcpy((char *)d + dst_offset, (char *)s + src_offset, size);
    }

    return CL_SUCCESS;
}
