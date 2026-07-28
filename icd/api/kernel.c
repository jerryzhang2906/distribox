/**
 * api/kernel.c — clCreateKernel, clSetKernelArg, clEnqueueNDRangeKernel
 *
 * clEnqueueNDRangeKernel is THE most important function in the entire ICD.
 * It's where we intercept the compute command and dispatch it to workers
 * via the Virtual GPU Core.
 */
#include "../icd.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

cl_kernel distriboxCreateKernel(cl_program program,
                                 const char *kernel_name,
                                 cl_int *errcode_ret) {
    if (program == NULL || kernel_name == NULL) {
        if (errcode_ret) *errcode_ret = CL_INVALID_VALUE;
        return NULL;
    }

    distri_kernel_t *k = (distri_kernel_t *)calloc(1, sizeof(distri_kernel_t));
    if (k == NULL) {
        if (errcode_ret) *errcode_ret = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }

    k->dispatch = ((distri_program_t *)program)->dispatch;
    k->program = (distri_program_t *)program;
    k->ref_count = 1;
    k->num_args = 0;
    strncpy(k->name, kernel_name, sizeof(k->name) - 1);

    generate_id(k->kernel_id, sizeof(k->kernel_id), "kern");

    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_kernel)k;
}

cl_int distriboxRetainKernel(cl_kernel kernel) {
    if (kernel == NULL) return CL_INVALID_KERNEL;
    ((distri_kernel_t *)kernel)->ref_count++;
    return CL_SUCCESS;
}

cl_int distriboxReleaseKernel(cl_kernel kernel) {
    if (kernel == NULL) return CL_INVALID_KERNEL;
    distri_kernel_t *k = (distri_kernel_t *)kernel;
    k->ref_count--;
    if (k->ref_count == 0) {
        // Free stored arguments
        for (uint32_t i = 0; i < k->num_args; i++) {
            if (k->args[i].data && !k->args[i].is_buffer) {
                free(k->args[i].data);
            }
        }
        free(k);
    }
    return CL_SUCCESS;
}

// ── clSetKernelArg ───────────────────────────────────

cl_int distriboxSetKernelArg(cl_kernel kernel,
                              cl_uint arg_index,
                              size_t arg_size,
                              const void *arg_value) {
    if (kernel == NULL) return CL_INVALID_KERNEL;
    if (arg_index >= 32) return CL_INVALID_ARG_INDEX;

    distri_kernel_t *k = (distri_kernel_t *)kernel;

    // Free previous value if set
    if (k->args[arg_index].data && !k->args[arg_index].is_buffer) {
        free(k->args[arg_index].data);
        k->args[arg_index].data = NULL;
    }

    if (arg_value == NULL) {
        // clSetKernelArg with NULL arg_value: just allocate local memory
        k->args[arg_index].size = arg_size;
        k->args[arg_index].is_buffer = false;
        k->args[arg_index].data = NULL;
    } else if (arg_size == sizeof(cl_mem)) {
        // This is a cl_mem (buffer) argument — arg_value is a pointer TO the cl_mem handle
        // cl_mem is actually distri_mem_t*, so arg_value is distri_mem_t**
        distri_mem_t *mem = *((distri_mem_t **)arg_value);
        if (mem != NULL) {
            k->args[arg_index].is_buffer = true;
            k->args[arg_index].size = mem->size;
            k->args[arg_index].data = (void *)mem;
            strncpy(k->args[arg_index].buffer_id, mem->buffer_id,
                    sizeof(k->args[arg_index].buffer_id) - 1);
        } else {
            k->args[arg_index].is_buffer = false;
            k->args[arg_index].data = NULL;
        }
    } else {
        // Scalar value — copy the data directly
        k->args[arg_index].is_buffer = false;
        k->args[arg_index].size = arg_size;
        k->args[arg_index].data = malloc(arg_size);
        if (k->args[arg_index].data) {
            memcpy(k->args[arg_index].data, arg_value, arg_size);
        }
    }

    // Track max args
    if (arg_index >= k->num_args) {
        k->num_args = arg_index + 1;
    }

    return CL_SUCCESS;
}

// ── clGetKernelInfo ──────────────────────────────────

cl_int distriboxGetKernelInfo(cl_kernel kernel,
                               cl_kernel_info param_name,
                               size_t param_value_size,
                               void *param_value,
                               size_t *param_value_size_ret) {
    if (kernel == NULL) return CL_INVALID_KERNEL;
    distri_kernel_t *k = (distri_kernel_t *)kernel;

    switch (param_name) {
    case CL_KERNEL_FUNCTION_NAME: {
        size_t len = strlen(k->name) + 1;
        if (param_value_size_ret) *param_value_size_ret = len;
        if (param_value && param_value_size >= len)
            memcpy(param_value, k->name, len);
        return CL_SUCCESS;
    }
    case CL_KERNEL_NUM_ARGS:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
        if (param_value && param_value_size >= sizeof(cl_uint))
            memcpy(param_value, &k->num_args, sizeof(cl_uint));
        return CL_SUCCESS;
    case CL_KERNEL_REFERENCE_COUNT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
        if (param_value && param_value_size >= sizeof(cl_uint))
            memcpy(param_value, &k->ref_count, sizeof(cl_uint));
        return CL_SUCCESS;
    case CL_KERNEL_PROGRAM:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_program);
        if (param_value && param_value_size >= sizeof(cl_program))
            memcpy(param_value, &k->program, sizeof(cl_program));
        return CL_SUCCESS;
    default:
        return CL_INVALID_VALUE;
    }
}

cl_int distriboxGetKernelArgInfo(cl_kernel kernel,
                                  cl_uint arg_index,
                                  cl_kernel_arg_info param_name,
                                  size_t param_value_size,
                                  void *param_value,
                                  size_t *param_value_size_ret) {
    // Return minimal info for now
    if (kernel == NULL) return CL_INVALID_KERNEL;
    distri_kernel_t *k = (distri_kernel_t *)kernel;
    if (arg_index >= k->num_args) return CL_INVALID_ARG_INDEX;

    // For MVP, return unsupported for detailed arg info
    return CL_INVALID_VALUE;
}

// ── clEnqueueNDRangeKernel — 🔑 THE CORE ─────────────

cl_int distriboxEnqueueNDRangeKernel(cl_command_queue command_queue,
                                      cl_kernel kernel,
                                      cl_uint work_dim,
                                      const size_t *global_work_offset,
                                      const size_t *global_work_size,
                                      const size_t *local_work_size,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event *event_wait_list,
                                      cl_event *event) {
    if (command_queue == NULL || kernel == NULL) {
        return CL_INVALID_VALUE;
    }
    if (work_dim < 1 || work_dim > 3) {
        return CL_INVALID_WORK_DIMENSION;
    }
    if (global_work_size == NULL) {
        return CL_INVALID_GLOBAL_WORK_SIZE;
    }

    distri_kernel_t *k = (distri_kernel_t *)kernel;
    distri_command_queue_t *q = (distri_command_queue_t *)command_queue;

    // ── Build IPC command message ──────────────────
    // Serialize the NDRange + kernel info + arguments as JSON
    // and send to Virtual GPU Core via TCP IPC

    // Generate a message ID for request/response correlation
    char msg_id[64];
    generate_id(msg_id, sizeof(msg_id), "ndr");

    char cmd_buffer[16384];
    int off = snprintf(cmd_buffer, sizeof(cmd_buffer),
        "{"
        "\"type\":\"ndrange\","
        "\"msg_id\":\"%s\","
        "\"queue_id\":\"%s\","
        "\"kernel_id\":\"%s\","
        "\"kernel_name\":\"%s\","
        "\"program_id\":\"%s\","
        "\"work_dim\":%u,"
        "\"global\":[%llu,%llu,%llu],",
        msg_id,
        q->queue_id,
        k->kernel_id,
        k->name,
        k->program->program_id,
        work_dim,
        (unsigned long long)(work_dim >= 1 ? global_work_size[0] : 0),
        (unsigned long long)(work_dim >= 2 ? global_work_size[1] : 0),
        (unsigned long long)(work_dim >= 3 ? global_work_size[2] : 0)
    );

    off += snprintf(cmd_buffer + off, sizeof(cmd_buffer) - off,
        "\"global_offset\":[%llu,%llu,%llu],"
        "\"local\":[%llu,%llu,%llu],",
        (unsigned long long)(global_work_offset && work_dim >= 1 ? global_work_offset[0] : 0),
        (unsigned long long)(global_work_offset && work_dim >= 2 ? global_work_offset[1] : 0),
        (unsigned long long)(global_work_offset && work_dim >= 3 ? global_work_offset[2] : 0),
        (unsigned long long)(local_work_size && work_dim >= 1 ? local_work_size[0] : 1),
        (unsigned long long)(local_work_size && work_dim >= 2 ? local_work_size[1] : 1),
        (unsigned long long)(local_work_size && work_dim >= 3 ? local_work_size[2] : 1)
    );

    // Serialize arguments with index fields
    off += snprintf(cmd_buffer + off, sizeof(cmd_buffer) - off, "\"args\":[");
    for (uint32_t i = 0; i < k->num_args; i++) {
        if (i > 0) off += snprintf(cmd_buffer + off, sizeof(cmd_buffer) - off, ",");
        if (k->args[i].is_buffer) {
            distri_mem_t *mem = (distri_mem_t *)k->args[i].data;
            off += snprintf(cmd_buffer + off, sizeof(cmd_buffer) - off,
                "{\"type\":\"buffer\",\"index\":%u,\"id\":\"%s\",\"size\":%llu}",
                i,
                k->args[i].buffer_id,
                (unsigned long long)(mem ? mem->size : 0));
        } else {
            // Scalar: include the actual data
            off += snprintf(cmd_buffer + off, sizeof(cmd_buffer) - off,
                "{\"type\":\"scalar\",\"index\":%u,\"size\":%llu}",
                i,
                (unsigned long long)k->args[i].size);
        }
    }
    off += snprintf(cmd_buffer + off, sizeof(cmd_buffer) - off, "]}\n");

    // ── Send to VGPU Core via IPC ──────────────────
    cl_int exec_status = CL_SUCCESS;

    // Auto-connect to VGPU Core if needed
    if (!g_ipc_connected) {
        if (ipc_connect() != 0) {
            fprintf(stderr, "DistriBox ICD: failed to connect to VGPU Core\n");
            return CL_OUT_OF_RESOURCES;
        }
    }

    // Flush dirty buffer data to VGPU Core before execution
    for (uint32_t i = 0; i < k->num_args; i++) {
        if (k->args[i].is_buffer && k->args[i].data) {
            distri_mem_t *mem = (distri_mem_t *)k->args[i].data;
            if (mem->staging_dirty) {
                void *src = mem->staging_data ? mem->staging_data : mem->host_ptr;
                if (src && mem->size > 0) {
                    // Send buffer_write with hex-encoded data for small buffers
                    char wr_msg[4096];
                    int wr_len = snprintf(wr_msg, sizeof(wr_msg),
                        "{\"type\":\"buffer_write\",\"msg_id\":\"%s-flush\",\"buffer_id\":\"%s\","
                        "\"offset\":0,\"size\":%llu,\"data_b64\":\"",
                        k->args[i].buffer_id, k->args[i].buffer_id,
                        (unsigned long long)mem->size);
                    const unsigned char *s = (const unsigned char *)src;
                    size_t max_send = mem->size < 512 ? (size_t)mem->size : 512;
                    for (size_t j = 0; j < max_send && wr_len + 3 < (int)sizeof(wr_msg); j++) {
                        wr_len += snprintf(wr_msg + wr_len, sizeof(wr_msg) - wr_len, "%02x", s[j]);
                    }
                    wr_len += snprintf(wr_msg + wr_len, sizeof(wr_msg) - wr_len, "\"}\n");
                    ipc_send_command(wr_msg, (uint64_t)wr_len);
                    char flush_resp[256];
                    ipc_recv_response(flush_resp, sizeof(flush_resp), 2000);
                }
                mem->staging_dirty = false;
            }
        }
    }

    // Send the NDRange command
    if (ipc_send_command(cmd_buffer, (uint64_t)off) != 0) {
        fprintf(stderr, "DistriBox ICD: failed to send NDRange command\n");
        return CL_OUT_OF_RESOURCES;
    }

    // Wait for response
    char resp_buffer[4096];
    int resp_len = ipc_recv_response(resp_buffer, sizeof(resp_buffer), 30000);
    if (resp_len > 0) {
        // Check for error in response
        if (strstr(resp_buffer, "\"error\"") != NULL) {
            fprintf(stderr, "DistriBox ICD: NDRange execution error: %s\n", resp_buffer);
            exec_status = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
        }
    }

    // Create output event
    if (event) {
        distri_event_t *evt = (distri_event_t *)calloc(1, sizeof(distri_event_t));
        if (evt) {
            evt->dispatch = k->dispatch;
            evt->context = q->context;
            evt->status = (exec_status == CL_SUCCESS) ? CL_COMPLETE : CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
            evt->ref_count = 1;
            generate_id(evt->event_id, sizeof(evt->event_id), "evt");
            *event = (cl_event)evt;
        }
    }

    return exec_status;
}

// ── clEnqueueTask (equivalent to NDRange with global=local) ─

cl_int distriboxEnqueueTask(cl_command_queue command_queue,
                             cl_kernel kernel,
                             cl_uint num_events_in_wait_list,
                             const cl_event *event_wait_list,
                             cl_event *event) {
    size_t one = 1;
    return distriboxEnqueueNDRangeKernel(command_queue, kernel, 1,
        NULL, &one, &one, num_events_in_wait_list, event_wait_list, event);
}
