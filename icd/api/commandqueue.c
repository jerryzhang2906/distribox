/**
 * api/commandqueue.c — Command queue creation and management
 */
#include "../icd.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

cl_command_queue distriboxCreateCommandQueue(cl_context context,
                                              cl_device_id device,
                                              cl_command_queue_properties properties,
                                              cl_int *errcode_ret) {
    return distriboxCreateCommandQueueWithProperties(context, device,
        (const cl_queue_properties *)&properties, errcode_ret);
}

cl_command_queue distriboxCreateCommandQueueWithProperties(cl_context context,
                                                            cl_device_id device,
                                                            const cl_queue_properties *properties,
                                                            cl_int *errcode_ret) {
    if (context == NULL || device == NULL) {
        if (errcode_ret) *errcode_ret = CL_INVALID_VALUE;
        return NULL;
    }

    distri_command_queue_t *q = (distri_command_queue_t *)calloc(1, sizeof(distri_command_queue_t));
    if (q == NULL) {
        if (errcode_ret) *errcode_ret = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }

    q->dispatch = ((distri_context_t *)context)->dispatch;
    q->context = (distri_context_t *)context;
    q->ref_count = 1;

    if (properties) {
        memcpy(&q->properties, properties, sizeof(cl_command_queue_properties));
    }

    generate_id(q->queue_id, sizeof(q->queue_id), "q");

    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_command_queue)q;
}

cl_int distriboxRetainCommandQueue(cl_command_queue command_queue) {
    if (command_queue == NULL) return CL_INVALID_COMMAND_QUEUE;
    ((distri_command_queue_t *)command_queue)->ref_count++;
    return CL_SUCCESS;
}

cl_int distriboxReleaseCommandQueue(cl_command_queue command_queue) {
    if (command_queue == NULL) return CL_INVALID_COMMAND_QUEUE;
    distri_command_queue_t *q = (distri_command_queue_t *)command_queue;
    q->ref_count--;
    if (q->ref_count == 0) {
        free(q);
    }
    return CL_SUCCESS;
}

cl_int distriboxGetCommandQueueInfo(cl_command_queue command_queue,
                                     cl_command_queue_info param_name,
                                     size_t param_value_size,
                                     void *param_value,
                                     size_t *param_value_size_ret) {
    if (command_queue == NULL) return CL_INVALID_COMMAND_QUEUE;
    distri_command_queue_t *q = (distri_command_queue_t *)command_queue;

    switch (param_name) {
    case CL_QUEUE_CONTEXT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_context);
        if (param_value && param_value_size >= sizeof(cl_context))
            memcpy(param_value, &q->context, sizeof(cl_context));
        return CL_SUCCESS;
    case CL_QUEUE_DEVICE:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_device_id);
        if (param_value && param_value_size >= sizeof(cl_device_id))
            memcpy(param_value, &q->context->device, sizeof(cl_device_id));
        return CL_SUCCESS;
    case CL_QUEUE_REFERENCE_COUNT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
        if (param_value && param_value_size >= sizeof(cl_uint))
            memcpy(param_value, &q->ref_count, sizeof(cl_uint));
        return CL_SUCCESS;
    case CL_QUEUE_PROPERTIES:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_command_queue_properties);
        if (param_value && param_value_size >= sizeof(cl_command_queue_properties))
            memcpy(param_value, &q->properties, sizeof(cl_command_queue_properties));
        return CL_SUCCESS;
    default:
        return CL_INVALID_VALUE;
    }
}

// ── Finish / Flush ──────────────────────────────────

cl_int distriboxFinish(cl_command_queue command_queue) {
    if (command_queue == NULL) return CL_INVALID_COMMAND_QUEUE;
    // IPC: wait for all pending commands on this queue to complete
    if (g_ipc_connected) {
        distri_command_queue_t *q = (distri_command_queue_t *)command_queue;
        char msg[256];
        char msg_id[64];
        generate_id(msg_id, sizeof(msg_id), "fin");
        int msg_len = snprintf(msg, sizeof(msg),
            "{\"type\":\"queue_finish\",\"msg_id\":\"%s\",\"queue_id\":\"%s\"}\n",
            msg_id, q->queue_id);
        ipc_send_command(msg, (uint64_t)msg_len);
        char resp[512];
        ipc_recv_response(resp, sizeof(resp), 10000);
    }
    return CL_SUCCESS;
}

cl_int distriboxFlush(cl_command_queue command_queue) {
    if (command_queue == NULL) return CL_INVALID_COMMAND_QUEUE;
    // IPC: submit all pending commands to VGPU Core
    // For MVP, commands are submitted immediately in EnqueueNDRange
    if (g_ipc_connected) {
        distri_command_queue_t *q = (distri_command_queue_t *)command_queue;
        char msg[256];
        char msg_id[64];
        generate_id(msg_id, sizeof(msg_id), "fls");
        int msg_len = snprintf(msg, sizeof(msg),
            "{\"type\":\"queue_finish\",\"msg_id\":\"%s\",\"queue_id\":\"%s\"}\n",
            msg_id, q->queue_id);
        ipc_send_command(msg, (uint64_t)msg_len);
    }
    return CL_SUCCESS;
}
