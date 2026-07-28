/**
 * api/event.c — Event management: clWaitForEvents, clGetEventProfilingInfo, etc.
 */
#include "../icd.h"
#include <string.h>
#include <stdlib.h>

cl_int distriboxWaitForEvents(cl_uint num_events,
                               const cl_event *event_list) {
    if (num_events == 0 || event_list == NULL) {
        return CL_INVALID_VALUE;
    }

    // For MVP, all events are immediately complete
    // In a real implementation, this would block until workers return results
    for (cl_uint i = 0; i < num_events; i++) {
        if (event_list[i] == NULL) return CL_INVALID_EVENT;
        // TODO: if event not complete, wait on VGPU Core
    }

    return CL_SUCCESS;
}

cl_int distriboxGetEventInfo(cl_event event,
                              cl_event_info param_name,
                              size_t param_value_size,
                              void *param_value,
                              size_t *param_value_size_ret) {
    if (event == NULL) return CL_INVALID_EVENT;
    distri_event_t *e = (distri_event_t *)event;

    switch (param_name) {
    case CL_EVENT_COMMAND_QUEUE: {
        // Return NULL for user events, queue for command events
        // Simplified: always return NULL for MVP
        cl_command_queue q = NULL;
        if (param_value_size_ret) *param_value_size_ret = sizeof(q);
        if (param_value && param_value_size >= sizeof(q))
            memcpy(param_value, &q, sizeof(q));
        return CL_SUCCESS;
    }
    case CL_EVENT_COMMAND_TYPE: {
        cl_command_type t = CL_COMMAND_NDRANGE_KERNEL;
        if (param_value_size_ret) *param_value_size_ret = sizeof(t);
        if (param_value && param_value_size >= sizeof(t))
            memcpy(param_value, &t, sizeof(t));
        return CL_SUCCESS;
    }
    case CL_EVENT_COMMAND_EXECUTION_STATUS:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_int);
        if (param_value && param_value_size >= sizeof(cl_int))
            memcpy(param_value, &e->status, sizeof(cl_int));
        return CL_SUCCESS;
    case CL_EVENT_REFERENCE_COUNT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
        if (param_value && param_value_size >= sizeof(cl_uint))
            memcpy(param_value, &e->ref_count, sizeof(cl_uint));
        return CL_SUCCESS;
    case CL_EVENT_CONTEXT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_context);
        if (param_value && param_value_size >= sizeof(cl_context))
            memcpy(param_value, &e->context, sizeof(cl_context));
        return CL_SUCCESS;
    default:
        return CL_INVALID_VALUE;
    }
}

cl_int distriboxRetainEvent(cl_event event) {
    if (event == NULL) return CL_INVALID_EVENT;
    ((distri_event_t *)event)->ref_count++;
    return CL_SUCCESS;
}

cl_int distriboxReleaseEvent(cl_event event) {
    if (event == NULL) return CL_INVALID_EVENT;
    distri_event_t *e = (distri_event_t *)event;
    e->ref_count--;
    if (e->ref_count == 0) {
        free(e);
    }
    return CL_SUCCESS;
}

cl_int distriboxGetEventProfilingInfo(cl_event event,
                                       cl_profiling_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
    if (event == NULL) return CL_INVALID_EVENT;
    distri_event_t *e = (distri_event_t *)event;

    switch (param_name) {
    case CL_PROFILING_COMMAND_QUEUED:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_ulong);
        if (param_value && param_value_size >= sizeof(cl_ulong))
            memcpy(param_value, &e->queued_time, sizeof(cl_ulong));
        return CL_SUCCESS;
    case CL_PROFILING_COMMAND_SUBMIT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_ulong);
        if (param_value && param_value_size >= sizeof(cl_ulong))
            memcpy(param_value, &e->submit_time, sizeof(cl_ulong));
        return CL_SUCCESS;
    case CL_PROFILING_COMMAND_START:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_ulong);
        if (param_value && param_value_size >= sizeof(cl_ulong))
            memcpy(param_value, &e->start_time, sizeof(cl_ulong));
        return CL_SUCCESS;
    case CL_PROFILING_COMMAND_END:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_ulong);
        if (param_value && param_value_size >= sizeof(cl_ulong))
            memcpy(param_value, &e->end_time, sizeof(cl_ulong));
        return CL_SUCCESS;
    default:
        return CL_INVALID_VALUE;
    }
}

cl_event distriboxCreateUserEvent(cl_context context,
                                   cl_int *errcode_ret) {
    if (context == NULL) {
        if (errcode_ret) *errcode_ret = CL_INVALID_CONTEXT;
        return NULL;
    }

    distri_event_t *evt = (distri_event_t *)calloc(1, sizeof(distri_event_t));
    if (evt == NULL) {
        if (errcode_ret) *errcode_ret = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }

    evt->dispatch = ((distri_context_t *)context)->dispatch;
    evt->context = (distri_context_t *)context;
    evt->status = CL_SUBMITTED;
    evt->ref_count = 1;
    generate_id(evt->event_id, sizeof(evt->event_id), "uevt");

    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_event)evt;
}

cl_int distriboxSetUserEventStatus(cl_event event,
                                    cl_int execution_status) {
    if (event == NULL) return CL_INVALID_EVENT;
    distri_event_t *e = (distri_event_t *)event;

    if (execution_status == CL_COMPLETE) {
        e->status = CL_COMPLETE;
    } else {
        return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}
