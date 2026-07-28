/**
 * api/context.c — clCreateContext, clRetainContext, clReleaseContext
 */
#include "../icd.h"
#include <string.h>
#include <stdlib.h>

cl_context distriboxCreateContext(const cl_context_properties *properties,
                                   cl_uint num_devices,
                                   const cl_device_id *devices,
                                   void (*pfn_notify)(const char *, const void *, size_t, void *),
                                   void *user_data,
                                   cl_int *errcode_ret) {
    (void)properties;
    (void)pfn_notify;
    (void)user_data;

    if (num_devices == 0 || devices == NULL) {
        if (errcode_ret) *errcode_ret = CL_INVALID_VALUE;
        return NULL;
    }

    distri_context_t *ctx = (distri_context_t *)calloc(1, sizeof(distri_context_t));
    if (ctx == NULL) {
        if (errcode_ret) *errcode_ret = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }

    ctx->dispatch = ((distri_device_t *)devices[0])->dispatch;
    ctx->device = (distri_device_t *)devices[0];
    ctx->ref_count = 1;

    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_context)ctx;
}

cl_context distriboxCreateContextFromType(const cl_context_properties *properties,
                                           cl_device_type device_type,
                                           void (*pfn_notify)(const char *, const void *, size_t, void *),
                                           void *user_data,
                                           cl_int *errcode_ret) {
    // Delegate to GetDeviceIDs + CreateContext
    cl_device_id device;
    cl_uint num_devices;
    cl_int err = distriboxGetDeviceIDs(g_platform, device_type, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        if (errcode_ret) *errcode_ret = CL_DEVICE_NOT_FOUND;
        return NULL;
    }
    return distriboxCreateContext(properties, 1, &device, pfn_notify, user_data, errcode_ret);
}

cl_int distriboxRetainContext(cl_context context) {
    if (context == NULL) return CL_INVALID_CONTEXT;
    distri_context_t *ctx = (distri_context_t *)context;
    ctx->ref_count++;
    return CL_SUCCESS;
}

cl_int distriboxReleaseContext(cl_context context) {
    if (context == NULL) return CL_INVALID_CONTEXT;
    distri_context_t *ctx = (distri_context_t *)context;
    ctx->ref_count--;
    if (ctx->ref_count == 0) {
        free(ctx);
    }
    return CL_SUCCESS;
}

cl_int distriboxGetContextInfo(cl_context context,
                                cl_context_info param_name,
                                size_t param_value_size,
                                void *param_value,
                                size_t *param_value_size_ret) {
    if (context == NULL) return CL_INVALID_CONTEXT;
    distri_context_t *ctx = (distri_context_t *)context;

    switch (param_name) {
    case CL_CONTEXT_DEVICES: {
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_device_id);
        if (param_value && param_value_size >= sizeof(cl_device_id)) {
            memcpy(param_value, &ctx->device, sizeof(cl_device_id));
        }
        return CL_SUCCESS;
    }
    case CL_CONTEXT_NUM_DEVICES: {
        cl_uint n = 1;
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
        if (param_value && param_value_size >= sizeof(cl_uint)) {
            memcpy(param_value, &n, sizeof(cl_uint));
        }
        return CL_SUCCESS;
    }
    case CL_CONTEXT_REFERENCE_COUNT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
        if (param_value && param_value_size >= sizeof(cl_uint)) {
            memcpy(param_value, &ctx->ref_count, sizeof(cl_uint));
        }
        return CL_SUCCESS;
    default:
        return CL_INVALID_VALUE;
    }
}
