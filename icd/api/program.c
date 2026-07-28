/**
 * api/program.c — clCreateProgramWithSource, clBuildProgram, etc.
 *
 * Programs are OpenCL C source code that gets compiled on workers.
 * We capture the source and forward it to VGPU Core for distribution.
 */
#include "../icd.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

cl_program distriboxCreateProgramWithSource(cl_context context,
                                             cl_uint count,
                                             const char **strings,
                                             const size_t *lengths,
                                             cl_int *errcode_ret) {
    if (context == NULL || count == 0 || strings == NULL) {
        if (errcode_ret) *errcode_ret = CL_INVALID_VALUE;
        return NULL;
    }

    // Calculate total source length
    size_t total_len = 0;
    for (cl_uint i = 0; i < count; i++) {
        if (lengths && lengths[i] > 0) {
            total_len += lengths[i];
        } else if (strings[i]) {
            total_len += strlen(strings[i]);
        }
    }

    distri_program_t *prog = (distri_program_t *)calloc(1, sizeof(distri_program_t));
    if (prog == NULL) {
        if (errcode_ret) *errcode_ret = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }

    prog->dispatch = ((distri_context_t *)context)->dispatch;
    prog->context = (distri_context_t *)context;
    prog->ref_count = 1;
    prog->compiled = false;
    prog->build_status = CL_BUILD_NONE;

    // Copy source
    prog->source = (char *)malloc(total_len + 1);
    if (prog->source == NULL) {
        free(prog);
        if (errcode_ret) *errcode_ret = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }

    prog->source_len = total_len;
    char *dst = prog->source;
    for (cl_uint i = 0; i < count; i++) {
        size_t len = (lengths && lengths[i] > 0) ? lengths[i] : strlen(strings[i]);
        memcpy(dst, strings[i], len);
        dst += len;
    }
    *dst = '\0';

    generate_id(prog->program_id, sizeof(prog->program_id), "prog");

    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_program)prog;
}

cl_program distriboxCreateProgramWithBinary(cl_context context,
                                             cl_uint num_devices,
                                             const cl_device_id *device_list,
                                             const size_t *lengths,
                                             const unsigned char **binaries,
                                             cl_int *binary_status,
                                             cl_int *errcode_ret) {
    if (context == NULL || num_devices == 0 || binaries == NULL) {
        if (errcode_ret) *errcode_ret = CL_INVALID_VALUE;
        return NULL;
    }

    distri_program_t *prog = (distri_program_t *)calloc(1, sizeof(distri_program_t));
    if (prog == NULL) {
        if (errcode_ret) *errcode_ret = CL_OUT_OF_HOST_MEMORY;
        return NULL;
    }

    prog->dispatch = ((distri_context_t *)context)->dispatch;
    prog->context = (distri_context_t *)context;
    prog->ref_count = 1;

    // Store binary (for cached compilation)
    // In MVP, we store the binary for later use by VGPU Core
    if (lengths && lengths[0] > 0 && binaries[0]) {
        prog->source_len = lengths[0];
        // For binary programs, source stores the binary data
        prog->source = (char *)malloc(prog->source_len);
        if (prog->source) {
            memcpy(prog->source, binaries[0], prog->source_len);
        }
        prog->compiled = true;
        prog->build_status = CL_BUILD_SUCCESS;
    }

    generate_id(prog->program_id, sizeof(prog->program_id), "prog");

    if (binary_status && num_devices > 0) {
        binary_status[0] = CL_SUCCESS;
    }

    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return (cl_program)prog;
}

cl_int distriboxRetainProgram(cl_program program) {
    if (program == NULL) return CL_INVALID_PROGRAM;
    ((distri_program_t *)program)->ref_count++;
    return CL_SUCCESS;
}

cl_int distriboxReleaseProgram(cl_program program) {
    if (program == NULL) return CL_INVALID_PROGRAM;
    distri_program_t *prog = (distri_program_t *)program;
    prog->ref_count--;
    if (prog->ref_count == 0) {
        if (prog->source) free(prog->source);
        if (prog->options) free(prog->options);
        free(prog);
    }
    return CL_SUCCESS;
}

// ── clBuildProgram — this triggers worker compilation ─

cl_int distriboxBuildProgram(cl_program program,
                              cl_uint num_devices,
                              const cl_device_id *device_list,
                              const char *options,
                              void (*pfn_notify)(cl_program, void *),
                              void *user_data) {
    if (program == NULL) return CL_INVALID_PROGRAM;

    distri_program_t *prog = (distri_program_t *)program;

    // Save compile options
    if (options) {
        prog->options = strdup(options);
    }

    // IPC: Send kernel source + options to VGPU Core
    // VGPU Core distributes to workers, each compiles locally
    if (g_ipc_connected || ipc_connect() == 0) {
        // Send program_build message with source
        // For large sources, we'd use shared memory; for MVP, we send inline
        char msg_id[64];
        generate_id(msg_id, sizeof(msg_id), "bld");
        char msg[8192];
        int msg_len = snprintf(msg, sizeof(msg),
            "{\"type\":\"program_build\",\"msg_id\":\"%s\","
            "\"program_id\":\"%s\",\"options\":\"%s\"}\n",
            msg_id, prog->program_id,
            options ? options : "");
        ipc_send_command(msg, (uint64_t)msg_len);
        char resp[1024];
        ipc_recv_response(resp, sizeof(resp), 5000);
    }

    // Mark as compiled
    // (actual compilation happens on workers when they first execute)
    prog->compiled = true;
    prog->build_status = CL_BUILD_SUCCESS;
    snprintf(prog->build_log, sizeof(prog->build_log),
             "Program compiled for DistriBox virtual GPU.\n"
             "Kernels will be compiled on workers at first execution.\n");

    // Callback
    if (pfn_notify) {
        pfn_notify(program, user_data);
    }

    return CL_SUCCESS;
}

// ── clGetProgramBuildInfo ────────────────────────────

cl_int distriboxGetProgramBuildInfo(cl_program program,
                                     cl_device_id device,
                                     cl_program_build_info param_name,
                                     size_t param_value_size,
                                     void *param_value,
                                     size_t *param_value_size_ret) {
    if (program == NULL) return CL_INVALID_PROGRAM;
    distri_program_t *prog = (distri_program_t *)program;

    switch (param_name) {
    case CL_PROGRAM_BUILD_STATUS: {
        cl_build_status s = prog->build_status;
        if (param_value_size_ret) *param_value_size_ret = sizeof(s);
        if (param_value && param_value_size >= sizeof(s))
            memcpy(param_value, &s, sizeof(s));
        return CL_SUCCESS;
    }
    case CL_PROGRAM_BUILD_LOG: {
        size_t len = strlen(prog->build_log) + 1;
        if (param_value_size_ret) *param_value_size_ret = len;
        if (param_value && param_value_size >= len)
            memcpy(param_value, prog->build_log, len);
        else if (param_value && param_value_size < len)
            return CL_INVALID_VALUE;
        return CL_SUCCESS;
    }
    case CL_PROGRAM_BUILD_OPTIONS: {
        const char *opts = prog->options ? prog->options : "";
        size_t len = strlen(opts) + 1;
        if (param_value_size_ret) *param_value_size_ret = len;
        if (param_value && param_value_size >= len)
            memcpy(param_value, opts, len);
        return CL_SUCCESS;
    }
    default:
        return CL_INVALID_VALUE;
    }
}

// ── clGetProgramInfo ────────────────────────────────

cl_int distriboxGetProgramInfo(cl_program program,
                                cl_program_info param_name,
                                size_t param_value_size,
                                void *param_value,
                                size_t *param_value_size_ret) {
    if (program == NULL) return CL_INVALID_PROGRAM;
    distri_program_t *prog = (distri_program_t *)program;

    switch (param_name) {
    case CL_PROGRAM_REFERENCE_COUNT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
        if (param_value && param_value_size >= sizeof(cl_uint))
            memcpy(param_value, &prog->ref_count, sizeof(cl_uint));
        return CL_SUCCESS;
    case CL_PROGRAM_CONTEXT:
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_context);
        if (param_value && param_value_size >= sizeof(cl_context))
            memcpy(param_value, &prog->context, sizeof(cl_context));
        return CL_SUCCESS;
    case CL_PROGRAM_NUM_DEVICES: {
        cl_uint n = 1;
        if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
        if (param_value && param_value_size >= sizeof(cl_uint))
            memcpy(param_value, &n, sizeof(cl_uint));
        return CL_SUCCESS;
    }
    case CL_PROGRAM_SOURCE: {
        if (prog->source == NULL) return CL_INVALID_VALUE;
        if (param_value_size_ret) *param_value_size_ret = prog->source_len + 1;
        if (param_value && param_value_size >= prog->source_len + 1)
            memcpy(param_value, prog->source, prog->source_len + 1);
        else if (param_value && param_value_size < prog->source_len + 1)
            return CL_INVALID_VALUE;
        return CL_SUCCESS;
    }
    default:
        return CL_INVALID_VALUE;
    }
}
