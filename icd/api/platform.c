/**
 * api/platform.c — clGetPlatformIDs, clGetPlatformInfo
 */

#include "../icd.h"
#include <string.h>

cl_int distriboxGetPlatformIDs(cl_uint num_entries,
                                cl_platform_id *platforms,
                                cl_uint *num_platforms) {
    if (num_platforms) {
        *num_platforms = 1;
    }
    if (platforms && num_entries >= 1) {
        platforms[0] = (cl_platform_id)g_platform;
    }
    return CL_SUCCESS;
}

cl_int distriboxGetPlatformInfo(cl_platform_id platform,
                                 cl_platform_info param_name,
                                 size_t param_value_size,
                                 void *param_value,
                                 size_t *param_value_size_ret) {
    if (platform == NULL) {
        return CL_INVALID_PLATFORM;
    }

    distri_platform_t *p = (distri_platform_t *)platform;
    const char *str = NULL;
    size_t str_len = 0;

    switch (param_name) {
    case CL_PLATFORM_PROFILE:
        str = "FULL_PROFILE";
        break;
    case CL_PLATFORM_VERSION:
        str = "OpenCL 2.0 DistriBox";
        break;
    case CL_PLATFORM_NAME:
        str = p->name;
        break;
    case CL_PLATFORM_VENDOR:
        str = p->vendor;
        break;
    case CL_PLATFORM_EXTENSIONS:
        str = "cl_khr_icd cl_khr_fp64";
        break;
    case CL_PLATFORM_ICD_SUFFIX_KHR:
        str = p->icd_suffix;
        break;
    default:
        return CL_INVALID_VALUE;
    }

    str_len = strlen(str) + 1;

    if (param_value_size_ret) {
        *param_value_size_ret = str_len;
    }
    if (param_value && param_value_size >= str_len) {
        memcpy(param_value, str, str_len);
    } else if (param_value && param_value_size < str_len) {
        return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}
