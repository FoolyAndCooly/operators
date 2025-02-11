#include "../utils.h"
#include "operators.h"
#include "ops/where/where.h"

#ifdef ENABLE_CPU
#include "cpu/where_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/where.cuh"
#endif

__C infiniopStatus_t infiniopCreateWhereDescriptor(
    infiniopHandle_t handle,
    infiniopWhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t condition,
    infiniopTensorDescriptor_t c,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateWhereDescriptor(handle, (WhereCpuDescriptor_t *) desc_ptr, condition, c, a, b);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateWhereDescriptor((CudaHandle_t) handle, (WhereCudaDescriptor_t *) desc_ptr, condition, c, a, b);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc, void const *condition, void *c, void const *a, void const *b, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuWhere((WhereCpuDescriptor_t) desc, condition, c, a, b, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaWhere((WhereCudaDescriptor_t) desc, condition, c, a, b, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyWhereDescriptor((WhereCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyWhereDescriptor((WhereCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
