#include "../utils.h"
#include "operators.h"
#include "ops/reduce_max/reduce_max.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_max_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/reduce_max.cuh"
#endif

__C infiniopStatus_t infiniopCreateReduceMaxDescriptor(
    infiniopHandle_t handle,
    infiniopReduceMaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t reduced,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t axes,
    int keepdims,
    int noop_with_empty_axes) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateReduceMaxDescriptor(handle, (ReduceMaxCpuDescriptor_t *) desc_ptr, reduced, data, axes, keepdims, noop_with_empty_axes);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateReduceMaxDescriptor((CudaHandle_t) handle, (ReduceMaxCudaDescriptor_t *) desc_ptr, reduced, data, axes, keepdims, noop_with_empty_axes);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc, void *reduced, void const *data, void const *axes, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuReduceMax((ReduceMaxCpuDescriptor_t) desc, reduced, data, axes, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaReduceMax((ReduceMaxCudaDescriptor_t) desc, reduced, data, axes, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyReduceMaxDescriptor((ReduceMaxCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyReduceMaxDescriptor((ReduceMaxCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
