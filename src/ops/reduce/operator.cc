#include "../utils.h"
#include "operators.h"
#include "ops/reduce/reduce.h"

#ifdef ENABLE_CPU
#include "cpu/reduce_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/reduce.cuh"
#endif

__C infiniopStatus_t infiniopCreateReduceDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t reduced,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t axes,
    int op,
    int keepdims,
    int noop_with_empty_axes) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateReduceDescriptor(handle, (ReduceCpuDescriptor_t *) desc_ptr, reduced, data, axes, op, keepdims, noop_with_empty_axes);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateReduceDescriptor((CudaHandle_t) handle, (ReduceCudaDescriptor_t *) desc_ptr, reduced, data, axes, op, keepdims, noop_with_empty_axes);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc, void *reduced, void const *data, void const *axes, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuReduce((ReduceCpuDescriptor_t) desc, reduced, data, axes, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaReduce((ReduceCudaDescriptor_t) desc, reduced, data, axes, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyReduceDescriptor((ReduceCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyReduceDescriptor((ReduceCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
