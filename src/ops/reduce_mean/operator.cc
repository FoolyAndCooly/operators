#include "../utils.h"
#include "operators.h"
#include "ops/reduce_mean/reduce_mean.h"

__C infiniopStatus_t infiniopCreateReduceMeanDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t reduced,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t axes,
    int keepdims,
    int noop_with_empty_axes) {
    
    return infiniopCreateReduceDescriptor(handle, desc_ptr, reduced, data, axes, 2, keepdims, noop_with_empty_axes);
}

__C infiniopStatus_t infiniopReduceMean(infiniopReduceDescriptor_t desc, void *reduced, void const *data, void const *axes, void *stream) {
    return infiniopReduce(desc, reduced, data, axes, stream);
}

__C infiniopStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceDescriptor_t desc) {
    return infiniopDestroyReduceDescriptor(desc);
}
