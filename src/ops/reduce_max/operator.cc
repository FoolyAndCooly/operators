#include "../utils.h"
#include "operators.h"
#include "ops/reduce_max/reduce_max.h"

__C infiniopStatus_t infiniopCreateReduceMaxDescriptor(
    infiniopHandle_t handle,
    infiniopReduceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t reduced,
    infiniopTensorDescriptor_t data,
    infiniopTensorDescriptor_t axes,
    int keepdims,
    int noop_with_empty_axes) {
    
    return infiniopCreateReduceDescriptor(handle, desc_ptr, reduced, data, axes, 0, keepdims, noop_with_empty_axes);
}

__C infiniopStatus_t infiniopReduceMax(infiniopReduceDescriptor_t desc, void *reduced, void const *data, void const *axes, void *stream) {
    return infiniopReduce(desc, reduced, data, axes, stream);
}

__C infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceDescriptor_t desc) {
    return infiniopDestroyReduceDescriptor(desc);
}
