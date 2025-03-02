#ifndef REDUCE_MIN_H
#define REDUCE_MIN_H

#include "../reduce/reduce.h"
#include "../../export.h"
#include "../../operators.h"

__C __export infiniopStatus_t infiniopCreateReduceMinDescriptor(
                                          infiniopHandle_t handle,
                                          infiniopReduceDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t reduced,
                                          infiniopTensorDescriptor_t data,
                                          infiniopTensorDescriptor_t axes,
                                          int keepdims,
                                          int noop_with_empty_axes);

__C __export infiniopStatus_t infiniopReduceMin(infiniopReduceDescriptor_t desc,
                                          void *reduced,
                                          void const *data,
                                          void const *axes,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMinDescriptor(infiniopReduceDescriptor_t desc);

#endif
