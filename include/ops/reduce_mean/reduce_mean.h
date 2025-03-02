#ifndef REDUCE_MEAN_H
#define REDUCE_MEAN_H

#include "../reduce/reduce.h"
#include "../../export.h"
#include "../../operators.h"

__C __export infiniopStatus_t infiniopCreateReduceMeanDescriptor(
                                          infiniopHandle_t handle,
                                          infiniopReduceDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t reduced,
                                          infiniopTensorDescriptor_t data,
                                          infiniopTensorDescriptor_t axes,
                                          int keepdims,
                                          int noop_with_empty_axes);

__C __export infiniopStatus_t infiniopReduceMean(infiniopReduceDescriptor_t desc,
                                          void *reduced,
                                          void const *data,
                                          void const *axes,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMeanDescriptor(infiniopReduceDescriptor_t desc);

#endif
