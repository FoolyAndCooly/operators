#ifndef REDUCE_H
#define REDUCE_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceDescriptor {
    Device device;
} ReduceDescriptor;

typedef ReduceDescriptor *infiniopReduceDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceDescriptor(
                                          infiniopHandle_t handle,
                                          infiniopReduceDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t reduced,
                                          infiniopTensorDescriptor_t data,
                                          infiniopTensorDescriptor_t axes,
					  int op,
                                          int keepdims,
                                          int noop_with_empty_axes);

__C __export infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc,
                                          void *reduced,
                                          void const *data,
                                          void const *axes,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc);

#endif
