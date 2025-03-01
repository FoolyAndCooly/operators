#ifndef reduce_max_H
#define reduce_max_H

#include "../../export.h"
#include "../../operators.h"

typedef struct ReduceMaxDescriptor {
    Device device;
} ReduceMaxDescriptor;

typedef ReduceMaxDescriptor *infiniopReduceMaxDescriptor_t;

__C __export infiniopStatus_t infiniopCreateReduceMaxDescriptor(
                                          infiniopHandle_t handle,
                                          infiniopReduceMaxDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t reduced,
                                          infiniopTensorDescriptor_t data,
                                          infiniopTensorDescriptor_t axes,
                                          int keepdims,
                                          int noop_with_empty_axes);

__C __export infiniopStatus_t infiniopReduceMax(infiniopReduceMaxDescriptor_t desc,
                                          void *reduced,
                                          void const *data,
                                          void const *axes,
                                          void *stream);

__C __export infiniopStatus_t infiniopDestroyReduceMaxDescriptor(infiniopReduceMaxDescriptor_t desc);

#endif
