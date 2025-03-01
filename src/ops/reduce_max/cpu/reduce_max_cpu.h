#ifndef __CPU_REDUCE_MAX_H__
#define __CPU_REDUCE_MAX_H__

#include "operators.h"
#include <numeric>

struct ReduceMaxCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t *shape;
    int ndim;
};

typedef struct ReduceMaxCpuDescriptor *ReduceMaxCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceMaxDescriptor(infiniopHandle_t,
                                         ReduceMaxCpuDescriptor_t *,
                                         infiniopTensorDescriptor_t reduced,
                                         infiniopTensorDescriptor_t data,
                                         infiniopTensorDescriptor_t axes,
					 int keepdims,
                                         int noop_with_empty_axes
					 );

infiniopStatus_t cpuReduceMax(ReduceMaxCpuDescriptor_t desc,
                         void *reduced, void const *data, void const *axes,
                         void *stream);

infiniopStatus_t cpuDestroyReduceMaxDescriptor(ReduceMaxCpuDescriptor_t desc);

#endif
