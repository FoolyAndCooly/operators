#ifndef __CPU_REDUCE_H__
#define __CPU_REDUCE_H__

#include "operators.h"
#include <numeric>

struct ReduceCpuDescriptor {
    Device device;
    int op;
    DT dtype;
    uint64_t *shape;
    int ndim;
};

typedef struct ReduceCpuDescriptor *ReduceCpuDescriptor_t;

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t,
                                         ReduceCpuDescriptor_t *,
                                         infiniopTensorDescriptor_t reduced,
                                         infiniopTensorDescriptor_t data,
                                         infiniopTensorDescriptor_t axes,
					 int op,
					 int keepdims,
                                         int noop_with_empty_axes
					 );

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                         void *reduced, void const *data, void const *axes,
                         void *stream);

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc);

#endif
