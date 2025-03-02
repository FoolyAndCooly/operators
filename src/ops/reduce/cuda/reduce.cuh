#ifndef __CUDA_REDUCE_H__
#define __CUDA_REDUCE_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct ReduceCudaDescriptor {
    Device device;
    int device_id;
    int op;
    DT dtype;
    uint64_t *shape;
    uint64_t ndim;
};

typedef struct ReduceCudaDescriptor *ReduceCudaDescriptor_t;

infiniopStatus_t cudaCreateReduceDescriptor(CudaHandle_t,
                                         ReduceCudaDescriptor_t *,
                                         infiniopTensorDescriptor_t reduced,
                                         infiniopTensorDescriptor_t data,
                                         infiniopTensorDescriptor_t axes,
					 int op,
					 int keepdims,
					 int noop_with_empty_axes
					 );

infiniopStatus_t cudaReduce(ReduceCudaDescriptor_t desc,
                         void *reduced, void const *data, void const *axes,
                         void *stream);

infiniopStatus_t cudaDestroyReduceDescriptor(ReduceCudaDescriptor_t desc);

#endif
