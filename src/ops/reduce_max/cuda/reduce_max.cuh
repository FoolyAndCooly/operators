#ifndef __CUDA_REDUCE_MAX_H__
#define __CUDA_REDUCE_MAX_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct ReduceMaxCudaDescriptor {
    Device device;
    int device_id;
    DT dtype;
    uint64_t *shape;
    uint64_t ndim;
};

typedef struct ReduceMaxCudaDescriptor *ReduceMaxCudaDescriptor_t;

infiniopStatus_t cudaCreateReduceMaxDescriptor(CudaHandle_t,
                                         ReduceMaxCudaDescriptor_t *,
                                         infiniopTensorDescriptor_t reduced,
                                         infiniopTensorDescriptor_t data,
                                         infiniopTensorDescriptor_t axes,
					 int keepdims,
					 int noop_with_empty_axes
					 );

infiniopStatus_t cudaReduceMax(ReduceMaxCudaDescriptor_t desc,
                         void *reduced, void const *data, void const *axes,
                         void *stream);

infiniopStatus_t cudaDestroyReduceMaxDescriptor(ReduceMaxCudaDescriptor_t desc);

#endif
