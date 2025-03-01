#ifndef __CUDA_gather_H__
#define __CUDA_gather_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct GatherCudaDescriptor {
    Device device;
    DT dtype;
    DT dtin;
    int device_id;
    uint64_t ndim;
    uint64_t stride;
    uint64_t othersize;
    uint64_t indsize;
    uint64_t dimsize;
    uint64_t max_grid_size;
};

typedef struct GatherCudaDescriptor *GatherCudaDescriptor_t;

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t,
                                         GatherCudaDescriptor_t *,
                                         infiniopTensorDescriptor_t output,
                                         infiniopTensorDescriptor_t input,
                                         infiniopTensorDescriptor_t indices,
					 int axis);

infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc,
                         void *output, void const *input, void const *indices,
                         void *stream);

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc);

#endif
