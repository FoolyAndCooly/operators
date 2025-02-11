#ifndef __CUDA_where_H__
#define __CUDA_where_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct WhereCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t c_data_size;
    uint64_t max_grid_size;
    int64_t const *condition_strides;
    int64_t const *a_strides;
    int64_t const *b_strides;
    int64_t const *c_strides;
    bool broadcasted;
};

typedef struct WhereCudaDescriptor *WhereCudaDescriptor_t;

infiniopStatus_t cudaCreateWhereDescriptor(CudaHandle_t,
                                         WhereCudaDescriptor_t *,
					 infiniopTensorDescriptor_t condition,
                                         infiniopTensorDescriptor_t c,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b);

infiniopStatus_t cudaWhere(WhereCudaDescriptor_t desc,
			 void const *condition,
                         void *c, void const *a, void const *b,
                         void *stream);

infiniopStatus_t cudaDestroyWhereDescriptor(WhereCudaDescriptor_t desc);

#endif
