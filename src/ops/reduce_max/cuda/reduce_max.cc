#include "reduce_max.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateReduceMaxDescriptor(CudaHandle_t handle,
                                         ReduceMaxCudaDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t reduced,
                                         infiniopTensorDescriptor_t data,
                                         infiniopTensorDescriptor_t axes,
					 int keepdims,
					 int noop_with_empty_axes
					 ) {
    if (reduced->dt != F16 && reduced->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (axes->dt != I64) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (reduced->dt != data->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    *desc_ptr = new ReduceMaxCudaDescriptor{
        DevNvGpu,
        handle->device_id,
        data->dt,
	data->shape,
	data->ndim
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyReduceMaxDescriptor(ReduceMaxCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}


