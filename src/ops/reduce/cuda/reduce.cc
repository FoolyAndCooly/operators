#include "reduce.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateReduceDescriptor(CudaHandle_t handle,
                                         ReduceCudaDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t reduced,
                                         infiniopTensorDescriptor_t data,
                                         infiniopTensorDescriptor_t axes,
					 int op,
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

    *desc_ptr = new ReduceCudaDescriptor{
        DevNvGpu,
        handle->device_id,
	op,
        data->dt,
	data->shape,
	data->ndim
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyReduceDescriptor(ReduceCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}


