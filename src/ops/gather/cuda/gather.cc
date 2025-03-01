#include "gather.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateGatherDescriptor(CudaHandle_t handle,
                                         GatherCudaDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t output,
                                         infiniopTensorDescriptor_t input,
                                         infiniopTensorDescriptor_t indices,
					 int axis) {
    uint64_t ndim = output->ndim;
    if (!is_contiguous(input) || !is_contiguous(indices) || !is_contiguous(output)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (output->dt != F16 && output->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (indices->dt != I32 && indices->dt != I64) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (output->dt != input->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t stride = std::accumulate(input->shape + axis + 1, input->shape + input->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t othersize = std::accumulate(input->shape, input->shape + input->ndim, 1ULL, std::multiplies<uint64_t>()) / input->shape[axis];
    uint64_t indsize = std::accumulate(indices->shape, indices->shape + indices->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t dimsize = input->shape[axis];    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, handle->device_id);

    *desc_ptr = new GatherCudaDescriptor{
        DevNvGpu,
        output->dt,
        indices->dt,
        handle->device_id,
        ndim,
	stride,
	othersize,
	indsize,
	dimsize,
        static_cast<uint64_t>(prop.maxGridSize[0]),
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyGatherDescriptor(GatherCudaDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}
