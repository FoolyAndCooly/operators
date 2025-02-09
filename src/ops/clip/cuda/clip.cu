#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "clip.cuh"

template<typename T>
__global__ void clip(
    T *y,
    const T *x,
    T *min,
    T *max,
    uint64_t data_size,
    uint64_t offset) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        y[idx] = (x[idx] < *min) ? *min : ((x[idx] > *max) ? *max : x[idx]);
    }
}

template<typename T>
void clip_nv_gpu(ClipCudaDescriptor_t desc, T *y, T const *x, T *min, T *max, uint64_t data_size, uint64_t offset, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < data_size; i += step) {
        clip<T><<<gridDims, blockDims, 0, cuda_stream>>>(y, x, min, max, offset + data_size, offset + i);
    }
}

template<typename T>
infiniopStatus_t clip_nv_gpu(ClipCudaDescriptor_t desc, void *y, void const *x, void *min, void *max, void *stream) {
    const auto data_size = desc->data_size;
    const auto x_vec = reinterpret_cast<const T *>(x);
    const auto y_vec = reinterpret_cast<T *>(y);
    const auto min_ = reinterpret_cast<T *>(min);
    const auto max_ = reinterpret_cast<T *>(max);
    clip_nv_gpu(desc, y_vec, x_vec, min_, max_, data_size, 0, stream);
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
                          void *y, void const *x,
			  void *min, void *max,
                          void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return clip_nv_gpu<half>(desc, y, x, min, max, stream);
    }
    if (desc->dtype == F32) {
        return clip_nv_gpu<float>(desc, y, x, min, max, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
