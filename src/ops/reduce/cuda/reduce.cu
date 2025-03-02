#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "reduce.cuh"
#include <float.h>
#define MAX(a, b) (a > b) ? a : b
#define MIN(a, b) (a < b) ? a : b
#define BLOCKX 64
#define NUM_PER_BLOCK 128

template <typename T>
struct NumericLimits {
    __host__ __device__ static T min() { return T(); }
    __host__ __device__ static T max() { return T(); }
};

template <>
struct NumericLimits<float> {
    __host__ __device__ static float min() { return -FLT_MAX; }
    __host__ __device__ static float max() { return FLT_MAX; }
};

template <>
struct NumericLimits<__half> {
    __host__ __device__ static __half min() {
        return  __float2half(-65504.0f); 
    }
    __host__ __device__ static __half max() {
        return  __float2half(65504.0f); 
    }
};

template<typename T>
__global__ void reduce_max(
    T *reduced,
    T const *data,
    uint64_t dimsize,
    uint64_t stride,
    uint64_t othersize
    ) {
    __shared__ T sdata[1024]; // blockDim.z * blockDim.y * blockDim.x
    uint64_t i = blockIdx.x * NUM_PER_BLOCK + threadIdx.x;
    uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t k = blockIdx.z * blockDim.z + threadIdx.z;
    uint64_t base = j * dimsize * stride + k;
    T a = ((i < dimsize) && (j < othersize) && (k < stride)) ? data[base + i * stride] : NumericLimits<T>::min();
    T b = ((i + blockDim.x < dimsize) && (j < othersize) && (k < stride)) ? data[base + (i + blockDim.x) * stride] : NumericLimits<T>::min();
    uint64_t sbase = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    sdata[sbase] = MAX(a, b);
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
	if (threadIdx.x < s) {
            sdata[sbase] = MAX(sdata[sbase], sdata[sbase + s]);
	}
	__syncthreads();
    }

    dimsize = (dimsize + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
    if (threadIdx.x == 0 && (j < othersize) && (k < stride)) reduced[j * dimsize * stride + blockIdx.x * stride + k] = sdata[sbase];
}

template<typename T>
__global__ void reduce_min(
    T *reduced,
    T const *data,
    uint64_t dimsize,
    uint64_t stride,
    uint64_t othersize
    ) {
    __shared__ T sdata[1024]; // blockDim.z * blockDim.y * blockDim.x
    uint64_t i = blockIdx.x * NUM_PER_BLOCK + threadIdx.x;
    uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t k = blockIdx.z * blockDim.z + threadIdx.z;
    uint64_t base = j * dimsize * stride + k;
    T a = ((i < dimsize) && (j < othersize) && (k < stride)) ? data[base + i * stride] : NumericLimits<T>::max();
    T b = ((i + blockDim.x < dimsize) && (j < othersize) && (k < stride)) ? data[base + (i + blockDim.x) * stride] : NumericLimits<T>::max();
    uint64_t sbase = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    sdata[sbase] = MIN(a, b);
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
	if (threadIdx.x < s) {
            sdata[sbase] = MIN(sdata[sbase], sdata[sbase + s]);
	}
	__syncthreads();
    }

    dimsize = (dimsize + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
    if (threadIdx.x == 0 && (j < othersize) && (k < stride)) reduced[j * dimsize * stride + blockIdx.x * stride + k] = sdata[sbase];
}

template<typename T>
__global__ void reduce_mean(
    T *reduced,
    T const *data,
    uint64_t dimsize,
    uint64_t stride,
    uint64_t othersize,
    T presize
    ) {
    __shared__ T sdata[1024]; // blockDim.z * blockDim.y * blockDim.x
    uint64_t i = blockIdx.x * NUM_PER_BLOCK + threadIdx.x;
    uint64_t j = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t k = blockIdx.z * blockDim.z + threadIdx.z;
    uint64_t base = j * dimsize * stride + k;
    T a = ((i < dimsize) && (j < othersize) && (k < stride)) ? data[base + i * stride] : (T)0;
    T b = ((i + blockDim.x < dimsize) && (j < othersize) && (k < stride)) ? data[base + (i + blockDim.x) * stride] : (T)0;
    uint64_t sbase = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    sdata[sbase] = a + b;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
	if (threadIdx.x < s) {
            sdata[sbase] = sdata[sbase] + sdata[sbase + s];
	}
	__syncthreads();
    }
    dimsize = (dimsize + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
    if (threadIdx.x == 0 && (j < othersize) && (k < stride)) reduced[j * dimsize * stride + blockIdx.x * stride + k] = (dimsize == 1) ? sdata[sbase] / (T)((int)presize) : sdata[sbase];
}

template<typename T>
infiniopStatus_t reduce_nv_gpu(ReduceCudaDescriptor_t desc, void *reduced, void const *data, void const *axes, void* stream){
    auto axes_ = reinterpret_cast<int64_t const*>(axes);
    uint64_t stride = std::accumulate(desc->shape + *axes_ + 1, desc->shape + desc->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t othersize = std::accumulate(desc->shape, desc->shape + *axes_, 1ULL, std::multiplies<uint64_t>());
    uint64_t dimsize = desc->shape[*axes_];
    int BLOCK_DIM_x = BLOCKX;
    int BLOCK_DIM_y = 4;
    int BLOCK_DIM_z = 4;
    int num_block_x = (dimsize + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
    int num_block_y = (othersize + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    int num_block_z = (stride + BLOCK_DIM_z - 1) / BLOCK_DIM_z;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, BLOCK_DIM_z);
    dim3 grid_dim(num_block_x, num_block_y, num_block_z);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    T presize = (T)((int)dimsize);
    while (dimsize >= 1) {
        if (desc->op == 0) {
            reduce_max<T>
                <<<grid_dim, block_dim, 0, cuda_stream>>>((T*)reduced, (T const*)data, dimsize, stride, othersize);
	}
	else if (desc->op == 1) {
            reduce_min<T>
                <<<grid_dim, block_dim, 0, cuda_stream>>>((T*)reduced, (T const*)data, dimsize, stride, othersize);
	}
	else if (desc->op == 2) {
            reduce_mean<T>
                <<<grid_dim, block_dim, 0, cuda_stream>>>((T*)reduced, (T const*)data, dimsize, stride, othersize, presize);
	}
	uint64_t next_dimsize = (dimsize + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
	dimsize = (next_dimsize == 1) ? 0 : next_dimsize;
        grid_dim.x = (dimsize + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
	data = (void const*) reduced;
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaReduce(ReduceCudaDescriptor_t desc,
                         void *reduced, void const *data, void const *axes,
                         void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) return reduce_nv_gpu<half>(desc, reduced, data, axes, stream);
    if (desc->dtype == F32) return reduce_nv_gpu<float>(desc, reduced, data, axes, stream);
}
