#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "gather.cuh"

template<typename T>
inline int getTM(int stride) {
    int ret = sizeof(float4) / sizeof(T);
    if (stride % 4) ret = 2;
    if (stride % 2) ret = 1;
    return ret;
}

inline int getdim(uint64_t num) {
  if (num > 31) return 32;
  else if (num > 15) return 16;
  else if (num > 7) return 8;
  else if (num > 3) return 4;
  else if (num > 1) return 2;
  return 1;
}

inline int getTN(int indSize) {
    return (indSize > 16) ? 16 : indSize;
}

template <typename T, typename Tind>
__global__ void warpGatherKernel_v3(T *output, T const *input, Tind const *indices, int stride, int indSize, int dimsize, int TM, int TN, int offsetx, int offsety)
{
    int otherIdx = (blockIdx.x * blockDim.x + threadIdx.x) * TM + offsetx;
    int index = (blockIdx.y * blockDim.y + threadIdx.y) * TN + offsety;
    
    __shared__ int s_indices[1024];
    #pragma unroll 
    for (int cycle = 0; cycle < TN/ blockDim.x; cycle++) {
        int indx = cycle * blockDim.x + threadIdx.x;
        s_indices[threadIdx.y * TN + indx] = indices[index + indx];
    }

    if (threadIdx.x < TN % blockDim.x) {
        s_indices[threadIdx.y * TN + threadIdx.x] = indices[index + threadIdx.x];
    }
    __syncthreads();

    int s_ = otherIdx % stride;
    int i_ = otherIdx - otherIdx % stride;
    int tid = s_ + i_ * indSize;
    int id =  s_ + i_ * dimsize;
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      int index_ = index + j;
      if constexpr (std::is_same<T, float>::value) {
          if (TM == 4)(float4 &)output[tid + index_ * stride] = (float4 &)input[id + s_indices[threadIdx.y * TN + j] * stride];
          else if (TM == 2)(float2 &)output[tid + index_ * stride] = (float2 &)input[id + s_indices[threadIdx.y * TN + j] * stride];
          else (float &)output[tid + index_ * stride] = (float &)input[id + s_indices[threadIdx.y * TN + j] * stride];
      } else {
          if (TM == 8) (float4 &)output[tid + index_ * stride] = (float4 &)input[id + s_indices[threadIdx.y * TN + j] * stride];
          else if (TM == 4)(float2 &)output[tid + index_ * stride] = (float2 &)input[id + s_indices[threadIdx.y * TN + j] * stride];
          else if (TM == 2)(half2 &)output[tid + index_ * stride] = (half2 &)input[id + s_indices[threadIdx.y * TN + j] * stride];
          else (half &)output[tid + index_ * stride] = (half &)input[id + s_indices[threadIdx.y * TN + j] * stride];
      }
    }
}

template <typename T, typename Tind>
infiniopStatus_t gather_nv_gpu(void *output, void const *input, void const *indices, uint64_t stride, uint64_t indSize, uint64_t othersize, uint64_t dimsize, void* stream)
{
    int TM = getTM<T>(stride);
    int TN = getTN(indSize);
    int othersize_ = othersize / TM;
    int othersize_remainder = othersize % TM;
    int indSize_ = indSize / TN;
    int indSize_remainder = indSize % TN;
    int BLOCK_DIM_x = getdim(othersize_);
    int BLOCK_DIM_y = getdim(indSize_);
    int num_block_x = (othersize_ + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_block_y = (indSize_ + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    warpGatherKernel_v3<T, Tind>
        <<<grid_dim, block_dim, 0, cuda_stream>>>((T *)output, (T const*)input, (Tind const*)indices, stride, indSize, dimsize, TM, TN, 0, 0);
    if (othersize_remainder != 0 && indSize_remainder != 0) {
        int BLOCK_DIM_x = getdim(othersize_remainder);
        int BLOCK_DIM_y = getdim(indSize_remainder);
        int num_block_x = (othersize_remainder + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (indSize_remainder + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        warpGatherKernel_v3<T, Tind>
            <<<grid_dim, block_dim, 0, cuda_stream>>>((T *)output, (T const*)input, (Tind const*)indices, stride, indSize, dimsize, 1, 1, othersize_ * TM, indSize_ * TN);
    }
    if (othersize_remainder != 0) {
        int BLOCK_DIM_x = getdim(othersize_remainder);
        int BLOCK_DIM_y = getdim(indSize_);
        int num_block_x = (othersize_remainder + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (indSize_ + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        warpGatherKernel_v3<T, Tind>
            <<<grid_dim, block_dim, 0, cuda_stream>>>((T *)output,(T const*)input, (Tind const*)indices, stride, indSize, dimsize, 1, TN, othersize_ * TM, 0);
    } 
    if (indSize_remainder != 0) {
        int BLOCK_DIM_x = getdim(othersize_);
        int BLOCK_DIM_y = getdim(indSize_remainder);
        int num_block_x = (othersize_ + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (indSize_remainder + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        warpGatherKernel_v3<T, Tind>
            <<<grid_dim, block_dim, 0, cuda_stream>>>((T *)output, (T const*)input, (Tind const*)indices, stride, indSize, dimsize, TM, 1, 0, indSize_ * TN);
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGather(GatherCudaDescriptor_t desc,
                         void *output, void const *input, void const *indices,
                         void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16 && desc->dtin == I32) {
        return gather_nv_gpu<half, uint32_t>(output, input, indices, desc->stride, desc->indsize, desc->othersize, desc->dimsize, stream);
    }
    if (desc->dtype == F32 && desc->dtin == I32) {
        return gather_nv_gpu<float, uint32_t>(output, input, indices, desc->stride, desc->indsize, desc->othersize, desc->dimsize, stream);
    }
    if (desc->dtype == F16 && desc->dtin == I64) {
        return gather_nv_gpu<half, uint64_t>(output, input, indices, desc->stride, desc->indsize, desc->othersize, desc->dimsize, stream);
    }
    if (desc->dtype == F32 && desc->dtin == I64) {
        return gather_nv_gpu<float, uint64_t>(output, input, indices, desc->stride, desc->indsize, desc->othersize, desc->dimsize, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
