#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "clip.cuh"
template<typename T, typename TComp, size_t N>
struct vecN {
    T data[N];
    constexpr static size_t pack_size = sizeof(T) / sizeof(TComp);

    // Constructor that initializes the data array with type TComp
    __device__ __forceinline__ constexpr vecN(const TComp &val) {
        const auto data_ = reinterpret_cast<TComp *>(data);
        const auto size = N * pack_size;
#pragma unroll
        for (size_t i = 0; i < size; ++i) {
            data_[i] = 0;
        }
    }

    // Assignment operator with relu assignment logic
    __device__ __forceinline__ vecN<T, TComp, N> clip(const vecN<T, TComp, N> &other, TComp min, TComp max) {
        if constexpr (std::is_same<T, TComp>::value) {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                data[i] = (other.data[i] < min) ? min : ((other.data[i] > max) ? max : other.data[i]);
            }
        } else {
            auto *data_this = reinterpret_cast<vecN<TComp, TComp, pack_size> *>(data);
            auto *data_other = reinterpret_cast<const vecN<TComp, TComp, pack_size> *>(other.data);
#pragma unroll
            for (int i = 0; i < N; ++i) {
                data_this[i].clip(data_other[i], min, max);
            }
        }
        return *this;
    }

    __device__ __forceinline__ const T &operator[](size_t i) const {
        return data[i];
    }
};


template<typename T>
struct is_vecN : std::false_type {};

template<typename T, typename TComp, size_t N>
struct is_vecN<vecN<T, TComp, N>> : std::true_type {};

template<typename Tdata, typename TIdata>
__global__ void clip_kernel(
    Tdata* y,
    const Tdata* x,
    TIdata* min,
    TIdata* max,
    uint64_t data_size,
    uint64_t offset) {
    
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx >= data_size) return;

    if constexpr (is_vecN<Tdata>::value) {
        y[idx].clip(x[idx], *min, *max);
    } else {
        TIdata val = x[idx];
        y[idx] = (val < *min) ? *min : ((val > *max) ? *max : val);
    }
}

template<typename Tdata, typename TIdata>
void _clip_nv_gpu(ClipCudaDescriptor_t desc, Tdata* y, const Tdata* x, 
                 TIdata* min, TIdata* max, uint64_t data_size, 
                 uint64_t offset, void* stream) {
    if (data_size == 0) return;

    dim3 blockDims(std::min(static_cast<uint64_t>(256), data_size));
    dim3 gridDims(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    clip_kernel<<<gridDims, blockDims, 0, cuda_stream>>>(y, x, min, max, data_size, offset);
}

template<typename Tdata, typename TIdata>
infiniopStatus_t clip_nv_gpu(ClipCudaDescriptor_t desc, void* y, const void* x,
                            void* min, void* max, void* stream, uint64_t pack_size) {
    const uint64_t total_elements = desc->data_size;
    const uint64_t packed_size = total_elements / pack_size;
    const uint64_t remainder = total_elements % pack_size;

    if (packed_size > 0) {
        const auto* x_packed = reinterpret_cast<const Tdata*>(x);
        auto* y_packed = reinterpret_cast<Tdata*>(y);
        _clip_nv_gpu(desc, y_packed, x_packed, 
                    reinterpret_cast<TIdata*>(min), 
                    reinterpret_cast<TIdata*>(max),
                    packed_size, 0, stream);
    }

    if (remainder > 0) {
        const auto* x_remain = reinterpret_cast<const TIdata*>(x) + packed_size * pack_size;
        auto* y_remain = reinterpret_cast<TIdata*>(y) + packed_size * pack_size;
        _clip_nv_gpu<TIdata, TIdata>(desc, y_remain, x_remain,
                                   reinterpret_cast<TIdata*>(min),
                                   reinterpret_cast<TIdata*>(max),
                                   remainder, 0, stream);
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaClip(ClipCudaDescriptor_t desc,
                          void *y, void const *x,
			  void *min, void *max,
                          void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return clip_nv_gpu<vecN<half, half, 4>, half>(desc, y, x, min, max, stream, 4);
    }
    if (desc->dtype == F32) {
        return clip_nv_gpu<vecN<float2, float, 2>, float>(desc, y, x, min, max, stream, 4);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
