#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "where.cuh"

/**
 * @brief A templated vector struct that supports element-wise whereition on arrays.
 *
 * @tparam T - The access data type for elements in the vector.
 * @tparam TComp - The computation data type used for arithmetic operations. 
 * @tparam N - The number of elements of type T in the vector for a single access.
 */

template<typename T, typename TComp, size_t N>
struct vecN {
    T data[N];

    template<typename TCon>
    __device__ __forceinline__ vecN<T, TComp, N> where(const vecN<TCon, uint8_t, N> &condition, const vecN<T, TComp, N> &other) const {
        vecN<T, TComp, N> result;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if constexpr (std::is_same<T, TComp>::value) {
                result.data[i] = condition.data[i] ? data[i] : other.data[i];
            } else {
                constexpr static size_t pack_size = sizeof(T) / sizeof(TComp);
                auto data_ = reinterpret_cast<vecN<TComp, TComp, pack_size> *>(result.data);
                data_[i] = std::move(
		    reinterpret_cast<vecN<TComp, TComp, pack_size> const *>(data)[i].where(
                        reinterpret_cast<vecN<uint8_t, uint8_t, pack_size> const *>(condition.data)[i],
                        reinterpret_cast<vecN<TComp, TComp, pack_size> const *>(other.data)[i])
			);
            }
        }

        return result;
    }

    __device__ __forceinline__ const T &operator[](size_t i) const {
        return data[i];
    }
};

template<typename T>
struct is_vecN : std::false_type {};

template<typename T, typename TComp, size_t N>
struct is_vecN<vecN<T, TComp, N>> : std::true_type {};

template<typename Tdata, typename TdataCon, typename TIdata>
__global__ void where(
    const TdataCon* condition,
    Tdata *c,
    const Tdata *a,
    const Tdata *b,
    const int64_t *a_strides,
    const int64_t *b_strides,
    const int64_t *c_strides,
    const int64_t *condition_strides,
    uint64_t data_size,
    uint64_t ndim,
    uint64_t offset,
    bool broadcasted,
    unsigned pack_size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < data_size) {
        if (broadcasted) {
	    idx *= pack_size;
            auto a_ = reinterpret_cast<const TIdata *>(a);
            auto b_ = reinterpret_cast<const TIdata *>(b);
            auto condition_ = reinterpret_cast<const uint8_t *>(condition);
            auto c_ = reinterpret_cast<TIdata *>(c);
#pragma unroll
	    for (int i = 0; i < pack_size; i++) {
                uint64_t a_idx = getDstOffset(idx + i, ndim, c_strides, a_strides);
                uint64_t b_idx = getDstOffset(idx + i, ndim, c_strides, b_strides);
	        uint64_t condition_idx = getDstOffset(idx + i, ndim, c_strides, condition_strides);
                c_[idx + i] = condition_[condition_idx] ? a_[a_idx] : b_[b_idx];
	    }
            return;
        }
        if constexpr (is_vecN<Tdata>::value) {
            c[idx] = a[idx].where(condition[idx], b[idx]);
	} else {
            c[idx] = condition[idx] ? a[idx] : b[idx];
	}
    }
}

template<typename Tdata, typename TdataCon, typename TIdata>
void _where_nv_gpu(WhereCudaDescriptor_t desc, TdataCon const *condition, Tdata *c, Tdata const *a, Tdata const *b, uint64_t data_size, uint64_t pack_size, uint64_t offset, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < data_size; i += step) {
        where<Tdata, TdataCon, TIdata><<<gridDims, blockDims, 0, cuda_stream>>>(
           condition, c, a, b, desc->a_strides, desc->b_strides, desc->c_strides, desc->condition_strides, offset + data_size, desc->ndim, offset + i, desc->broadcasted, pack_size);
    }
}

template<typename Tdata, typename TdataCon, typename TIdata>
infiniopStatus_t where_nv_gpu(WhereCudaDescriptor_t desc, void const *condition, void *c, void const *a, void const *b, void *stream, uint64_t pack_size) {
    const uint64_t total_elements = desc->c_data_size;
    const uint64_t data_size = total_elements / pack_size;
    const uint64_t remainder = total_elements % pack_size;
    
    if (data_size > 0) {
        const auto condition_vec = reinterpret_cast<const TdataCon *>(condition);
        const auto a_vec = reinterpret_cast<const Tdata *>(a);
        const auto b_vec = reinterpret_cast<const Tdata *>(b);
        const auto c_vec = reinterpret_cast<Tdata *>(c);
        _where_nv_gpu<Tdata, TdataCon, TIdata>(desc, condition_vec, c_vec, a_vec, b_vec, data_size, pack_size, 0, stream);
    }

    if (remainder > 0) {
        const auto condition_vec = reinterpret_cast<const uint8_t *>(condition);
        const auto a_vec = reinterpret_cast<const TIdata *>(a);
        const auto b_vec = reinterpret_cast<const TIdata *>(b);
        const auto c_vec = reinterpret_cast<TIdata *>(c);
        _where_nv_gpu<TIdata, uint8_t, TIdata>(desc, condition_vec, c_vec, a_vec, b_vec, remainder, 1, data_size * pack_size, stream);
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaWhere(WhereCudaDescriptor_t desc,
			 void const *condition,
                         void *c, void const *a, void const *b,
                         void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return where_nv_gpu<vecN<half, half, 4>, vecN<uint8_t, uint8_t, 4>, half>(desc, condition, c, a, b, stream, 4);
    }
    if (desc->dtype == F32) {
        return where_nv_gpu<vecN<float2, float, 2>, vecN<uint16_t, uint8_t, 2>, float>(desc, condition, c, a, b, stream, 4);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
