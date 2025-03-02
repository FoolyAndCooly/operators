#include "reduce_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateReduceDescriptor(infiniopHandle_t,
                                         ReduceCpuDescriptor_t *desc_ptr,
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
    *desc_ptr = new ReduceCpuDescriptor{
        DevCpu,
	op,
        data->dt,
	data->shape,
	data->ndim
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyReduceDescriptor(ReduceCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename T>
infiniopStatus_t reduce_cpu(ReduceCpuDescriptor_t desc, void *reduced, void const *data, void const *axes) {
    auto reduced_ = reinterpret_cast<T *>(reduced);
    auto data_ = reinterpret_cast<T const*>(data);
    auto axes_ = reinterpret_cast<int64_t const*>(axes);
    uint64_t stride = std::accumulate(desc->shape + *axes_ + 1, desc->shape + desc->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t othersize = std::accumulate(desc->shape, desc->shape + *axes_, 1ULL, std::multiplies<uint64_t>());
    uint64_t dimsize = desc->shape[*axes_];
    for (int i = 0; i < othersize; i++) {
        for (int j = 0; j < stride; j++) {
            int tid = i * dimsize * stride + j;
	    if (desc->op == 0) {
                T max = std::numeric_limits<T>::min();
                for (int k = 0; k < dimsize; k++) {
                    T element = data_[tid + k * stride];
                    max = (element > max) ? element : max;
                }
                int outid = i * stride + j;
                reduced_[outid] = max;
	    }
	    else if (desc->op == 1) {
                T min = std::numeric_limits<T>::max();
                for (int k = 0; k < dimsize; k++) {
                    T element = data_[tid + k * stride];
                    min = (element < min) ? element : min;
                }
                int outid = i * stride + j;
                reduced_[outid] = min;
	    }	
	    else if (desc->op == 2) {
                T mean = 0;
                for (int k = 0; k < dimsize; k++) {
                    T element = data_[tid + k * stride];
                    mean =  element + mean;
                }
                int outid = i * stride + j;
                reduced_[outid] = mean / dimsize;
	    }
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuReduce(ReduceCpuDescriptor_t desc,
                        void *reduced, void const *data, void const *axes,
                        void *stream) {
    if (desc->dtype == F16) {
        return reduce_cpu<uint16_t>(desc, reduced, data, axes);
    }
    if (desc->dtype == F32) {
        return reduce_cpu<float>(desc, reduced, data, axes);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
