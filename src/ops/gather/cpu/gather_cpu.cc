#include "gather_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t,
                                        GatherCpuDescriptor_t *desc_ptr,
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
    uint64_t othersize = std::accumulate(input->shape, input->shape + indices->ndim, 1ULL, std::multiplies<uint64_t>()) / input->shape[axis];
    uint64_t indsize = std::accumulate(indices->shape, indices->shape + indices->ndim, 1ULL, std::multiplies<uint64_t>());
    uint64_t dimsize = input->shape[axis];
    *desc_ptr = new GatherCpuDescriptor{
        DevCpu,
        output->dt,
	indices->dt,
        ndim,
	stride,
	othersize,
	indsize,
	dimsize
    };
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata, typename Tind>
infiniopStatus_t gather_cpu(GatherCpuDescriptor_t desc, void *output, void const *input, void const *indices) {
    auto input_ = reinterpret_cast<Tdata const *>(input);
    auto indices_ = reinterpret_cast<Tind const *>(indices);
    auto output_ = reinterpret_cast<Tdata *>(output);
    uint64_t stride = desc->stride;
    uint64_t othersize = desc->othersize;
    uint64_t indsize = desc->indsize;
    uint64_t dimsize = desc->dimsize;
    for (uint64_t i = 0; i < othersize; i++) {
        uint64_t s_ = i % stride;
        uint64_t i_ = i - i % stride;
        uint64_t tid = s_ + i_ * indsize;
        uint64_t id =  s_ + i_ * dimsize;
	for (uint64_t j = 0; j < indsize; j++) {
            output_[tid + j * stride] = input_[id + indices_[j] * stride];
	}
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                        void *output, void const *input, void const *indices,
                        void *stream) {
    if (desc->dtype == F16 && desc->dtin == I32) {
        return gather_cpu<uint16_t, int32_t>(desc, output, input, indices);
    }
    if (desc->dtype == F32 && desc->dtin == I32) {
        return gather_cpu<float, int32_t>(desc, output, input, indices);
    }
    if (desc->dtype == F16 && desc->dtin == I64) {
        return gather_cpu<uint16_t, int64_t>(desc, output, input, indices);
    }
    if (desc->dtype == F32 && desc->dtin == I64) {
        return gather_cpu<float, int64_t>(desc, output, input, indices);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
