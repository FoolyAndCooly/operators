#include "clip_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                         ClipCpuDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x,
					 void* min,
					 void* max) {
    uint64_t ndim = y->ndim;
    if (ndim != x->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (y->shape[i] != x->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t data_size = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>());

    *desc_ptr = new ClipCpuDescriptor{
        DevCpu,
        y->dt,
        data_size
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t clip_cpu(ClipCpuDescriptor_t desc, void *y, void const *x, void* min, void* max) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    auto min_ = reinterpret_cast<Tdata*>(min);
    auto max_ = reinterpret_cast<Tdata*>(max);

#pragma omp parallel for
    for (uint64_t i = 0; i < desc->data_size; ++i) {
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            float x_f32 = f16_to_f32(x_[i]);
	    float min_f32 = f16_to_f32(*min_);
	    float max_f32 = f16_to_f32(*max_);
            y_[i] = f32_to_f16((x_f32 < min_f32) ? min_f32 : ((x_f32 > max_f32) ? max_f32 : x_f32));
        } else {
            Tdata x_val = x_[i];
	    Tdata min_val = *min_;
	    Tdata max_val = *max_;
            y_[i] = (x_val < min_val) ? min_val : ((x_val > max_val) ? max_val : x_val);
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *y, void const *x, void *min, void* max,
                         void *stream) {
    if (desc->dtype == F16) {
        return clip_cpu<uint16_t>(desc, y, x, min, max);
    }
    if (desc->dtype == F32) {
        return clip_cpu<float>(desc, y, x, min, max);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
