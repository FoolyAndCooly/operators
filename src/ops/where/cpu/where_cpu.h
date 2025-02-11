#ifndef __CPU_where_H__
#define __CPU_where_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

struct WhereCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t c_data_size;
    uint64_t const *c_shape;
    uint64_t const *condition_strides;
    uint64_t const *a_strides;
    uint64_t const *b_strides;
    uint64_t *c_indices;
};

typedef struct WhereCpuDescriptor *WhereCpuDescriptor_t;

infiniopStatus_t cpuCreateWhereDescriptor(infiniopHandle_t,
                                        WhereCpuDescriptor_t *,
					infiniopTensorDescriptor_t condition,
                                        infiniopTensorDescriptor_t c,
                                        infiniopTensorDescriptor_t a,
                                        infiniopTensorDescriptor_t b);

infiniopStatus_t cpuWhere(WhereCpuDescriptor_t desc,
			void const *condition,
                        void *c, void const *a, void const *b,
                        void *stream);

infiniopStatus_t cpuDestroyWhereDescriptor(WhereCpuDescriptor_t desc);

#endif
