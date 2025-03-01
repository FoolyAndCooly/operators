#ifndef __CPU_gather_H__
#define __CPU_gather_H__

#include "operators.h"
#include <numeric>
#include <type_traits>

struct GatherCpuDescriptor {
    Device device;
    DT dtype;
    DT dtin;
    uint64_t ndim;
    uint64_t stride;
    uint64_t othersize;
    uint64_t indsize;
    uint64_t dimsize;
};

typedef struct GatherCpuDescriptor *GatherCpuDescriptor_t;

infiniopStatus_t cpuCreateGatherDescriptor(infiniopHandle_t,
                                        GatherCpuDescriptor_t *,
                                        infiniopTensorDescriptor_t output,
                                        infiniopTensorDescriptor_t input,
                                        infiniopTensorDescriptor_t indices,
					int axis);

infiniopStatus_t cpuGather(GatherCpuDescriptor_t desc,
                        void *output, void const *input, void const *indices,
                        void *stream);

infiniopStatus_t cpuDestroyGatherDescriptor(GatherCpuDescriptor_t desc);

#endif
