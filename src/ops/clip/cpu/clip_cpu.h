#ifndef __CPU_CLIP_H__
#define __CPU_CLIP_H__

#include "operators.h"
#include <numeric>

struct ClipCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t data_size;
};

typedef struct ClipCpuDescriptor *ClipCpuDescriptor_t;

infiniopStatus_t cpuCreateClipDescriptor(infiniopHandle_t,
                                         ClipCpuDescriptor_t *,
                                         infiniopTensorDescriptor_t y,
                                         infiniopTensorDescriptor_t x,
					 void* min,
					 void* max);

infiniopStatus_t cpuClip(ClipCpuDescriptor_t desc,
                         void *y, void const *x,
			 void *min, void *max,
                         void *stream);

infiniopStatus_t cpuDestroyClipDescriptor(ClipCpuDescriptor_t desc);

#endif
