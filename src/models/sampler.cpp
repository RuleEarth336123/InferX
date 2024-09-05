#include "models/sampler.h"

size_t model::ArgmaxSampler::sample(const float *logits, size_t size, void *stream)
{
    if(device_type_ == base::DeviceType::kDeviceCPU){
        size_t next = std::distance(logits, std::max_element(logits, logits + size));
        return next;
    }
}