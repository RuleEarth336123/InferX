#ifndef MODEL_H_SAMPLER_
#define MODEL_H_SAMPLER_

#include <cstddef>
#include <cstdint>
#include "base/base.h"

namespace model{
    class Sampler{
    public:
        explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}
        virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;
    protected:
        base::DeviceType device_type_;
    };
   
    class ArgmaxSampler : public Sampler{
    public:
        explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}
        size_t sample(const float* logits, size_t size, void* stream) override;
    };

}


#endif