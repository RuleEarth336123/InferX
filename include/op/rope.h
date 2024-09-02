#ifndef INCLUDE_OP_ROPE_H_
#define INCLUDE_OP_ROPE_H_

#include "layer.h"

namespace op{
    class RopeLayer : public Layer{
    public:
        explicit RopeLayer(base::DeviceType device_type,int32_t dim,int32_t kv_dim,int32_t head_size);

        base::Status check() const override;
        base::Status forward() override;

    private:
        int32_t dim_ = 0;
        int32_t kv_dim_ = 0;
        int32_t head_size_  = 0;
    };
}

#endif
