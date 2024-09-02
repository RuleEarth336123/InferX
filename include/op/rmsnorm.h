#ifndef INCLUDE_OP_RMSNORM_H_
#define INCLUDE_OP_RMSNORM_H_

#include "op/layer.h"

namespace op{
    class RmsnormLayer : public LayerParam{
    public:
        explicit RmsnormLayer(base::DeviceType device_type,int32_t dim);

        base::Status check() const override;
        base::Status forward();

    private:
        int32_t dim_ = 0;
    };
}

#endif