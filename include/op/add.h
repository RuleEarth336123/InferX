#ifndef OP_H_ADD
#define OP_H_ADD

#include "base/base.h"
#include "op/layer.h"

namespace op{
    class VecAddLayer : public Layer{
    public:
        explicit VecAddLayer(base::DeviceType device_type);
        base::Status check() const override;
        base::Status forward() override;
    };
}


#endif

