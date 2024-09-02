#ifndef INCLUDE_OP_MATMUL_H_
#define INCLUDE_OP_MATMUL_H_

#include "op/layer.h"

namespace op{

    class MatmulLayer : public LayerParam{
    public:
        explicit MatmulLayer(\
            base::DeviceType device_type,int32_t dim0,int32_t dim1,bool is_quant_layer = false\
        );

        base::Status check() const override;
        base::Status forward() override;

    private:
        int dim0_ = 0;
        int dim1_ = 0;
    };

}//namespace op


#endif
