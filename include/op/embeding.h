#ifndef INCLUDE_OP_EMBEDING_H_
#define INCLUDE_OP_EMBEDING_H_

#include <utility>
#include "layer.h"

namespace op{

    struct embedingOutput{
        tensor::Tensor input_tokens;
        tensor::Tensor input_embedings;
        tensor::Tensor input_token_num;
        explicit embedingOutput(tensor::Tensor input_tokens,tensor::Tensor input_embedings,\
            tensor::Tensor input_token_num) 
            : input_tokens(std::move(input_tokens)),\
            input_embedings(std::move(input_embedings)),\
            input_token_num(std::move(input_token_num)){}
    };

    class  EmbeddingLayer: public LayerParam{
    public:
        /**
         * 构造函数
         */
        explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                          int32_t vocab_size);
        base::Status check() const override;
        base::Status forward() override;

    private:
        int32_t dim_ = 0;
        int32_t seq_len_ = 0;
        int32_t vocab_size_ = 0;
    };

}//namespace op

#endif