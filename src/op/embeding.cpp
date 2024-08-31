#include "op/embeding.h"
#include <iostream>
op::EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerParam(device_type, LayerType::kLayerEmbedding, false, "Embedding") {
    reset_weight_size(1);
    reset_input_size(2);
    reset_output_size(1);
}

base::Status op::EmbeddingLayer::check() const
{
    const auto& input_tensor = get_input(0);
    const auto& input_size = get_input(1).size();
    if(token_size > input_tensor.size()){
        
    }


}

base::Status op::EmbeddingLayer::forward()
{
    return base::Status();
}
