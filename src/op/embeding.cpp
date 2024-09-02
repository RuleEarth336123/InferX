#include "op/embeding.h"
#include "cpu/emb_kernel.h"
#include "kernel_interface.h"
#include "op/layer.h"
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
    const auto& token_size = get_input(1).size();

    // 如果token大小大于输入张量的大小，则返回错误信息
    if(token_size > input_tensor.size()){
        return base::error::InvalidArgument("The number of input tensor is greater than seq len.");
    }

    // 检查输入张量是否具有正确的维度、设备类型、数据类型和大小
    base::Status status = check_tensor_with_dim(input_tensor,base::DeviceType::kDeviceCPU,base::DataType::kDataTypeInt32, token_size);
    if(!status){
        LOG(ERROR) << "The input tensor error in the embedding layer.";
        return status;
    }

    // 检查权重张量是否具有正确的维度、设备类型、数据类型、词汇表大小和维度
    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, vocab_size_, dim_);
    if (!status) {
        LOG(ERROR) << "The weight tensor error in the embedding layer.";
        return status;
    }

    // 检查输出张量是否具有正确的维度、设备类型、数据类型、token大小和维度
    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, token_size, dim_);
    if (!status) {
        LOG(ERROR) << "The output tensor error in the embedding layer.";
        return status;
    }
    return base::error::Success();

}

base::Status op::EmbeddingLayer::forward()
{
    base::Status status = check();
    if (!status) {
        return status;
    }
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        ;
    }
    kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), vocab_size_, nullptr);
    return base::StatusCode::kSuccess;
}
