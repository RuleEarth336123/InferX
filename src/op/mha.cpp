#include "op/mha.h"
#include "kernel_interface.h"
#include "cpu/mha_kernel.h"
#include <iostream>

op::MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t layer_index, \
    int32_t kv_mul, int32_t kv_dim, int32_t seq_len, int32_t head_num, int32_t head_size)
    :  Layer(device_type, LayerType::kLayerMHA,"MultiHead")
{
    
            
    layer_index_ = layer_index;
    kv_mul_ = kv_mul;
    kv_dim_ = kv_dim;
    seq_len_ = seq_len;
    head_num_ = head_num;
    head_size_ = head_size;
    reset_input_size(5);
    reset_output_size(1);
       
}

base::Status op::MultiHeadAttention::check() const
{
    const int32_t input_tensor_num = 4;
    for(int32_t i=0; i < input_tensor_num; i++){
        auto status = check_tensor(get_input(i),device_type_,data_type_);
        if(!status){
            LOG(ERROR) << "The input tensor : "<< std::to_string(i) << " error in the matmul layer.";
            return status;
        }
    }
    auto status = check_tensor(get_output(0),device_type_,data_type_);
    if(!status){
        LOG(ERROR) << "The output tensor" << " error in the matmul layer.";
        return status;
    }
    
    return base::error::Success();
}

base::Status op::MultiHeadAttention::forward()
{
    using namespace tensor;
    auto status = check();
    if(!status){
        std::cerr << "check multiheadattention layer faied!"<<std::endl;
        return status;
    }
    const Tensor& mha_out = get_output(0);

    const Tensor& query_tensor = get_input(0);
    const Tensor& score_tensor = get_input(1);
    const Tensor& key_cache_tensor = get_input(2);
    const Tensor& value_cache_tensor = get_input(3);

    if(device_type_ == base::DeviceType::kDeviceCPU){
        kernel::get_mha_kernel(device_type_)(pos_, head_num_, layer_index_, seq_len_, kv_dim_, kv_mul_,\
            head_size_, mha_out, query_tensor, score_tensor,
            key_cache_tensor, value_cache_tensor, device_type_);
    }
    // using MhaKernel = std::function<void(int32_t , int32_t , int32_t , int32_t , int32_t ,\
    //             int32_t , int32_t , const tensor::Tensor& ,const tensor::Tensor& , \
    //             const tensor::Tensor& ,const tensor::Tensor& , const tensor::Tensor& ,base::DeviceType)>;

    return base::error::Success();
}

void op::MultiHeadAttention::set_pos(int32_t pos)
{
    this->pos_ = pos;
}

void op::MultiHeadAttention::set_layer_idx(int32_t layer_idx)
{
    this->layer_index_ = layer_idx;
}
