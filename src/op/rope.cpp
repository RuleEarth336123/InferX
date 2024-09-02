#include "op/rope.h"
#include "kernel_interface.h"
#include "cpu/rope_kernel.h"
#include <iostream>

op::RopeLayer::RopeLayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size)
    : Layer(device_type,LayerType::kLayerRoPe,"RoPe"),
    dim_(dim),
    kv_dim_(kv_dim),
    head_size_(head_size)
{
    reset_input_size(3);
    reset_output_size(1);
}

base::Status op::RopeLayer::check() const
{
    auto status = check_tensor_with_dim(get_input(0),device_type_,data_type_,dim_);
    if(!status){
        LOG(ERROR) << "The input tensor 0 error in the rmsnorm layer.";
        return status;
    }

    status = check_tensor_with_dim(get_input(1),device_type_,data_type_,dim_);
    if(!status){
        LOG(ERROR) << "The weight tensor 1 error in the rmsnorm layer.";
        return status;
    }

    status = check_tensor_with_dim(get_input(2),device_type_,data_type_,dim_);
    if(!status){
        LOG(ERROR) << "The weight tensor 2 error in the rmsnorm layer.";
        return status;
    }
    return base::error::Success();
}

base::Status op::RopeLayer::forward()
{
    using namespace tensor;
    auto status = check();
    if(!status){
        std::cerr << "Check rope layer failed." << std::endl;
        return status;
    }

    Tensor input_q = get_input(0);
    Tensor input_k = get_input(1);
    Tensor input_pos = get_input(2);

    kernel::get_rope_kernel(device_type_)(
        dim_,kv_dim_,head_size_,input_q,input_k,input_pos,nullptr
    );

    return base::error::Success();
}
