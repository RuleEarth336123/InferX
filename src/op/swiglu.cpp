#include "op/swiglu.h"
#include "kernel_interface.h"
#include "cpu/swiglu_kernel.h"
#include <iostream>

op::SwigluLayer::SwigluLayer(base::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type , op::LayerType::kLayerSwiGLU,"SwiGLU"),hidden_dim_(hidden_dim)
{
    reset_input_size(2);
    reset_output_size(1);
}

base::Status op::SwigluLayer::check() const
{
    auto status = check_tensor_with_dim(get_input(0),device_type_,data_type_,hidden_dim_);
    if(!status){
        LOG(ERROR) << "The input tensor 0 error in the rmsnorm layer.";
        return status;
    }

    status = check_tensor_with_dim(get_input(1),device_type_,data_type_,hidden_dim_);
    if(!status){
        LOG(ERROR) << "The weight tensor 1 error in the rmsnorm layer.";
        return status;
    }

    status = check_tensor_with_dim(get_output(0),device_type_,data_type_,hidden_dim_);
    if(!status){
        LOG(ERROR) << "The output tensor error in the rmsnorm layer.";
        return status;
    }
    return base::error::Success();
}

base::Status op::SwigluLayer::forward()
{
    auto status = check();
    if(!status){
        std::cerr << "Check swiglu layer failed." << std::endl;
        return status;
    }

    auto input1 = this->get_input(0);
    auto input2 = this->get_input(1);

    auto output = this->get_output(0);

    kernel::get_swiglu_kernel(device_type_)(
        input1,input2,output,nullptr
    );


    return base::error::Success();
}
