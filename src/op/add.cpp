#include "op/add.h"
#include "cpu/add_kernel.h"
#include "kernel_interface.h"
op::VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : Layer(device_type,LayerType::kLayerAdd,"Add")
{
    reset_input_size(2);
    reset_output_size(1);
}

base::Status op::VecAddLayer::check() const
{
    tensor::Tensor input1 = this->get_input(0);
    tensor::Tensor input2 = this->get_input(1);

    int32_t size = input1.size();
    base::Status status;
    status = check_tensor_with_dim(input1, device_type_, data_type_, size);
    if (!status) {
        std::cerr << "The input tensor 1 error in the add layer." << std::endl;
        return status;
    }

    status = check_tensor_with_dim(input2, device_type_, data_type_, size);
    if (!status) {
        std::cerr << "The input tensor 2 error in the add layer." << std::endl;
        return status;
    }

    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, size);
    if (!status) {
        std::cerr << "The output tensor error in the add layer." << std::endl;
        return status;
    }
    return base::error::Success();
}

base::Status op::VecAddLayer::forward()
{
    auto status = this->check();
    if(!status){
        return status;
    }
    auto input0 = this->get_input(0);
    auto input1 = this->get_input(1);
    auto output = this->get_output(0);

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        //CHECK(cuda_config_ != nullptr);
    }

    kernel::get_add_kernel(device_type_)(input0, input1, output, nullptr);

    // kernel::get_add_kernel(device_type_)(input1, input2, output,
    //                                     cuda_config_ ? cuda_config_->stream : nullptr);
    return base::error::Success(); 
}
