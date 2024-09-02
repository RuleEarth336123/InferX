#include "op/matmul.h"
#include "cpu/matmul_kernel.h"
#include "kernel_interface.h"
#include <iostream>
op::MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1, bool is_quant_layer)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer,"Matmul")
{
    dim0_ = dim0;
    dim1_ = dim1;
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
}

base::Status op::MatmulLayer::check() const
{
    auto status = check_tensor_with_dim(get_input(0),device_type_,data_type_,dim1_);
    if(!status){
        LOG(ERROR) << "The input tensor error in the matmul layer.";
        return status;
    }

    
    status = check_tensor_with_dim(get_weight(0),device_type_,data_type_,dim0_,dim1_);
    if(!status){
        LOG(ERROR) << "The weight tensor error in the matmul layer.";
        return status;
    }

    status = check_tensor_with_dim(scales_,device_type_,base::DataType::kDataTypeFp32,scales_.size());
    if(!status){
        LOG(ERROR) << "The scale tensor error in the matmul layer.";
        return status;
    }

    status = check_tensor_with_dim(get_output(0),device_type_,data_type_,dim0_);
    if(!status){
        LOG(ERROR) << "The output tensor error in the matmul layer.";
        return status;
    }
    return base::error::Success();
}

base::Status op::MatmulLayer::forward()
{
    auto status = check();
    if(!status){
        std::cerr << "Check matmul layer failed." << std::endl;
        return status;
    }

    if(device_type_ == base::DeviceType::kDeviceCPU){
        if(is_quant_layer_){    //是不是量化
            // kernel::get_matmul_kernel_quant8(device_type_)(
            //     get_input(0),get_weight(0),get_output(0),1.f,nullptr
            // );
            kernel::get_matmul_kernel(device_type_)(
                get_input(0),get_weight(0),get_output(0),1.f
            );
        }else{
            kernel::get_matmul_kernel(device_type_)(
                get_input(0),get_weight(0),get_output(0),1.f
            );
        }
    }else{
        ;
    }
    return base::error::Success();
}
