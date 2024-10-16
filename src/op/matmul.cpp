#include "op/matmul.h"
#include "cpu/matmul_kernel.h"
#include "kernel_interface.h"
#include <iostream>

op::MatmulLayer::MatmulLayer(base::DeviceType device_type,int32_t dim0,int32_t dim1,bool is_quant_layer,bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer,"Matmul")
{
    dim0_ = dim0;
    dim1_ = dim1;
    has_bias_ = has_bias;
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
    if(has_bias_){
        bias_.resize(1);
    }
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

        if(has_bias_){
            kernel::get_add_kernel(device_type_)(get_output(0), get_bias(0), get_output(0),nullptr);
        }


    }else{
        ;
    }
    return base::error::Success();
}

base::Status op::MatmulLayer::set_bias(int32_t idx, int32_t &dims, const void *bias_ptr, base::DeviceType device_type)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    CHECK_NE(bias_ptr, nullptr);

    size_t size = dims * sizeof(float);
    std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
    
    if(device_type != base::DeviceType::kDeviceUnknown){
        buffer->set_device_type(device_type);
    }

    if(!is_quant_layer_){
        tensor::Tensor bias(base::DataType::kDataTypeFp32,dims);
        bias.set_device_type(device_type);
        CHECK(bias.assign(buffer));
        bias_.at(idx) = bias;
    }else{
        tensor::Tensor bias(base::DataType::kDataTypeInt8,dims);
        bias.set_device_type(device_type);
        CHECK(bias.assign(buffer));

        bias_.at(idx) = bias;

        const int32_t bias_size = static_cast<int32_t>(bias.size());
        CHECK(bias_size % group_size_ == 0);

        int32_t scale_nums = bias_size / group_size_;
        scales_ = tensor::Tensor{base::DataType::kDataTypeFp32,scale_nums,false,nullptr,
            reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
        scales_.set_device_type(device_type);
    }
    return base::error::Success();
}

tensor::Tensor &op::MatmulLayer::get_bias(int32_t idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
}

const tensor::Tensor &op::MatmulLayer::get_bias(int32_t idx) const
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
}
