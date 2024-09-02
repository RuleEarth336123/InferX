#include "kernel_interface.h"
#include <armadillo>
#include <iostream>

kernel::AddKernel kernel::get_add_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return add_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
     return nullptr;
}

kernel::MatmulKernel kernel::get_matmul_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return matmul_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
     return nullptr;
}

kernel::RmsnormKernel kernel::get_rmsnorm_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return rmsnorm_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
     return nullptr;
}

kernel::EmbKernel kernel::get_emb_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return emb_kernel_normal;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
     return nullptr;
}

kernel::ScaleKernel kernel::get_scale_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return scale_inplace_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
     return nullptr;
}

kernel::ScaleSumKernel kernel::get_scale_sum_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return scale_sum_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
     return nullptr;
}

kernel::SoftmaxInplaceKernel kernel::get_softmax_inplace_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return softmax_inplace_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
     return nullptr;
}

kernel::RopeKernel kernel::get_rope_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return rope_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
    return nullptr;
}

kernel::SwigluKernel kernel::get_swiglu_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return swiglu_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
    return nullptr;
}

kernel::MhaKernel kernel::get_mha_kernel(base::DeviceType device_type)
{
    if (device_type == base::DeviceType::kDeviceCPU) {
        return mha_kernel_cpu;
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
       ;
    } else {
        //std::cerr << "Unknown device type for get a add kernel." << std::end;
    }
    return nullptr;
}
