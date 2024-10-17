#ifndef KERNEL_H_INTERFACE
#define KERNEL_H_INTERFACE

#include "base/base.h"
#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/scale_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/swiglu_kernel.h"
#include <functional>

namespace kernel{
    // typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
    //                       const tensor::Tensor& output, void* stream);
    using AddKernel = std::function<void(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, void*)>; 
    using MatmulKernel = std::function<void(const tensor::Tensor&, const tensor::Tensor&, const tensor::Tensor&, float)>;
    using RmsnormKernel = std::function<void(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, void*)>;
    using EmbKernel = std::function<void(const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, int32_t, void*)>;
    using ScaleKernel = std::function<void(const tensor::Tensor&,float,void*)>;
    using ScaleSumKernel = std::function<void(const tensor::Tensor& , const tensor::Tensor& , const tensor::Tensor& , int , int , int ,void* )>;
    using SoftmaxInplaceKernel = std::function<void(const tensor::Tensor&,void*)>;
    using RopeKernel = std::function<void(int32_t , int32_t , int32_t , \
        const tensor::Tensor& ,const tensor::Tensor& , const tensor::Tensor& , const tensor::Tensor& , const tensor::Tensor& , void*)>;
    using SwigluKernel = std::function<void(const tensor::Tensor& , const tensor::Tensor& ,const tensor::Tensor&, void*)>;
    using MhaKernel = std::function<void(int32_t , int32_t , int32_t , int32_t , int32_t ,\
                int32_t , int32_t , const tensor::Tensor& ,const tensor::Tensor& , \
                const tensor::Tensor& ,const tensor::Tensor& , const tensor::Tensor& ,base::DeviceType)>;

    kernel::AddKernel get_add_kernel(base::DeviceType device_type);
    kernel::MatmulKernel get_matmul_kernel(base::DeviceType device_type);
    kernel::MatmulKernel get_matmul_kernel(base::DeviceType device_type);
    kernel::RmsnormKernel get_rmsnorm_kernel(base::DeviceType device_type);
    kernel::EmbKernel get_emb_kernel(base::DeviceType device_type);
    kernel::ScaleKernel get_scale_kernel(base::DeviceType device_type);
    kernel::ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);
    kernel::SoftmaxInplaceKernel get_softmax_inplace_kernel(base::DeviceType device_type);
    kernel::RopeKernel get_rope_kernel(base::DeviceType device_type);
    kernel::SwigluKernel get_swiglu_kernel(base::DeviceType device_type);
    kernel::MhaKernel get_mha_kernel(base::DeviceType device_type);
}

#endif