#ifndef KERNEL_CPU_H_SCALE_KERNEL
#define KERNEL_CPU_H_SCALE_KERNEL

#include "tensor/tensor.h"
namespace kernel{
    void scale_inplace_cpu(const tensor::Tensor& tensor, float scale, void* stream = nullptr);
    void scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale, 
                          const tensor::Tensor& output, int pos, int size, int stride,
                          void* stream);
}

#endif