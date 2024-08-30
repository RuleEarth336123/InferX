#ifndef KERNEL_CPU_H_RMSNORM_KERNEL
#define KERNEL_CPU_H_RMSNORM_KERNEL

#include "tensor/tensor.h"

namespace kernel{
/**
 * 使用RMSnorm方法在CPU上对输入张量进行归一化处理。
 *
 * @param input 输入的张量，需要是CPU设备类型，且不能为空。
 * @param weight 权重张量，需要是CPU设备类型，且不能为空。
 * @param output 输出的张量，需要是CPU设备类型，且不能为空。
 * @param stream 用于并行计算的流，可以为nullptr。
 */
   void rmsnorm_kernel_cpu(const tensor::Tensor& input,const tensor::Tensor& weight,
                const tensor::Tensor& output,void* stream = nullptr);
}

#endif