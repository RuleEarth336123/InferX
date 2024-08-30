#ifndef KERNEL_CPU_H_MATMUL_KERNEL
#define KERNEL_CPU_H_MATMUL_KERNEL

#include "tensor/tensor.h"
#include "base/base.h"
namespace kernel{
    /**
 * 使用CPU设备进行矩阵乘法运算。
 *
 * @param input 输入的张量，用于计算矩阵乘法。
 * @param weight 权重的张量，用于计算矩阵乘法。
 * @param output 输出的张量，用于存储计算结果。
 * @param scale 缩放因子，用于调整计算结果。
 */
    void matmul_kernel_cpu(const tensor::Tensor& input,const tensor::Tensor& weight,
                const tensor::Tensor& output,float scale = 1.f);
}

#endif