#ifndef KERNEL_CPU_H_SCALE_KERNEL
#define KERNEL_CPU_H_SCALE_KERNEL

#include "tensor/tensor.h"
namespace kernel{
    /**
 * 使用softmax函数对输入的张量进行原地操作，并计算其结果。
 *
 * @param input 需要进行softmax操作的输入张量。
 * @param stream 用于并行计算的流。
 *
 * @note 该函数会直接修改输入张量，使其变为softmax后的结果。
 */
    void softmax_inplace_cpu(const tensor::Tensor& input, void* stream = nullptr);
    void softmax_inplace_cpu(const float* input_ptr, size_t size);
}

#endif