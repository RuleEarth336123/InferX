#ifndef KERNEL_CPU_H_ADD_KERNEL
#define KERNEL_CPU_H_ADD_KERNEL
#include "tensor/tensor.h"
namespace kernel {
/**
 * 使用CPU对两个输入的张量进行加法运算，并将结果存储在输出张量中。
 *
 * @param input1 第一个输入张量。
 * @param input2 第二个输入张量。
 * @param output 用于存储结果的输出张量。
 * @param stream 未使用的参数，可以忽略。
 */
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel
#endif