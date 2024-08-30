#ifndef KERNEL_CPU_H_EMB_KERNEL
#define KERNEL_CPU_H_EMB_KERNEL
#include "base/base.h"
#include "tensor/tensor.h"

namespace kernel{
    /**
     * 使用给定的输入、权重和输出张量，以及词汇大小，在CPU设备上执行嵌入核归一化操作。
     *
     * @param input 输入张量，包含要进行嵌入的数据。
     * @param weight 权重张量，用于对输入数据进行变换。
     * @param output 输出张量，存储经过嵌入核归一化后的结果。
     * @param vocab_size 词汇表的大小，用于确定嵌入的维度。
     * @param stream CUDA流，用于并行计算。
     */
    void emb_kernel_normal(const tensor::Tensor& input,const tensor::Tensor& weight,\
        const tensor::Tensor& output,int32_t vocab_size,
        void* stream = nullptr);
}

#endif