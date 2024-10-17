#ifndef KERNEL_CPU_H_ROPE_KERNEL
#define KERNEL_CPU_H_ROPE_KERNEL

#include "tensor/tensor.h"

namespace kernel{
    /**
     * 使用CPU执行rope_kernel操作。
     *
     * @param dim 输入张量的维度。
     * @param kv_dim 键值对的维度。
     * @param head_size 头部大小。
     * @param input_q 查询输入张量。
     * @param input_k 键输入张量。
     * @param input_pos 位置输入张量。
     * @param stream 用于并行计算的流。
     */
    // void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, 
    //                 const tensor::Tensor& input_q,
    //                 const tensor::Tensor& input_k, 
    //                 const tensor::Tensor& input_pos, void* stream = nullptr);
    void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache);

    void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                        const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                        const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                        void* stream);
}

#endif