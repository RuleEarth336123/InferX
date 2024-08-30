#ifndef KERNEL_CPU_H_MHA_KERNEL
#define KERNEL_CPU_H_MHA_KERNEL

#include "tensor/tensor.h"
#include "base/base.h"
namespace kernel{
    /**
     * 使用多头自注意力机制（Multi-Head Attention）对给定的序列进行操作。
     *
     * @param pos 当前处理的位置。
     * @param head_num 使用的头的数量。
     * @param layer_index 当前的层索引。
     * @param seq_len 序列的长度。
     * @param kv_dim 键值对的维度。
     * @param kv_mul 键值对的乘数。
     * @param head_size 每个头的尺寸。
     * @param mha_out 输出的张量，用于存储多头自注意力的结果。
     * @param query_tensor 查询张量，用于存储查询结果。
     * @param score_tensor 分数张量，用于存储得分结果。
     * @param key_cache_tensor 键缓存张量，用于存储键的缓存结果。
     * @param value_cache_tensor 值缓存张量，用于存储值的缓存结果。
     * @param device_type 设备类型，可以是CPU或CUDA。
     * @param config CUDA配置信息，如果设备类型为CUDA时需要提供。
     */
    void mha_kernel_cpu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type);
}

#endif