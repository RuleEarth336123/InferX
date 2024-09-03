#ifndef MODEL_H_CONFIG_
#define MODEL_H_CONFIG_

namespace model{
    struct ModelConfig {
        int32_t dim = 0;// 模型的维度，通常用于表示模型中向量的维度。
        int32_t hidden_dim = 0;// 隐藏层的维度，用于神经网络中隐藏层的输出大小。
        int32_t layer_num = 0;// 模型中的层数，通常用于控制神经网络的深度。
        int32_t head_num = 0;// 多头注意力机制中的头数，用于分割和并行处理信息。
        int32_t kv_head_num = 0;// 键值对（Key-Value pairs）的头数，通常用于注意力机制中的键和值的表示。
        int32_t vocab_size = 0;// 词汇表的大小，用于表示模型能够处理的不同词汇的数量。
        int32_t seq_len = 0;// 序列的长度，用于控制模型处理的输入序列的最大长度。
    };

    struct TransformerConfig {
        
        int32_t kv_dim_ = 0;// 键值对（Key-Value pairs）的维度，用于定义键和值的向量大小。
        int32_t kv_mul_ = 0;// 键值对维度的乘数，用于调整键和值的维度大小。
        int32_t head_size_ = 0;// 每个注意力头的大小，通常用于定义每个头处理的向量维度。
        int32_t vocab_size_ = 0;// 词汇表的大小，用于表示模型能够处理的不同词汇的数量。
        int32_t dim_ = 0; // 模型的维度，通常用于表示模型中向量的维度。
        int32_t hidden_dim_ = 0;// 隐藏层的维度，用于神经网络中隐藏层的输出大小。
        int32_t layer_num_ = 0;// 模型中的层数，通常用于控制神经网络的深度。
        int32_t head_num_ = 0;// 多头注意力机制中的头数，用于分割和并行处理信息。
        int32_t kv_head_num_ = 0;// 键值对的头数，通常用于注意力机制中的键和值的表示。
        int32_t seq_len_ = 0; // 序列的长度，用于控制模型处理的输入序列的最大长度。
        bool is_shared_weight_ = 0;// 是否共享权重，通常用于控制模型中的某些层是否共享权重。
    };
}



#endif 