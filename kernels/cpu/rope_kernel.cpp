#include "cpu/rope_kernel.h"
#include <cmath>
void kernel::rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor &input_q, const tensor::Tensor &input_k, const tensor::Tensor &input_pos, void *stream)
{
    const int32_t pos = *input_pos.ptr<int32_t>(0);
    for(int32_t i = 0;i < dim;i += 2){
        int32_t head_dim = i % head_size;
        float val = static_cast<float>(pos) * \
            1.0f / std::pow(10000.0f,static_cast<float>(head_dim) / static_cast<float>(head_size));
        
        int32_t rotn = i < kv_dim ? 2 : 1;
        for(int32_t v = 0;v < rotn;v++){
            // the vector to rotate (query or key)
            float* vec = const_cast<float*>(v == 0 ? input_q.ptr<float>() : input_k.ptr<float>());
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i] = v0 * std::cos(val) - v1 * std::sin(val);
            vec[i + 1] = v0 * std::sin(val) + v1 * std::cos(val);
        }
    }
}