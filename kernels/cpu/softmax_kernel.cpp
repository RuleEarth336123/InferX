#include "cpu/softmax_kernel.h"
#include <armadillo>
#include "base/base.h"
#include <cblas.h>
namespace kernel {

    void softmax_inplace_cpu(const tensor::Tensor& input, void* stream) {
        int32_t size = static_cast<int32_t>(input.size());
        float* input_ptr = const_cast<float*>(input.ptr<float>());

        // 找到最大值的索引
        CBLAS_INDEX max_index = cblas_isamax(size, input_ptr, 1);

        // 获取最大值
        float max_value = input_ptr[max_index];

        // 计算 exp(x - max_value)
        for (int32_t i = 0; i < size; ++i) {
            input_ptr[i] = expf(input_ptr[i] - max_value);
        }

        // 计算和
        float sum_value = 0.0f;
        sum_value = cblas_sasum(size, input_ptr, 1);

        // 归一化
        float one = 1.0f / sum_value;
        cblas_sscal(size, one, input_ptr, 1);
    }

    // void softmax_inplace_cpu(const float* input_ptr, size_t size) {
    //     tensor::Tensor input(base::DataType::kDataTypeFp32, size);
    //     std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
    //         size * sizeof(float), nullptr, (void*)input_ptr, true);
    //     input.assign(buffer);
    //     return softmax_inplace_cpu(input);
    // }
}