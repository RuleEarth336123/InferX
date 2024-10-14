#include "cpu/swiglu_kernel.h"
#include <armadillo>
#include "base/base.h"
#include "glog/logging.h"
#include "cblas.h"
namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, void* stream) {
    UNUSED(stream);
    CHECK_EQ(input1.is_empty(),false);
    CHECK_EQ(input2.is_empty(),false);
    CHECK_EQ(output.is_empty(),false);

    CHECK(input1.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(input2.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);
#if 0
  arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false,
                        true);
  arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false,
                        true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false,
                        true);
    //这里%是重载成点乘
  input1_vec %= (1.0f / (1.0f + arma::exp(-input1_vec)));
  output_vec = input1_vec % input2_vec;
#else
    int size = input1.size();
    float* input1_ptr = const_cast<float*>(input1.ptr<float>());
    float* input2_ptr = const_cast<float*>(input2.ptr<float>());
    float* output_ptr = const_cast<float*>(output.ptr<float>());

    // 计算sigmoid函数
    /*
        input1 = input1 * sigmod(input1) = input1 * 1/(1 + exp(-input))
        output = input1 * input2

        function swiglu(x1,x2) = silu(x1) @ x2 = (x1 @ sigmod(x1)) @ x2 
    */
    for (int i = 0; i < size; ++i) {
        input1_ptr[i] = 1.0f / (1.0f + expf(-input1_ptr[i]));
    }

    // 逐元素乘法
    cblas_saxpy(size, 1.0f, input1_ptr, 1, output_ptr, 1);
    cblas_sscal(size, 1.0f, output_ptr, 1);
    cblas_saxpy(size, 1.0f, input2_ptr, 1, output_ptr, 1);
#endif
}
}  // namespace kernel