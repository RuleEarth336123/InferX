#include "add_kernel.h"
#include <armadillo>
#include "base/base.h"

#if 0
void kernel::add_kernel_cpu(const tensor::Tensor &input1, const tensor::Tensor &input2, \
            const tensor::Tensor &output, void *stream)
{
    UNUSED(stream);
    CHECK_EQ(input1.is_empty(),false);
    CHECK_EQ(input2.is_empty(),false);
    CHECK_EQ(output.is_empty(),false);

    CHECK_EQ(input1.size(),input2.size());
    CHECK_EQ(input1.size(),output.size());

    arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()),input1.size(),false,true);
    arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()),input2.size(),false,true);
    arma::fvec output_vec(const_cast<float*>(output.ptr<float>()),output.size(),false,true);

    output_vec = input_vec1 + input_vec2;
}
#else
    #include "add_kernel.h"
    #include "base/base.h"
    #include <cblas.h>

    void kernel::add_kernel_cpu(const tensor::Tensor &input1, const tensor::Tensor &input2, 
                const tensor::Tensor &output, void *stream)
    {
        UNUSED(stream);
        CHECK_EQ(input1.is_empty(), false);
        CHECK_EQ(input2.is_empty(), false);
        CHECK_EQ(output.is_empty(), false);

        CHECK_EQ(input1.size(), input2.size());
        CHECK_EQ(input1.size(), output.size());

        // 获取输入向量的指针
        const float* x = input1.ptr<float>();
        const float* y = input2.ptr<float>();
        float* result = const_cast<float*>(output.ptr<float>());

        // 调用 cblas_saxpy 函数执行向量加法
        cblas_saxpy(input1.size(), 1.0f, x, 1, result, 1);
        cblas_saxpy(input1.size(), 1.0f, y, 1, result, 1);
    }

#endif