#include "cpu/matmul_kernel.h"
#include <armadillo>

#if 0
void kernel::matmul_kernel_cpu(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output, float scale)
{
    CHECK(input.is_empty() == false);
    CHECK(weight.is_empty() == false);
    CHECK(output.is_empty() == false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    const float* input_ptr = input.ptr<float>();
    const float* weight_ptr = weight.ptr<float>();
    const float* output_ptr = output.ptr<float>();

    std::vector<int32_t> in_dims(2,1);
    for(int32_t i = 0; i < input.dims_size(); i++){
        in_dims[i] = input.get_dim(i);
    }

    CHECK_EQ(weight.dims_size(),2);

    std::vector<int32_t> wei_dims(2,0);
    for(int32_t i = 0; i < weight.dims_size(); i++){
        wei_dims[i] = weight.get_dim(i);
    }

    CHECK_EQ(in_dims[0],wei_dims[1]);
    CHECK_EQ(output.size(),wei_dims[0] * in_dims[1]);
    
    arma::fmat input_mat(const_cast<float*>(input_ptr),in_dims[1],in_dims[0],false,true);
    arma::fmat weight_mat(const_cast<float*>(weight_ptr),wei_dims[1],wei_dims[0],false,true);
    arma::fmat output_mat(const_cast<float*>(output_ptr),in_dims[1],wei_dims[0],false,true);
    
    output_mat = (input_mat * weight_mat) * scale;
    
}

#else

#include <cblas.h>

void kernel::matmul_kernel_cpu(const tensor::Tensor &input, const tensor::Tensor &weight, const tensor::Tensor &output, float scale)
{
    CHECK(input.is_empty() == false);
    CHECK(weight.is_empty() == false);
    CHECK(output.is_empty() == false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    const float* input_ptr = input.ptr<float>();
    const float* weight_ptr = weight.ptr<float>();
    float* output_ptr = const_cast<float*>(output.ptr<float>());

    std::vector<int32_t> in_dims(2,1);
    for(int32_t i = 0; i < input.dims_size(); i++){
        in_dims[i] = input.get_dim(i);
    }

    CHECK_EQ(weight.dims_size(),2);

    std::vector<int32_t> wei_dims(2,0);
    for(int32_t i = 0; i < weight.dims_size(); i++){
        wei_dims[i] = weight.get_dim(i);
    }

    CHECK_EQ(in_dims[0],wei_dims[1]);
    CHECK_EQ(output.size(),wei_dims[0] * in_dims[1]);
    
    int m = wei_dims[0]; // number of rows of B
    int n = in_dims[1]; // number of columns of C
    int k = in_dims[0]; // number of columns of A and rows of B
    float alpha = 1.0f; // scale factor for C = A * B
    float beta = 0.0f; // scale factor for C = beta * C

    // Perform matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                 alpha, weight_ptr, k, input_ptr, k,
                 beta, output_ptr, n);

    // Apply additional scaling if necessary
    for (int i = 0; i < output.size(); ++i) {
        output_ptr[i] *= scale;
    }
    
}

#endif 