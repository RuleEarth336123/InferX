#include "cpu/rmsnorm_kernel.h"
#include <armadillo>
#include "base/base.h"

#if 0
void kernel::rmsnorm_kernel_cpu(const tensor::Tensor &input, \
    const tensor::Tensor &weight, const tensor::Tensor &output, void *stream)
{
    CHECK(input.is_empty() == false);
    CHECK(weight.is_empty() == false);
    CHECK(output.is_empty() == false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    const float* in_ptr = input.ptr<float>();
    const float* wei_ptr = weight.ptr<float>();
    const float* out_ptr = output.ptr<float>();
    const int32_t dim = static_cast<int32_t>(input.size());

    arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
    arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
    arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

    const float eps = 1e-5f;
    const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor,2))) + eps;
    const float rsqrt = 1.f / std::sqrt(mean);
    out_tensor = wei_tensor % (rsqrt * in_tensor);
}

#else 
#include <cblas.h>
void kernel::rmsnorm_kernel_cpu(const tensor::Tensor &input, \
    const tensor::Tensor &weight, const tensor::Tensor &output, void *stream)
{
    CHECK(input.is_empty() == false);
    CHECK(weight.is_empty() == false);
    CHECK(output.is_empty() == false);
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    float* in_ptr = const_cast<float*>(input.ptr<float>());
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    int32_t dim = static_cast<int32_t>(input.size());

    //1. Compute the square of the input vector
    cblas_sscal(dim, 0.0f, in_ptr, 1); // Initialize to zero if needed
    cblas_saxpy(dim, 1.0f, in_ptr, 1, in_ptr, 1); // in_ptr[i] = in_ptr[i] * in_ptr[i]
    for (int32_t i = 0; i < dim; ++i) {
        in_ptr[i] = in_ptr[i] * in_ptr[i];
    }

    // 2. Compute the mean of the squared vector
    float mean = 0.0f;
    cblas_saxpy(dim, 1.0f, in_ptr, 1, &mean, 1); // mean += in_ptr[i]
    mean = mean / dim;

    // 3. Compute the inverse square root of the mean
    const float eps = 1e-5f;
    float rsqrt = 1.0f / std::sqrt(mean + eps);
 
    //4. Scale the weight vector by the computed factor
    cblas_sscal(dim, rsqrt, wei_ptr, 1); // wei_ptr[i] *= rsqrt
    cblas_saxpy(dim, 1.0f, in_ptr, 1, wei_ptr, 1); // wei_ptr[i] += rsqrt * in_ptr[i]
    for (int32_t i = 0; i < dim; ++i) {
        out_ptr[i] = wei_ptr[i] * rsqrt * in_ptr[i];
    }
}
#endif