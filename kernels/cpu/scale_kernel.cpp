#include "cpu/scale_kernel.h"
#include <armadillo>
#include "base/base.h"
#include "cblas.h"
void kernel::scale_inplace_cpu(const tensor::Tensor &tensor, float scale,void *stream)
{
    CHECK(tensor.is_empty() == false);
#if 0    
    arma::fvec tensor_mat(const_cast<float*>(tensor.ptr<float>()),tensor.size(),false,
                            true);
    tensor_mat = tensor_mat * scale;
#else
    float* data = const_cast<float*>(tensor.ptr<float>());
    int size = static_cast<int32_t>(tensor.size());
    cblas_sscal(size, scale, data, 1);
#endif
}
void kernel::scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale, 
                          const tensor::Tensor& output, int pos, int size, int stride,
                          void* stream)
{
    CHECK_EQ(value.is_empty(), false);
    CHECK_EQ(scale.is_empty(), false); 
    CHECK_EQ(output.is_empty(), false);
    CHECK_EQ(size, value.size());
    CHECK_GE(size, scale.size());
    CHECK_EQ(size, output.size());

    float* value_data = const_cast<float*>(value.ptr<float>());
    const float* scale_data = const_cast<float*>(scale.ptr<float>());
    float* output_data = const_cast<float*>(output.ptr<float>());

    std::cout<<"stride: "<<stride<<std::endl;
    std::cout<<"pos: "<<pos<<std::endl;   
    //遍历尺度向量
    for(int i=0;i<=pos;i++){
        //每个尺度对应的向量片段
        float* value_segment = value_data + i*stride;
        float scale_factor = scale_data[i];
        //累加
        cblas_saxpy(size,scale_factor,value_segment,1,output_data,1);
    }
}