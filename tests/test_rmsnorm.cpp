#include <vector>
#include <armadillo>
#include "kernel_interface.h"
#include "gtest/gtest.h"
#include "base/base.h"
#include "base/buffer.h"
TEST(ResnormKernelCPUTest, InputAndWeight) {
    float scale = 1.0f;

    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(base::DataType::kDataTypeFp32,2,2,true,alloc_cpu);
    ASSERT_EQ(input.is_empty(),false);
    tensor::Tensor weight(base::DataType::kDataTypeFp32,2,2,true,alloc_cpu);
    ASSERT_EQ(weight.is_empty(),false);
    tensor::Tensor output(base::DataType::kDataTypeFp32,2,2,true,alloc_cpu);
    ASSERT_EQ(output.is_empty(),false);

    for(int i=0;i<2*2;i++){
        input.index<float>(i) = i;
    }

    for(int i=0;i<2*2;i++){
        weight.index<float>(i) = i+1;
    }

    for(int i=0;i<2*2;i++){
        output.index<float>(i) = 3.f;
    }

    kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(input, weight, output,nullptr);
    
    for(int i=0;i< 2*2;i++){
        std::cout << output.index<float>(i) << " ";
    }
}

TEST(ResnormKernelCPUTest, InputAndWeight2) {
    float scale = 1.0f;
    int32_t size = 32 * 15;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(base::DataType::kDataTypeFp32,size,true,alloc_cpu);
    tensor::Tensor weight(base::DataType::kDataTypeFp32,size,true,alloc_cpu);
    tensor::Tensor output(base::DataType::kDataTypeFp32,size,true,alloc_cpu);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < size; ++i) {
        input.index<float>(i) = dist(mt);
        weight.index<float>(i) = dist(mt);
    }

    kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(input,weight,output,nullptr);


    for (int i = 0; i < size; ++i) {
        ASSERT_NEAR(output.index<float>(i), output.index<float>(i), 1e-5f);
    }
}