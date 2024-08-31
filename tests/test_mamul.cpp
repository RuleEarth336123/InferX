#include <vector>
#include <armadillo>
#include "kernel_interface.h"
#include "gtest/gtest.h"
#include "base/base.h"

TEST(MatmulKernelCPUTest, InputAndWeight) {

    float scale = 1.0f;

    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor input(base::DataType::kDataTypeFp32,2,2,true,alloc_cpu);
    ASSERT_EQ(input.is_empty(),false);
    tensor::Tensor weight(base::DataType::kDataTypeFp32,2,2,true,alloc_cpu);
    ASSERT_EQ(weight.is_empty(),false);
    tensor::Tensor output(base::DataType::kDataTypeFp32,2,2,true,alloc_cpu);
    ASSERT_EQ(output.is_empty(),false);

    for(int i=0;i<2*2;i++){
        input.index<float>(i) = 1.f;
    }

    for(int i=0;i<2*2;i++){
        weight.index<float>(i) = 2.f;
    }

    for(int i=0;i<2*2;i++){
        output.index<float>(i) = 3.f;
    }

    kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input, weight, output, 1.0f);
    
    // for(int i=0;i< 2*2;i++){
    //     std::cout << output.index<float>(i) << " ";
    // }
}

TEST(MatmulKernelCPUTest, InputAndWeight2) {

    float scale = 1.0f;

    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    //列主序，代表2列1行
    tensor::Tensor input(base::DataType::kDataTypeFp32,2,1,true,alloc_cpu);
    ASSERT_EQ(input.is_empty(),false);
    tensor::Tensor weight(base::DataType::kDataTypeFp32,3,2,true,alloc_cpu);
    ASSERT_EQ(weight.is_empty(),false);
    tensor::Tensor output(base::DataType::kDataTypeFp32,3,1,true,alloc_cpu);
    ASSERT_EQ(output.is_empty(),false);

    for(int i=0;i<2*1;i++){
        input.index<float>(i) = 1.f;
    }

    for(int i=0;i<2*3;i++){
        weight.index<float>(i) = 2.f;
    }

    for(int i=0;i<1*3;i++){
        output.index<float>(i) = 3.f;
    }

    kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input, weight, output, 1.0f);
    
    // for(int i=0;i< 1*3;i++){
    //     std::cout << output.index<float>(i) << " ";
    // }

    //(1,1)*[] = []
}