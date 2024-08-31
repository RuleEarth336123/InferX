#include "glog/logging.h"
#include "gtest/gtest.h"
#include "kernel_interface.h"
#include "base/buffer.h"

TEST(test_softmax,softmax_cpu){
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 21 * 151;

    srand(0);

    tensor::Tensor in_cpu(base::DataType::kDataTypeFp32,size,true,alloc_cpu);
    for(int i = 0;i < size;i++){
        in_cpu.index<float>(i) = rand() % 31;
    }
    kernel::get_softmax_inplace_kernel(base::DeviceType::kDeviceCPU)(in_cpu,nullptr);

    // for(int i=0;i<size;i++){
    //     ASSERT_NEAR(in_cpu.index<float>(i),in_cpu.index<float>(i),1e-5f);
    // }

}