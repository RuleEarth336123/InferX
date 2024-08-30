#include "glog/logging.h"
#include "gtest/gtest.h"
#include "kernel_interface.h"
#include "base/buffer.h"

TEST(test_scale_cpu,scale){
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
    for(int i = 0; i < size; i++){
        t1.index<float>(i) = 2.f;
    }
    kernel::get_scale_kernel(base::DeviceType::kDeviceCPU)(t1,0.5f,nullptr);
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(t1.index<float>(i), 1.f);
    }

}