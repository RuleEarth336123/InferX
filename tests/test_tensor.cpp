#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include <iostream>
#include "base/buffer.h"

using namespace std;
using namespace base;


TEST(test_tensor, clone_cpu) {
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32,32,32,true,alloc_cpu);

    ASSERT_EQ(t1_cpu.is_empty(),false);
    for(int i=0;i<32*32;i++){
        t1_cpu.index<float>(i) = 1.f;
    }

    tensor::Tensor t2_cpu = t1_cpu.clone();
    float* p2 = new float[32 * 32];
    std::memcpy(p2,t2_cpu.ptr<float>(),sizeof(float)*32*32);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 1.f);
    }

    std::memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 1.f);
    }
    delete[] p2;

}

TEST(test_tensor, assign1) {
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32,32,32,true,alloc_cpu);
    
    float* ptr = new float[32 * 32];
    for(int i=0;i<32*32;i++){
        ptr[i] = float(i);
    }

    std::shared_ptr<Buffer> buffer =
        std::make_shared<Buffer>(32*32 * sizeof(float), nullptr, ptr, true);
    buffer->set_device_type(DeviceType::kDeviceCPU);

    ASSERT_EQ(t1_cpu.assign(buffer), true);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    ASSERT_NE(t1_cpu.ptr<float>(), nullptr);
    delete[] ptr;
}