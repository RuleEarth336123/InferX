#include "glog/logging.h"
#include "gtest/gtest.h"
#include "kernel_interface.h"
#include "base/buffer.h"
#include <random>

TEST(test_swiglu,swiglu_cpu){
    using namespace tensor;
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    int32_t size = 32 * 151;

    tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
    tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < size; ++i) {
        in_cpu.index<float>(i) = dist(mt);
        wei_cpu.index<float>(i) = dist(mt);
    }

    kernel::get_swiglu_kernel(base::DeviceType::kDeviceCPU)(in_cpu,wei_cpu,out_cpu,nullptr);
    // for (int i = 0; i < size; ++i) {
    //     ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
    // }
}