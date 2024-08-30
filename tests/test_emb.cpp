#include "glog/logging.h"
#include "gtest/gtest.h"
#include "base/buffer.h"
#include "kernel_interface.h"

TEST(test_emb, emb1_nostream){
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t token = 4,dim = 512, size = 2048;

    tensor::Tensor input(base::DataType::kDataTypeFp32, 1, true, alloc_cpu);
    input.index<int32_t>(0) = 1;

    tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
    tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);

    for (int i = 0; i < size; ++i) {
        weight.index<float>(i) = static_cast<float>(i);
    }
    //weight.to_cpu();

    kernel::get_emb_kernel(base::DeviceType::kDeviceCPU)(input, weight, output, token,
                                                        nullptr);

    //output.to_cpu();
    for (int i = 0; i < dim; ++i) {
        ASSERT_EQ(output.index<float>(i), 512 + i);
    }
}