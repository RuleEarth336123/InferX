#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include <iostream>
#include "base/buffer.h"
#include "kernel_interface.h"
#include "op/add.h"

using namespace std;
using namespace base;
using namespace kernel;

TEST(test_add, add_align1) {
    
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 8 * 8;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  float j = 0.f;
  for(int i=0;i<size;i++){
        t1.index<float>(i) = j++;
  }

  for(int i=0;i<size;i++){
      t2.index<float>(i) = j--;
  }

  kernel::get_add_kernel(base::DeviceType::kDeviceCPU)(t1, t2, out, nullptr);

  for(int i=0;i<size;i++){
    std::cout << out.index<float>(i) << " ";
  }

}

