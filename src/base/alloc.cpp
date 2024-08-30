#include "base/alloc.h"
#include <cstdlib>
#include <iostream>
using namespace base;

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

void base::DeviceAllocator::memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size, MemcpyKind memcpy_kind, void *stream, bool need_sync) const
{
    CHECK_NE(src_ptr,nullptr);
    CHECK_NE(dest_ptr,nullptr);
    if (!byte_size) {
        return;
    }

    if(memcpy_kind == base::MemcpyKind::kMemcpyCPU2CPU){
        std::memcpy(dest_ptr,src_ptr,byte_size);
    }else{

    }

}

void base::DeviceAllocator::memset_zero(void *ptr, size_t byte_size, void *stream, bool need_sync)
{
    CHECK(device_type_ != DeviceType::kDeviceUnknown);
    
    if (device_type_ == DeviceType::kDeviceCPU) {
        std::memset(ptr, 0, byte_size);
    } 
}

base::CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) 
{
}

void *base::CPUDeviceAllocator::allocate(size_t byte_size) const
{
    if(!byte_size){
        return nullptr;
    }
    void* data = malloc(byte_size);
    return data;
}

void base::CPUDeviceAllocator::release(void *ptr) const
{
    if (ptr) {
        free(ptr);
    }    
}

std::shared_ptr<CPUDeviceAllocator> base::CPUDeviceAllocatorFactory::instance_ = nullptr;



