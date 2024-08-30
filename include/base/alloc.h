#ifndef BASE_H_ALLOC
#define BASE_H_ALLOC

#include "base/base.h"
#include <map>
#include <memory>

namespace base{
    class DeviceAllocator;

    enum class MemcpyKind{
        kMemcpyCPU2CPU = 0,
        kMemcpyCPU2CUDA = 1,
        kMemcpyCUDA2CPU = 2,
        kMemcpyCUDA2CUDA = 3,
    };

    class DeviceAllocator{
    public:

        explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type){}
        virtual DeviceType device_type() const {return device_type_;}
        virtual void release(void* ptr) const = 0;
        virtual void* allocate(size_t byte_size) const = 0;

        virtual void memcpy(const void* src_ptr,void* dest_ptr,size_t byte_size,\
            MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,void* stream = nullptr,
            bool need_sync = false
        ) const;

        virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

    private:
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
    };

    class CPUDeviceAllocator : public DeviceAllocator
    {
    private:
        /* data */
    public:
        explicit CPUDeviceAllocator();

        void* allocate(size_t byte_size) const override;

        void release(void* ptr) const override;
    };
    
    class CPUDeviceAllocatorFactory{
    public:
        static std::shared_ptr<CPUDeviceAllocator> get_instance(){
            if(instance_ == nullptr){
                instance_ = std::make_shared<CPUDeviceAllocator>();
            }
            return instance_;
        }
    private:
        static std::shared_ptr<CPUDeviceAllocator> instance_;
    };

}

#endif