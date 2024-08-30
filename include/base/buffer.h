#ifndef BASE_H_BUFFER
#define BASE_H_BUFFER

#include <memory>
#include "base/base.h"
#include "base/alloc.h"
using namespace base; 
using namespace model;

namespace base{
    class Buffer : public NoCopyable,std::enable_shared_from_this<Buffer>{


    private:
        size_t byte_size_ = 0;
        void* ptr_ = nullptr;
        //置为false表示需要对其生命周期进行管理，会自行释放
        bool use_external_ = false;
        base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
        std::shared_ptr<DeviceAllocator> allocator_;
    
    public:
        explicit Buffer() = default;

        explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                        void* ptr = nullptr, bool use_external = false);

        virtual ~Buffer();
        /**
         * 使用分配器为缓冲区分配内存空间。
         * @return 如果成功分配了内存，则返回true；
         * 否则，如果分配器不存在或请求的内存大小为0，则返回false。
         */
        bool allocate();
        /**
         * 从给定的Buffer对象中复制数据到当前Buffer对象。
         * @param buffer 要复制数据的源Buffer对象。
         * 注意：此函数会检查分配器和指针是否为空，以及设备类型是否未知。如果设备类型是CPU，则直接进行内存拷贝；如果是CUDA，则根据当前设备和目标设备的类型选择适当的内存拷贝方式。
         */
        void copy_from(const Buffer& buffer) const;

        void copy_from(const Buffer* buffer) const;

        void* ptr();

        const void* ptr() const;

        size_t byte_size() const;

        std::shared_ptr<DeviceAllocator> allocator() const;

        base::DeviceType device_type() const;

        void set_device_type(base::DeviceType device_type);

        std::shared_ptr<Buffer> get_shared_from_this();

        bool is_external() const;
    

    };

}

#endif