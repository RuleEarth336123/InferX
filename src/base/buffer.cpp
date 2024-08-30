#include "base/buffer.h"

using namespace base;

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,
               bool use_external)
    : byte_size_(byte_size),
      allocator_(allocator),
      ptr_(ptr),
      use_external_(use_external) {
    if (!ptr_ && allocator_) {
        device_type_ = allocator_->device_type();
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size);
    }
}

Buffer::~Buffer() {
    if (!use_external_) {
        if (ptr_ && allocator_) {
        allocator_->release(ptr_);
        ptr_ = nullptr;
        }
    }
}

bool base::Buffer::allocate()
{
    if(allocator_ && byte_size_ != 0){
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
        if(ptr_ == nullptr){
            return false;
        }
    }else{
        return false;
    }
    return true;
}
void base::Buffer::copy_from(const Buffer &buffer) const
{
    CHECK(allocator_ != nullptr);
    CHECK(buffer.ptr_ != nullptr);

    size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
    const DeviceType& buffer_device = buffer.device_type();
    const DeviceType current_device = this->device_type();

    CHECK(buffer_device != DeviceType::kDeviceUnknown && current_device != DeviceType::kDeviceUnknown);
    
    if(buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCPU){
        return allocator_->memcpy(buffer.ptr(),this->ptr_,byte_size);
    }else if(buffer_device == DeviceType::kDeviceCUDA && current_device == DeviceType::kDeviceCPU){
        ;
    }else if(buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCUDA){
        ;
    }else if(buffer_device == DeviceType::kDeviceCUDA && current_device == DeviceType::kDeviceCUDA){
        ;
    }
}

void base::Buffer::copy_from(const Buffer *buffer) const
{
    CHECK(allocator_ != nullptr);
    CHECK(buffer != nullptr || buffer->ptr_ != nullptr);

    size_t src_size = byte_size_;
    size_t dest_size = buffer->byte_size_;
    size_t byte_size = src_size < dest_size ? src_size : dest_size;

    const DeviceType& buffer_device = buffer->device_type();
    const DeviceType& current_device = this->device_type();
    CHECK(buffer_device != DeviceType::kDeviceUnknown && current_device != DeviceType::kDeviceUnknown);

    if (buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCPU) {
        return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size);
    } else if (buffer_device == DeviceType::kDeviceCUDA && current_device == DeviceType::kDeviceCPU) {

    } else if (buffer_device == DeviceType::kDeviceCPU && current_device == DeviceType::kDeviceCUDA) {

    } else {
    }
}
void *base::Buffer::ptr()
{
    return ptr_;
}

const void *base::Buffer::ptr() const
{
    return ptr_;
}
size_t base::Buffer::byte_size() const
{
    return byte_size_;
}

std::shared_ptr<DeviceAllocator> base::Buffer::allocator() const
{
    return allocator_;
}

DeviceType base::Buffer::device_type() const
{
    return device_type_;
}

void base::Buffer::set_device_type(DeviceType device_type)
{
    device_type_ = device_type;
}

std::shared_ptr<Buffer> base::Buffer::get_shared_from_this()
{
    return shared_from_this();
}

bool base::Buffer::is_external() const
{
    return this->use_external_;
}
