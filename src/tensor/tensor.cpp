#include "tensor/tensor.h"
#include <iostream>
#include <algorithm>
#include <numeric>
/**
 * 构造一个Tensor对象。
 *
 * @param data_type 数据类型，用于指定Tensor中存储的数据类型。
 * @param dim0 维度大小，用于指定Tensor在第一个维度上的大小。
 * @param need_alloc 是否需要分配内存，如果为true，则需要调用allocate函数进行内存分配；否则，根据ptr参数决定是否初始化缓冲区。
 * @param alloc 设备分配器，用于分配Tensor的内存空间。
 * @param ptr 指向已分配内存的指针，如果need_alloc为false且ptr不为nullptr，则使用ptr初始化Tensor的缓冲区。
 * @throws std::runtime_error 如果need_alloc为true但ptr为nullptr，将抛出异常。
 */
tensor::Tensor::Tensor(DataType data_type, int32_t dim0, bool need_alloc, 
                        std::shared_ptr<DeviceAllocator> alloc, void *ptr)
                        : data_type_(data_type)
{
    dims_.push_back(dim0);
    size_ = dim0;
    if(need_alloc && alloc){
        allocate(alloc);
    }else{
        if(ptr != nullptr){
            CHECK(need_alloc == false)
                << "The need_alloc is is true when ptr parameter is not a null pointer.";
            init_buffer(alloc,data_type,need_alloc,ptr);
        }
    }
}
/**
 * 构造一个Tensor对象。
 *
 * @param data_type 数据类型，用于指定Tensor中存储的数据类型。
 * @param dim0 第一个维度的大小。
 * @param dim1 第二个维度的大小。
 * @param need_alloc 是否需要分配内存空间。如果为true，则在构造函数中进行内存分配；否则，使用提供的ptr参数进行初始化。
 * @param alloc 设备分配器，用于分配Tensor的内存空间。如果need_alloc为true，此参数有效。
 * @param ptr 指向已分配内存的指针，用于初始化Tensor。如果need_alloc为false，此参数有效。
 */
tensor::Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc, 
                        std::shared_ptr<DeviceAllocator> alloc, void *ptr)
                        : data_type_(data_type)
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0 * dim1;
    if(need_alloc && alloc){
        allocate(alloc);
    }else{
        init_buffer(alloc,data_type_,need_alloc,ptr);
    }
}
/**
 * 构造一个Tensor对象。
 *
 * @param data_type 数据类型，用于指定Tensor中存储的数据类型。
 * @param dim0 第一个维度的大小。
 * @param dim1 第二个维度的大小。
 * @param dim2 第三个维度的大小。
 * @param need_alloc 是否需要分配内存空间。如果为true，则在构造函数中进行内存分配；否则，使用提供的ptr参数作为已分配的内存空间。
 * @param alloc 设备分配器，用于分配和释放Tensor使用的内存空间。如果need_alloc为true，此参数将被忽略。
 * @param ptr 指向已分配内存的指针。如果need_alloc为false，此参数应指向有效的内存区域。
 */
tensor::Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc, 
                        std::shared_ptr<DeviceAllocator> alloc, void *ptr)
                        : data_type_(data_type)
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = dim0 * dim1 * dim2;
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}
/**
 * 构造一个Tensor对象。
 *
 * @param data_type 数据类型，用于指定Tensor中存储的数据类型。
 * @param dims 维度向量，用于指定Tensor的维度。
 * @param need_alloc 是否需要分配内存。如果为true，则需要分配内存；否则，不需要。
 * @param alloc 设备分配器，用于在特定设备上分配内存。
 * @param ptr 指向已分配内存的指针。如果need_alloc为false，则此参数有效。
 */
tensor::Tensor::Tensor(DataType data_type, std::vector<int32_t> dims, bool need_alloc, 
                        std::shared_ptr<DeviceAllocator> alloc, void *ptr)
                        : dims_(std::move(dims)), data_type_(data_type)
{
    size_ = reduce_dimension(dims_.begin(),dims_.end(),1);
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}
/**
 * 构造一个Tensor对象。
 *
 * @param data_type 数据类型，用于指定Tensor中存储的数据类型。
 * @param dim0 第一个维度的大小。
 * @param dim1 第二个维度的大小。
 * @param dim2 第三个维度的大小。
 * @param dim3 第四个维度的大小。
 * @param need_alloc 是否需要分配内存空间。如果为true，则在构造函数中进行内存分配；否则，使用提供的ptr参数进行初始化。
 * @param alloc 设备分配器，用于分配Tensor的内存空间。如果need_alloc为false，此参数将被忽略。
 * @param ptr 指向已分配内存的指针。如果need_alloc为false，此参数应指向有效的内存区域。
 */
tensor::Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3, \
            bool need_alloc, std::shared_ptr<DeviceAllocator> alloc, void *ptr)
            : data_type_(data_type)
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    dims_.push_back(dim3);
    size_ = dim0 * dim1 * dim2 * dim3;
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}
/**
 * 将tensor从GPU设备转移到CPU。
 * @param 无参数。
 * @return 无返回值。
 * @note 如果tensor的设备类型未知，则记录错误日志；如果tensor的设备类型为CUDA，则将其内容复制到CPU分配的内存中；如果tensor的设备类型已经是CPU，则记录信息日志。
 */
void tensor::Tensor::to_cpu()
{
    CHECK_NE(buffer_,nullptr);
    const DeviceType device_type = this->device_type();
    if(device_type == DeviceType::kDeviceUnknown){
        LOG(ERROR) << "The device type of the tensor is unknows.";
    }else if(device_type == DeviceType::kDeviceCUDA){
        size_t byte_size = this->byte_size();
        auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
        auto cpu_buffer = std::make_shared<base::Buffer>(byte_size,cpu_alloc);
        cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                      base::MemcpyKind::kMemcpyCUDA2CPU);
        this->buffer_ = cpu_buffer;
    }else{
        LOG(INFO) << "The device type of the tensor is already cuda.";
    }
}

void tensor::Tensor::to_cuda()
{
}
/**
 * 检查当前张量是否为空。
 * @return 如果张量的大小为0，或者其缓冲区为nullptr，或者其缓冲区的指针为nullptr，则返回true，否则返回false。
 */
bool tensor::Tensor::is_empty() const
{
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}
/**
 * 初始化tensor的缓冲区。如果不需要分配内存，则直接创建一个基础缓冲区；否则，使用指定的设备分配器进行内存分配。
 *
 * @param alloc 用于分配内存的设备分配器。如果为nullptr且need_alloc为false，则不进行内存分配。
 * @param data_type 数据类型，用于确定缓冲区的大小。
 * @param need_alloc 是否需要分配内存。如果为true，则需要分配内存；否则，根据alloc的值决定是否分配内存。
 * @param ptr 指向已分配内存的指针。如果需要分配内存，该参数将被忽略。
 */
void tensor::Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, DataType data_type, bool need_alloc, void *ptr)
{
    if (!alloc && !need_alloc) {
        std::shared_ptr<base::Buffer> buffer =
            std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
        this->buffer_ = buffer;
    } else {
        allocate(alloc, true);
    }
}
/**
 * 使用给定的维度重新调整张量的形状。
 * @param dims 一个包含新维度的向量，用于重新定义张量的尺寸。
 */
void tensor::Tensor::reshape(const std::vector<int32_t> &dims)
{
    size_t size = reduce_dimension(dims.begin(),dims.end(),1);
    
    if(!buffer_){
        this->dims_ = dims;
        this->size_ = size;
        return;
    }

    if(size > size_){
        auto new_buffer = std::make_shared<Buffer>(
            size * base::DataTypeSize(this->data_type_),
            buffer_->allocator()
        );
        CHECK(new_buffer->allocate());
        new_buffer->copy_from(buffer_.get());
        this->buffer_ = new_buffer;
    }

    this->dims_ = dims;
    this->size_ = size;
}
/**
 * 获取Tensor对象的大小。
 * @return size_ 返回Tensor对象的大小。
 */
size_t tensor::Tensor::size() const
{
    return this->size_;
}
/**
 * 获取Tensor对象的buffer。
 * @return 返回一个指向Buffer的智能指针，该Buffer是Tensor对象的数据存储区。
 */
std::shared_ptr<Buffer> tensor::Tensor::get_buffer() const
{
    return buffer_;
}
/**
 * 计算并返回张量所占用的字节大小。
 * @return size_t 返回张量所占用的字节大小，该大小由张量的大小和数据类型大小决定。
 */
size_t tensor::Tensor::byte_size() const
{
     return this->size() * DataTypeSize(data_type_);
}
/**
 * 获取张量的维度大小。
 * @return 返回张量的维度数量，以int32_t类型表示。
 */
int32_t tensor::Tensor::dims_size() const
{
    return static_cast<int32_t>(dims_.size());
}
/**
 * 获取Tensor的数据类型。
 * @return data_type_ 返回Tensor的数据类型。
 */
base::DataType tensor::Tensor::data_type() const
{
    return data_type_;
}
/**
 * 获取指定维度的大小。
 * @param idx 要获取的维度的索引。
 * @return 返回指定维度的大小。
 */
int32_t tensor::Tensor::get_dim(int32_t idx) const
{
    CHECK_GE(idx,0);
    CHECK_LT(idx,this->dims_.size());
    return this->dims_.at(idx);
}
/**
 * 获取Tensor对象的维度信息。
 * @return 返回一个包含所有维度信息的vector，每个元素代表一个维度的大小。
 */
const std::vector<int32_t> &tensor::Tensor::dims() const
{
    return this->dims_;
}
/**
 * 计算并返回张量在每个维度上的步长。
 * @return 一个包含每个维度步长的向量，如果张量的维度为空，则返回的向量也为空。
 */
std::vector<size_t> tensor::Tensor::strides() const
{
    std::vector<size_t> strides;
    if(!dims_.empty()){
        for(auto i=0;i<dims_size();i++){
            strides.push_back(reduce_dimension(dims_.begin()+i+1,dims_.end(),1));
        }
        strides.push_back(1);
    }
    return strides;

}
/**
 * 使用给定的缓冲区为张量分配内存。
 *
 * @param buffer 用于分配内存的缓冲区，其设备类型应与当前张量相同。
 * 如果缓冲区为空指针或其大小小于张量的大小，则函数将返回false并记录错误。
 * 否则，函数将成功分配内存并返回true。
 *
 * @return 如果成功分配内存，则返回true；否则返回false。
 */
bool tensor::Tensor::assign(std::shared_ptr<base::Buffer> buffer)
{
    if(!buffer){
        LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
        return false;
    }
    if(buffer_){
        if(buffer_->device_type() != buffer->device_type()){
            LOG(ERROR) << "The device type of the new buffer is different from the original one.";
        }
    }
    size_t byte_size = this->byte_size();
    if (byte_size > buffer->byte_size()) {
        LOG(ERROR) << "The size of buffer is too small for the tensor!";
        return false;
    }
    buffer_ = buffer;
    return true;
}
/**
 * 重置Tensor对象的数据类型和维度，并计算其大小。
 *
 * @param data_type 要设置的新的Tensor数据类型。
 * @param dims 要设置的新的Tensor维度，以std::vector<int32_t>的形式给出。
 */
void tensor::Tensor::reset(base::DataType data_type, const std::vector<int32_t> &dims)
{
    this->data_type_ = data_type;
    this->dims_ = dims;
    this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
    this->buffer_ = nullptr;
}
/**
 * 设置Tensor的设备类型。
 *
 * @param device_type 要设置的设备类型。
 */
void tensor::Tensor::set_device_type(base::DeviceType device_type) const
{
    if (buffer_) {
        buffer_->set_device_type(device_type);
    }
}
/**
 * 获取Tensor的设备类型。
 *
 * @return 如果buffer_为空，则返回设备类型为kDeviceUnknown；否则返回buffer_的设备类型。
 */
base::DeviceType tensor::Tensor::device_type() const
{
    if (!buffer_) {
        return base::DeviceType::kDeviceUnknown;
    }
    return buffer_->device_type();
}
/**
 * 使用给定的分配器为Tensor对象分配内存。
 *
 * @param allocator 用于分配内存的设备分配器。如果此参数为null，函数将返回错误并退出。
 * @param need_realloc 如果为true，当buffer_存在且byte_size小于等于buffer_的byte_size时，会重新分配内存。否则，直接返回true。
 * @return 如果成功分配内存，则返回true；否则，返回false。
 */
bool tensor::Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc)
{
    if(!allocator){
        LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!";
        return false;
    }

    size_t byte_size = this->byte_size();
    if(!byte_size){
        LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
        return false; 
    }
    //如果buffer_已经存在，并且byte_size小于等于buffer_的byte_size，则不需要重新分配内存
    if (buffer_ && byte_size <= buffer_->byte_size()) {
        if (!need_realloc) {
            return true;
        }
    }
    buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
    if (!buffer_->ptr()) {
        LOG(ERROR) << "The memory allocated is a null pointer!";
        return false;
    }
    return true;  
}
/**
 * 克隆当前张量，并返回新的张量。
 *
 * @return 返回一个新的张量，其内容与当前张量相同。
 */
tensor::Tensor tensor::Tensor::clone() const
{
    Tensor new_tensor = *this;
    size_t byte_size = this->byte_size();

    auto allocator = buffer_->allocator();
    new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
    new_tensor.buffer_->copy_from(buffer_.get());
    return new_tensor;
}
/**
 * 使用给定的初始值，对指定范围内的元素进行累乘操作，并返回结果。
 *
 * @param begin 指向要进行累乘操作的元素范围的起始迭代器。
 * @param end   指向要进行累乘操作的元素范围的结束迭代器。
 * @param init  用于累乘操作的初始值。
 * @return      返回经过累乘操作后的结果。如果指定的范围为空（即begin >= end），则返回0。
 */
template <typename T, typename Tp>
size_t tensor::reduce_dimension(T begin, T end, Tp init)
{
    if (begin >= end) {
        return 0;
    }
    //累乘(init为累乘的初始值)的乘积
    size_t size = std::accumulate(begin, end, init, std::multiplies<>());
    return size;
}
/**
 * 获取给定数据类型的大小。
 *
 * @param data_type 需要获取大小的base::DataType枚举类型。
 * @return 返回对应数据类型的大小，单位为字节。如果传入未知的数据类型，则打印错误日志并返回0。
 */
size_t tensor::data_type_size(base::DataType data_type)
{
    switch (data_type) {
        case base::DataType::kDataTypeFp32: {
            return 4;
        }
        case base::DataType::kDataTypeInt8: {
            return 1;
        }
        case base::DataType::kDataTypeInt32: {
            return 4;
        }
        default: {
            LOG(FATAL) << "Unknown data type size for " << int(data_type);
            return 0;
        }
    }
}
