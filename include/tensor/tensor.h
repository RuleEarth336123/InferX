#ifndef TENSOR_H_TENSOR
#define TENSOR_H_TENSOR

#include "glog/logging.h"
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"

using namespace base;

namespace tensor{
    
    class Tensor{
    public:
        explicit Tensor() = default;
        explicit Tensor(DataType data_type,int32_t dim0,
            bool need_alloc = false,std::shared_ptr<DeviceAllocator> alloc =nullptr,
            void* ptr = nullptr
        );
        explicit Tensor(DataType data_type,int32_t dim0,int32_t dim1,
            bool need_alloc = false,std::shared_ptr<DeviceAllocator> alloc =nullptr,
            void* ptr = nullptr
        );
        explicit Tensor(DataType data_type,int32_t dim0,int32_t dim1,int32_t dim2,
            bool need_alloc = false,std::shared_ptr<DeviceAllocator> alloc =nullptr,
            void* ptr = nullptr
        );
        explicit Tensor(DataType data_type,int32_t dim0,int32_t dim1,int32_t dim2, int32_t dim3,
            bool need_alloc = false,std::shared_ptr<DeviceAllocator> alloc =nullptr,
            void* ptr = nullptr
        );
        explicit Tensor(DataType data_type,std::vector<int32_t> dims,
            bool need_alloc = false,std::shared_ptr<DeviceAllocator> alloc =nullptr,
            void* ptr = nullptr
        );
        /**
         * 将Tensor对象从GPU设备转移到CPU。
         * @param 无参数。
         * @return 无返回值。
         * @note 如果Tensor的设备类型未知，则记录错误日志；
         * 如果Tensor的设备类型为CUDA，则将其内容复制到新的CPU缓冲区中；
         * 如果Tensor的设备类型已经是CPU，则记录信息日志。
         */
        void to_cpu();
        /**
         * 将Tensor对象的数据转移到CUDA设备上。
         * @param stream 用于数据传输的cudaStream_t流。
         * 如果Tensor的设备类型是未知，则记录错误日志。
         * 如果Tensor的设备类型是CPU，则创建一个新的CUDA缓冲区，并将原始数据复制到新的CUDA缓冲区中。
         * 如果Tensor的设备类型已经是CPU，则记录信息日志。
         */
        void to_cuda();

        bool is_empty() const;
        void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,DataType data_type,\
            bool need_alloc,void* ptr);

        template <typename T>
        T* ptr();

        template <typename T>
        const T* ptr() const;     

        void reshape(const std::vector<int32_t>& dims);
        std::shared_ptr<Buffer> get_buffer() const;
        size_t size() const;
        size_t byte_size() const;
        int32_t dims_size() const;
        base::DataType data_type() const;
        int32_t get_dim(int32_t idx) const;
        const std::vector<int32_t>& dims() const;
        /**
         * 计算并返回张量各维度的步
         * @return 一个包含各维度步长的向量，如果张量的维度为空，则返回空向量。
         */
        std::vector<size_t> strides() const;

        bool assign(std::shared_ptr<base::Buffer> buffer);
        void reset(base::DataType data_type, const std::vector<int32_t>& dims);
        void set_device_type(base::DeviceType device_type) const;
        base::DeviceType device_type() const;
        bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

        template <typename T>
        T* ptr(int64_t index);

        template <typename T>
        const T* ptr(int64_t index) const;

        template <typename T>
        T& index(int64_t offset);

        template <typename T>
        const T& index(int64_t offset) const;

        tensor::Tensor clone() const;

        int fill(float set_num) const;


    private:


    private:
        size_t size_ = 0;
        std::vector<int32_t> dims_;
        std::shared_ptr<base::Buffer> buffer_;
        base::DataType data_type_ = base::DataType::kDataTypeUnknown;
    };

    template <typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init);
    static size_t data_type_size(base::DataType data_type);
/**
* 获取Tensor对象的指针。
* @return 如果buffer_存在，返回指向buffer_的指针；否则返回nullptr。
*/
    template <typename T>
    const T* Tensor::ptr() const {
    if (!buffer_) {
        return nullptr;
    }
    return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
    }
/**
    * 获取Tensor对象的指针。
    * @return 如果buffer_存在，返回指向buffer_的指针；否则返回nullptr。
    */
    template <typename T>
    T* Tensor::ptr() {
    if (!buffer_) {
        return nullptr;
    }
        return reinterpret_cast<T*>(buffer_->ptr());
    }
    /**
    * 返回指定索引处的Tensor数据指针。
    * @param index 指定的索引值。
    * @return 返回指向指定索引处数据的指针。如果数据区域缓冲区为空或指向空指针，则抛出异常。
    */
    template <typename T>
    T* Tensor::ptr(int64_t index) {
    CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
        << "The data area buffer of this tensor is empty or it points to a null pointer.";
    return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
    }
    /**
    * 返回指定索引处的Tensor数据指针。
    * @param index 指定的索引值。
    * @return 返回指向指定索引处数据的指针。如果数据区域缓冲区为空或指向空指针，则抛出异常。
    */
    template <typename T>
    const T* Tensor::ptr(int64_t index) const {
        CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
            << "The data area buffer of this tensor is empty or it points to a null pointer.";
        return reinterpret_cast<const T*>(buffer_->ptr()) + index;
    }
    /**
    * 使用给定的偏移量获取张量的值。
    * @param offset 要获取的值的偏移量，必须大于等于0且小于张量的大小。
    * @return 返回在给定偏移量处的张量值的引用。
    */    
    template <typename T>
    inline T &Tensor::index(int64_t offset)
    {
        CHECK_GE(offset, 0);
        CHECK_LT(offset, this->size());
        T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
        return val;
    }
    /**
    * 获取Tensor中指定偏移量的元素。
    * @param offset 要获取元素的偏移量，必须大于等于0且小于Tensor的大小。
    * @return 返回Tensor中指定偏移量的元素。
    */
    template <typename T>
    inline const T &Tensor::index(int64_t offset) const
    {
        CHECK_GE(offset, 0);
        CHECK_LT(offset, this->size());
        const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
        return val;
    }
    /**
    * 使用给定的数值填充张量的所有元素。
    * @param set_num 用于填充张量的数值。
    * @return 返回0表示函数执行成功。
    */
    inline int Tensor::fill(float set_num) const
    {
        // if (this->size() <= 0) {
        //     std::cerr << "Error: Trying to fill an empty tensor." << std::endl;
        //     return -1;
        // }
        // for(int i=0;i<this->size();i++){
        //     if (!this->index<float>(i)) {
        //         std::cerr << "Error: Failed to set element at index " << i << "." << std::endl;
        //         return -1;
        //     }
        //     this->index<float>(i) = set_num;
        // }
        return 0;
    }
}

#endif