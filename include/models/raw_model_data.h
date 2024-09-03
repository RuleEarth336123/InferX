#ifndef MODEL_H_RAWDATA_
#define MODEL_H_RAWDATA_

#include <cstddef>
#include <cstdint>

namespace model{
    struct RawModelData{
        ~RawModelData();
        int32_t fd = -1;
        size_t file_size = 0;
        void* data = nullptr;
        void* weight_data = nullptr;

        virtual const void* weight(size_t offsset) const = 0;
    };

    struct RowModelDataFp32 : RawModelData{
        const void* weight(size_t offset) const override;
    };

    struct RowModelDataInt8 : RawModelData{
        const void* weight(size_t offset) const override;
    };

}

#endif


