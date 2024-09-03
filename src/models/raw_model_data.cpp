#include "models/raw_model_data.h"
#include <sys/mman.h>
#include <unistd.h>

model::RawModelData::~RawModelData()
{
    if(data != nullptr && data != MAP_FAILED){
        munmap(data,file_size);
        data = nullptr;
    }
    if(fd != -1){
        close(fd);
        fd = -1;
    }
}
const void *model::RowModelDataFp32::weight(size_t offset) const
{
    return static_cast<float*>(weight_data) + offset;
}

const void *model::RowModelDataInt8::weight(size_t offset) const
{
    return static_cast<int8_t*>(weight_data) + offset;
}
