#include "models/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <cstdio>
#include <iostream>
model::Model::Model(base::ModelType model_type, std::string token_path, std::string model_path, bool is_quant_modedl)
                : model_type_(model_type),
                token_path_(std::move(token_path)),
                model_path_(std::move(model_path)),
                is_quant_model_(is_quant_modedl)
{
}

base::ModelType model::Model::model_type() const
{
    return model_type_;
}

const std::string &model::Model::token_path() const
{
    return token_path_;
}

const std::string &model::Model::model_path() const
{
    return model_path_;
}

tensor::Tensor &model::Model::get_buffer(ModelBufferType buffer_idx)
{
    CHECK_GE(buffers_.count(buffer_idx),0) << int(buffer_idx);
    return buffers_.at(buffer_idx);
}

const tensor::Tensor &model::Model::get_buffer(ModelBufferType buffer_idx) const
{
    CHECK_GE(buffers_.count(buffer_idx),0) << int(buffer_idx);
    return buffers_.at(buffer_idx);
}


base::Status model::Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor &tensor)
{
    if(buffers_.count(buffer_idx) > 0){
        return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
    }
    if(tensor.is_empty()){
        return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
    }
    buffers_.insert({buffer_idx,tensor});
    return base::error::Success();
}

base::Status model::Model::read_model_file()
{
    using namespace base;
    
    if(model_path_.empty()){
        std::cerr << "Failed to open the weight file, the model path is empty!" <<std::endl;
        return error::PathNotValid("Failed to open the weight file, the model path is empty!");
    }

    int32_t fd = open(model_path_.data(),O_RDONLY);
    if(fd < 0){
        std::cerr << "Failed to open the weight file " << model_path_ << " may be the path does not exist!" <<std::endl;
        return error::PathNotValid("Failed to open the weight file " + model_path_ + " may be the path does not exist!");
    }

    FILE* file = fopen(model_path_.data(),"rb");
    if(!file){
        std::cerr <<  "Failedto open the file. The path may be invalid." <<std::endl;
        return error::PathNotValid( "Failedto open the file. The path may be invalid.");
    }

    auto config = ModelConfig{};

    if(fread(&config,sizeof(ModelConfig),1,file) != 1){
        std::cerr << "Failed retrieve the configuration information from the model file." <<std::endl;
        return error::ModelParseError("Failed retrieve the configuration information from the model file.");
    }

    if(is_quant_model_){
        if(fread(&group_size_,sizeof(int32_t),1,file) != 1){
            std::cerr << "Failed retrieve the group size information from the model file." << std::endl;
            return error::ModelParseError("Failed retrieve the group size information from the model file.");
        }
    }

    auto gen_status = generate_model_infos(config);
    if(!gen_status){
        std::cerr << "Failed generate model infos" <<std::endl;
    }

    if(!is_quant_model_){
        raw_model_data_ = std::make_shared<RowModelDataFp32>();
    }else{
        raw_model_data_ = std::make_shared<RowModelDataInt8>();
    }

    fseek(file,0,SEEK_END);
    raw_model_data_->file_size = ftell(file);
    fclose(file);

    raw_model_data_->fd = fd;
    raw_model_data_->data = mmap(nullptr,raw_model_data_->file_size,PROT_READ,MAP_PRIVATE,raw_model_data_->fd,0);

    if(raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr){
        std::cerr << "Failed to map the weight file " << model_path_ << " info memory." <<std::endl;
        return error::ModelParseError("Failed to map the weight file " + model_path_ + " info memory.");
    }

    if(!is_quant_model_){
        raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_);
    }

    if(raw_model_data_ == nullptr){
        LOG(ERROR);
        std::cerr <<"Failed to map the weight file " << model_path_ << " into memory, the pointer to weight start address is null" <<std::endl;
        return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory, the pointer to weight start address is null");
    }
    return error::Success();
}

base::Status model::Model::create_encode_layer()
{
    using namespace base;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spe = std::make_unique<sentencepiece::SentencePieceProcessor>();
    
    const auto status = spe->Load(token_path_);
    if(!status.ok()){
        return error::PathNotValid(token_path_);
    }

    config_->vocab_size_ = spe->GetPieceSize();
    if(config_->vocab_size_ <= 0){
        std::cerr << "The vocab size param read error from the model file!" <<std::endl;
        return error::InternalError("The vocab size param read error from the model file!");
    }

    encode_layer_ = std::make_unique<op::EncodeLayer>(device_type_,true,false,std::move(spe));
    if(!encode_layer_){
        std::cerr << "Create the encode layer failed." <<std::endl;
        return error::InternalError("Create the encode layer failed.");
    }

    return error::Success();
}

base::Status model::Model::gen_model_from_file()
{
    using namespace base;
    config_ = std::make_unique<TransformerConfig>();

    auto create_encode_status = create_encode_layer();
    if(!create_encode_status){
        LOG(ERROR) << "Create encode layer failed!";
        return create_encode_status;
    }

    auto mmap_status = read_model_file();
    if(!mmap_status){
        std::cerr << "Handle model file " << model_path_ << " failed!" << std::endl;
        LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
        return mmap_status;
    }

    auto layer_create_status = create_layers();
    if(!layer_create_status){
        LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
        return layer_create_status;
    }

    return error::Success();
}

base::Status model::Model::generate_model_infos(const ModelConfig &config) const
{
    config_->dim_ = config.dim;
    config_->hidden_dim_ = config.hidden_dim;
    config_->layer_num_ = config.layer_num;
    config_->head_num_ = config.head_num;
    config_->kv_head_num_ = config.kv_head_num;
    config_->seq_len_ = config.seq_len;

    config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
    config_->kv_mul_ = config.head_num /config.kv_head_num;
    config_->head_size_ = config.dim / config.head_num;

    config_->is_shared_weight_ = config.vocab_size > 0 ? true : false; 

    if(std::abs(config.vocab_size) != config_->vocab_size_){
        std::cerr << "Vocabulary size mismatch between the model file and the token list." << std::endl;
        return base::error::ModelParseError("Vocabulary size mismatch between the model file and the token list.");
    }

    return base::error::Success();
}
