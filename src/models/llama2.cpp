#include "models/llama2.h"
#include <iostream>

model::Llama2Model::Llama2Model(string token_path, string model_path, bool is_quant_model)
    : Model(base::ModelType::kModelTypeLLama2,std::move(token_path),std::move(model_path),is_quant_model)
{
}

base::Status model::Llama2Model::init(base::DeviceType device_type)
{
    using namespace base;
    if(token_path_.empty()){
        std::cerr << "token's path : " << token_path_ << " not valid."<<std::endl;
    }

    device_type_ = device_type;

    Status status = gen_model_from_file();
    if(!status){
        std::cerr << "generate model from the file failed." <<std::endl;
        return status;
    }
    
    init_mem();

    sampler_ = std::make_unique<ArgmaxSampler>(device_type_);

    return error::Success();
}

base::Status model::Llama2Model::predict(const tensor::Tensor &init, const tensor::Tensor &pos_tensor, bool is_prompt, int &next) const
{
    return base::Status();
}

base::Status model::Llama2Model::forward(const tensor::Tensor &input, const tensor::Tensor &pos_tensor, int next) const
{
    return base::Status();
}

std::vector<int32_t> model::Llama2Model::encode(const string &sentence) const
{
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->encode(sentence);
}

std::string model::Llama2Model::decode(int32_t token_idx) const
{
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->decode(token_idx);
}

int32_t model::Llama2Model::get_eos() const
{
    CHECK(this->encode_layer_ != nullptr);
    return encode_layer_->eos();
}

std::pair<tensor::Tensor, tensor::Tensor> model::Llama2Model::slice_kv_cache(int32_t layer_idx, int32_t token_pos) const
{

    int32_t layre_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
    int32_t cache_offset = layre_offset + token_pos * config_->kv_dim_;

    float* key_cache_ptr = const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
    float* val_cache_ptr = const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

    auto key_cache = std::make_shared<base::Buffer>(config_->kv_dim_ * sizeof(float), nullptr,key_cache_ptr,true);
    auto val_cache = std::make_shared<base::Buffer>(config_->kv_dim_ * sizeof(float), nullptr,val_cache_ptr,true);

    key_cache->set_device_type(device_type_);
    val_cache->set_device_type(device_type_);

    tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_);
    tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_);

    key.assign(key_cache);
    val.assign(val_cache);

    return {key,val};
}

op::embedingOutput model::Llama2Model::embedding(const std::vector<int> &tokens) const
{
    auto input_tokens = get_buffer(ModelBufferType::kInputTokens);




    return op::embedingOutput();
}

tensor::Tensor model::Llama2Model::fill_output(const tensor::Tensor &pos_tensor, const op::embedingOutput &, bool is_prompt) const
{
    return tensor::Tensor();
}

void model::Llama2Model::init_mem()
{
    using namespace tensor;
    std::shared_ptr<base::DeviceAllocator> alloc;
    if(device_type_ = base::DeviceType::kDeviceCPU){
        alloc = base::CPUDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);

    CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
    CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

    Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
    CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
    CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
    CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

    Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
    Tensor w3_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);    CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
    CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

    Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,config_->kv_dim_, true, alloc);
    Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,config_->kv_dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
    CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

    Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kQuery, query));

    tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));
    
    // Attention output
    tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,alloc);
    CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
    CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

    tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status model::Llama2Model::create_layers()
{
    using namespace base;
    if(!llama_layers_){
        llama_layers_ = std::make_unique<Llama2layers>();
    }

    if(!is_quant_model_){
        create_param_layers();
    }else{
        create_param_quant_layers();
    }

    create_nonparam_layers();

    if(!llama_layers_->embeding_layers_){
        std::cerr << "create the embedding layer for the llama model failed!" <<std::endl;
        return error::InternalError("create the embedding layer for the llama model failed!");
    }

    if(!llama_layers_->resnorm_layers_.size() != 2 * config_->layer_num_ = 1){
        std::cerr << "create the rmsnorm layers for the llama model failed!" << std::endl;
        return error::InternalError("create the rmsnorm layers for the llama model failed!" );
    }

    if(llama_layers_->wq_layers_.size() != config_->layer_num_ ||
        llama_layers_->wk_layers_.size() != config_->layer_num_ ||
        llama_layers_->wv_layers_.size() != config_->layer_num_ ||
        llama_layers_->wo_layers_.size() != config_->layer_num_ ){
        std::cerr << "Create the matmul layer in the attention and ffn attention layers failed! " << std::endl;
        return error::InternalError("Create the matmul layer in the attention and ffn attention layers failed! ");
    }

    for(int32_t i = 0;i < config_->layer_num_; i++){
        if(!llama_layers_->wq_layers_.at(i) ||
        !llama_layers_->wq_layers_.at(i) ||
        !llama_layers_->wq_layers_.at(i) ||
        !llama_layers_->wq_layers_.at(i) ){
            std::cerr << "Create the matmul layer in the attention and ffn attention layers failed! " << std::endl;
            return error::InternalError("Create the matmul layer in the attention and ffn attention layers failed! ");
        }
    }

    if(llama_layers_->w1_layers_.size() != config_->layer_num_ ||
    llama_layers_->w2_layers_.size() != config_->layer_num_ ||
    llama_layers_->w3_layers_.size() != config_->layer_num_ ){
        std::cerr << "Create the matmul layer in the feedforward layers for the llama model failed! " << std::endl;
        return error::InternalError("Create the matmul layer in the feedforward layers for the llama model failed! ");

    }

    for(int32_t i = 0;i < config_->layer_num_; i++){
        if(!llama_layers_->w1_layers_.at(i) ||
        !llama_layers_->w2_layers_.at(i) ||
        !llama_layers_->w3_layers_.at(i)){
            std::cerr << "Create the matmul layer in the feedforward layers for the llama model failed! " << std::endl;
            return error::InternalError("Create the matmul layer in the feedforward layers for the llama model failed! ");
        }
    }

    if(!llama_layers_->rope_layer_){
        std::cerr << "Create the rope layer for the llama model failed! " << std::endl;
        return error::InternalError("Create the rope layer for the llama model failed! ");
    }

    if(!llama_layers_->add_layer_){
        std::cerr << "Create the add layer for the llama model failed!" << std::endl;
        return error::InternalError("Create the add layer for the llama model failed!");
    }

    if(!llama_layers_->mha_layer_){
        std::cerr << "Create the mha layer for the llama model failed!" << std::endl;
        return error::InternalError("Create the mha layer for the llama model failed!");
    }

    if(!llama_layers_->swiglu_layer_){
        std::cerr << "Create the swiglu layer for the llama model failed!" << std::endl;
        return error::InternalError("Create the swiglu layer for the llama model failed!");
    }

    return error::Success();
}

void model::Llama2Model::create_param_layers()
{
    CHECK(!is_quant_model_);
    CHECK(llama_layers_ != nullptr);

    //创建embeding层(dim_ * seq * vocab)
    llama_layers_->embeding_layers_ = std::make_shared<op::EmbeddingLayer>(
        base::DeviceType::kDeviceCPU,config_->dim_,  config_->seq_len_,std::abs(config_->vocab_size_)
    );
    //设置embeding层权重参数
    const void* weight_embedding = raw_model_data_->weight(0);
    llama_layers_->embeding_layer_->set_weight(
        0, {std::abs(config_->vocab_size_), config_->dim_},weight_embedding,base::DeviceType::kDeviceCPU
    );

    // 初始化matmul层，pos用于跟踪权重数据的位置
    size_t pos = config_->dim_ * std::abs(config_->vocab_size_) +  config_->dim_ * config_->layer_num_;
    
    //创建查询矩阵(dim * dim)
    for(int32_t i = 0;i < config_->layer_num_; i++){
        auto wq = std::make_shared<op::MatmulLayer>(
            base::DeviceType::kDeviceCPU,config_->dim_,config_->dim_
        );
        wq->set_weight(
            0, {config_->dim_, config_->dim_}, this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU
        );
        llama_layers_->wq_layers_.push_back(wq);
        pos += config_->dim_ * config_->dim_;
    }

    //创建键矩阵(kx_dim * dim)
    for(int32_t i = 0;i < config_->layer_num_; i++){
        auto wk = std::make_shared<op::MatmulLayer>(
            base::DeviceType::kDeviceCPU,config_->kv_dim_,config_->dim_
        );
        wk->set_weight(
            0, {config_->kv_dim_, config_->dim_}, this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU
        );
        llama_layers_->wk_layers_.push_back(wk);
        pos += config_->kv_dim_ * config_->dim_;
    } 

    //创建值矩阵(kx_dim * dim)
    for(int32_t i = 0;i < config_->layer_num_; i++){
        auto wv = std::make_shared<op::MatmulLayer>(
            base::DeviceType::kDeviceCPU,config_->kv_dim_,config_->dim_
        );
        wv->set_weight(
            0, {config_->kv_dim_, config_->dim_}, this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU
        );
        llama_layers_->wv_layers_.push_back(wv);
        pos += config_->kv_dim_ * config_->dim_;
    } 

    //创建输出矩阵(dim * dim)
    for(int32_t i = 0;i < config_->layer_num_; i++){
        auto wo = std::make_shared<op::MatmulLayer>(
            base::DeviceType::kDeviceCPU,config_->dim_,config_->dim_
        );
        wo->set_weight(
            0, {config_->dim_, config_->dim_}, this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU
        );
        llama_layers_->wo_layers_.push_back(wo);
        pos += config_->dim_ * config_->dim_;
    } 

    // 跳过ffn resnorm层的权重
    pos += config_->layer_num_ * config_->dim_;

    //w1 layer
    for(int32_t i = 0; i < config_->layer_num_; i++){
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_,config_->hidden_dim_,config_->dim_);
        w1->set_weight(
            0,{config_->hidden_dim_,config_->dim_},this->raw_model_data_->weight(pos),base::DeviceType::kDeviceCPU);
        llama_layers_->w1_layers.push_back(w1);
        pos += config_->dim_ * config_->hidden_dim_;
    }

    //w2 layer
    for(int32_t i = 0; i < config_->layer_num_; i++){
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_,config_->hidden_dim_,config_->dim_);
        w2->set_weight(
            0,{config_->hidden_dim_,config_->dim_},this->raw_model_data_->weight(pos),base::DeviceType::kDeviceCPU);
        llama_layers_->w2_layers.push_back(w2);
        pos += config_->dim_ * config_->hidden_dim_;
    }

    //w3 layer
    for(int32_t i = 0; i < config_->layer_num_; i++){
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_,config_->hidden_dim_,config_->dim_);
        w3->set_weight(
            0,{config_->hidden_dim_,config_->dim_},this->raw_model_data_->weight(pos),base::DeviceType::kDeviceCPU);
        llama_layers_->w3_layers.push_back(w3);
        pos += config_->dim_ * config_->hidden_dim_;
    }

    //skip最终rmsnorm权重
    pos += config_->dim_;
    pos += config_->seq_len_ * config_->head_size_;

    // 创建分类层（classification layer）
    llama_layers_->cls_layer_ = std::make_shared<op::MatmulLayer>(device_type_,config_->vocab_size_,config_->dim_);
    if(config_->is_shared_weight_){
        llama_layers_->cls_layer_->set_weight(
            0,{config_->vocab_size_, config_->dim_},this->raw_model_data_->weight(0), base::DeviceType::kDeviceCPU);
    }else{
        llama_layers_->cls_layer_->set_weight(
            0,{config_->vocab_size_, config_->dim_},this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU);
    }

    //创建rmsnorm层
    size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

    for(int32_t i = 0;i < config_->layer_num_ ; i++){
        std::shared_ptr<op::RmsnormLayer> rmsnorm_layer = std::make_shared<op::RmsnormLayer>(device_type_,config_->dim_);
        const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
        rmsnorm_layer->set_weight(
            0, {config_->dim_}, weight_rmsnorm, base::DeviceType::kDeviceCPU);
        llama_layers_->resnorm_layer.push_back(rmsnorm_layer);
        rmsnorm_pos += config_->dim_;
    }

    // 跳过其他层的rmsnorm权重
    rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
    rmsnorm_pos += config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
    rmsnorm_pos += config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
    rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
    // 创建最终的rmsnorm层
    for(int32_t i = 0;i < config_->layer_num_;i++){
        std::shared_ptr<op::RmsnormLayer> rmsnorm_layer = std::make_shared<op::RmsnormLayer>(device_type_,config_->dim_);
        const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
        rmsnorm_layer->set_weight(
            0, {config_->dim_}, weight_rmsnorm, base::DeviceType::kDeviceCPU);
        llama_layers_->resnorm_layer.push_back(rmsnorm_layer);
        rmsnorm_pos += config_->dim_;
    }

    // 跳过w1, w2, w3层的rmsnorm权重
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
    // 创建最终的rmsnorm层
    std::shared_ptr<op::RmsnormLayer> rms_final_layer = std::make_shared<op::RmsnormLayer>(device_type_,config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_final_layer->set_weight(
        0, {config_->dim_}, weight_rmsnorm, base::DeviceType::kDeviceCPU);
    llama_layers_->resnorm_layer.push_back(rms_final_layer);
}

void model::Llama2Model::create_nonparam_layers()
{
    CHECK(llama_layers_ != nullptr);
    llama_layers_->rope_layer_ = std::make_shared<op::RopeLayer>(
        device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);
    llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
        device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,config_->head_size_);
    llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
    llama_layers_->swiglu_layer_ = std::make_shared<op::SwigluLayer>(device_type_, config_->hidden_dim_);

}

void model::Llama2Model::create_param_quant_layers()
{
    CHECK(is_quant_model_);
    CHECK(llama_layers_ != nullptr);

    // 初始化matmul层，pos用于跟踪权重数据的位置
    size_t pos = 0;
    
    //创建查询矩阵(dim * dim)
    for(int32_t i = 0;i < config_->layer_num_; i++){
        auto wq = std::make_shared<op::MatmulLayer>(
            device_type_,config_->dim_,config_->dim_,true
        );
        wq->set_group_size(group_size_);
        wq->set_weight(
            0, {config_->dim_, config_->dim_}, this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU
        );
        llama_layers_->wq_layers_.push_back(wq);
        pos = pos + config_->dim_ * config_->dim_ + wq->get_scale_num() * sizeof(float);
    }

    //创建键矩阵(kx_dim * dim)
    for(int32_t i = 0;i < config_->layer_num_; i++){
        auto wk = std::make_shared<op::MatmulLayer>(
            device_type_,config_->kv_dim_,config_->dim_,true
        );
        wk->set_group_size(group_size_);
        wk->set_weight(
            0, {config_->kv_dim_, config_->dim_}, this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU
        );
        llama_layers_->wk_layers_.push_back(wk);
        pos = pos + config_->kv_dim_ * config_->dim_ + wk->get_scale_num() * sizeof(float);
    } 

    //创建值矩阵(kx_dim * dim)
    for(int32_t i = 0;i < config_->layer_num_; i++){
        auto wv = std::make_shared<op::MatmulLayer>(
            device_type_,config_->kv_dim_,config_->dim_,true
        );
        wv->set_group_size(group_size_);
        wv->set_weight(
            0, {config_->kv_dim_, config_->dim_}, this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU
        );
        llama_layers_->wv_layers_.push_back(wv);
        pos = pos + config_->kv_dim_ * config_->dim_ + wv->get_scale_num() * sizeof(float);
    } 

    //创建输出矩阵(dim * dim)
    for(int32_t i = 0;i < config_->layer_num_; i++){
        auto wo = std::make_shared<op::MatmulLayer>(
            device_type_,config_->dim_,config_->dim_,true
        );
        wo->set_group_size(group_size_);
        wo->set_weight(
            0, {config_->dim_, config_->dim_}, this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU
        );
        llama_layers_->wo_layers_.push_back(wo);
        pos = pos + config_->dim_ * config_->dim_ + wo->get_scale_num() * sizeof(float);
    } 


    //w1 layer
    for(int32_t i = 0; i < config_->layer_num_; i++){
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_,config_->hidden_dim_,config_->dim_,true);
        w1->set_weight(
            0,{config_->hidden_dim_,config_->dim_},this->raw_model_data_->weight(pos),base::DeviceType::kDeviceCPU);
        llama_layers_->w1_layers.push_back(w1);
        pos += config_->dim_ * config_->hidden_dim_;
    }

    //w2 layer
    for(int32_t i = 0; i < config_->layer_num_; i++){
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_,config_->hidden_dim_,config_->dim_,true);
        w2->set_weight(
            0,{config_->hidden_dim_,config_->dim_},this->raw_model_data_->weight(pos),base::DeviceType::kDeviceCPU);
        llama_layers_->w2_layers.push_back(w2);
        pos += config_->dim_ * config_->hidden_dim_;
    }

    //w3 layer
    for(int32_t i = 0; i < config_->layer_num_; i++){
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_,config_->hidden_dim_,config_->dim_,true);
        w3->set_weight(
            0,{config_->hidden_dim_,config_->dim_},this->raw_model_data_->weight(pos),base::DeviceType::kDeviceCPU);
        llama_layers_->w3_layers.push_back(w3);
        pos += config_->dim_ * config_->hidden_dim_;
    }



    // 创建分类层（classification layer）
    llama_layers_->cls_layer_ = std::make_shared<op::MatmulLayer>(device_type_,config_->vocab_size_,config_->dim_,true);
    llama_layers_->cls_layer_->set_group_size(group_size_);
    if(config_->is_shared_weight_){
        llama_layers_->cls_layer_->set_weight(
            0,{config_->vocab_size_, config_->dim_},this->raw_model_data_->weight(0), base::DeviceType::kDeviceCPU);
    }else{
        llama_layers_->cls_layer_->set_weight(
            0,{config_->vocab_size_, config_->dim_},this->raw_model_data_->weight(pos), base::DeviceType::kDeviceCPU);
        pos = pos + config_->vocab_size_ * config_->dim_ + llama_layers_->cls_layer_->get_scale_num() * sizeof(float);
    }

    //embeding层
    float* weight_ptr = (float*)raw_model_data_->weight(pos);
    llama_layers_->embeding_layer_ = std::make_shared<op::EmbeddingLayer>(
        device_type_,config_->dim_,config_->seq_len_,std::abs(config_->vocab_size_)
    );
    llama_layers_->embedding_layer_->set_weight(
        0, {std::abs(config_->vocab_size_), config_->dim_}, weight_ptr,base::DeviceType::kDeviceCPU
    );

    weight_ptr += config_->vocab_size_ * config_->dim_;

    // rmsnorm attention attention,ffn,final
    for(int32_t i = 0;i < 2*config_->layer_num_ + 1;i++){
        std::shared_ptr<op::RmsnormLayer> rms_norm_layer = std::make_shared<op::RmsnormLayer>(device_type_,config_->dim_);
        rms_norm_layer->set_weight(0, {config_->dim_}, weight_ptr, base::DeviceType::kDeviceCPU);
        llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
        weight_ptr += config_->dim_;
    } 
}

void model::Llama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor &pos_tensor) const
{
    using namespace tensor;
    CHECK(llama_layers_ != nullptr);

    Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
    Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    Tensor query = get_buffer(ModelBufferType::kQuery);

    const auto& mha_layer = llama_layers_->mha_layer_;
    CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";

    int pos = pos_tensor.index<int32_t>(0);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

    Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
    const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
    CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
    STATUS_CHECK(wo_layer->forward(mha_output, attn_output));

}

void model::Llama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor &input) const
{
    CHECK(llama_layers_ != nullptr);
    tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
    if(!rmsnorm_layer){
        std::cerr << "The attention rmsnorm layer is a null pointer in the llama2 model" << std::endl;
    }
    STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));

}

void model::Llama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor &input) const
{
    using namespace tensor;
    CHECK(llama_layers_ != nullptr);

    CHECK_NE(llama_layers_->add_layer_,nullptr) << " The add layer in the feedforward block is null pointer";
    STATUS_CHECK(
        llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input)
    );

    //ffn resnorm
    Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
    const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
    CHECK_NE(ffn_rmsnorm,nullptr) << "The final rmsnorm layer in the feedforward block is null pointer";
    STATUS_CHECK(
        ffn_rmsnorm->forward(input,ffn_norm_output);
    );

    //w1
    Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
    const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
    CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
    STATUS_CHECK(
        w1_layer->forward(ffn_norm_output, w1_output)
    );

    //w3
    Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
    CHECK_NE(w3_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
    STATUS_CHECK(
        w3_layer->forward(ffn_norm_output, w3_output)
    );

    //swiglu
    CHECK_NE(llama_layers_->swiglu_layer_, nullptr) << "The swiglu layer in the feedforward block is null pointer";
    STATUS_CHECK(
        llama_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output)
    );

    //w2
    Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
    const auto& w2_layer = llama_layers_->w3_layers_.at(layer_idx);
    CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
    STATUS_CHECK(
        w2_layer->forward(w1_output, w2_output)
    );
 
    // residual add
    CHECK_NE(llama_layers_->add_layer_, nullptr) << "The add layer in the feedforward block is null pointer";
    STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));

}

void model::Llama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor &pos_tensor) const
{
    CHECK(llama_layers_ != nullptr);
    tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
    int32_t pos = pos_tensor.index<int32_t>(0);

    // wq wk wv @ input
    const auto& [key, val] = slice_kv_cache(layer_idx, pos);
    // query
    const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
    CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";
    auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
    // key
    const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
    CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
    // value
    const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
    CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
    STATUS_CHECK(value_layer->forward(rmsnorm_output, val));
    // rope
    CHECK_NE(llama_layers_->rope_layer_, nullptr)
        << "The RoPE layer in the attention block is null pointer.";
    STATUS_CHECK(llama_layers_->rope_layer_->forward(query, key, pos_tensor, tensor::Tensor{}));

}

void model::Llama2Model::cls_logits(const tensor::Tensor &input) const
{
    CHECK(llama_layers_ != nullptr);
    const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
    CHECK_NE(norm,nullptr);
    STATUS_CHECK(norm->forward(input,input));

    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    CHECK_NE(llama_layers_->cls_layer_, nullptr);
    STATUS_CHECK(
        llama_layers_->cls_layer_->forward(input, forward_output);
    );
}

int32_t model::Llama2Model::post_processing(const tensor::Tensor &pos, bool is_prompt) const
{
    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    const float* forward_logits = forward_output.ptr<float>();

    int32_t next = 0;
    if(is_prompt){
        next = -1;
    }else{
        next = static_cast<int32_t>(sampler_->sample(
            forward_logits, forward_output.size(),nullptr
        ));
    }
    return 0;
}
