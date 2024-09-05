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
    return std::vector<int32_t>();
}

std::string model::Llama2Model::decode(int32_t token_idx) const
{
    return std::string();
}

int32_t model::Llama2Model::get_eos() const
{
    return 0;
}

std::pair<tensor::Tensor, tensor::Tensor> model::Llama2Model::slice_kv_cache(int32_t layer_idx, int32_t token_pos) const
{
    return std::pair<tensor::Tensor, tensor::Tensor>();
}

op::embedingOutput model::Llama2Model::embedding(const std::vector<int> &tokens) const
{
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
    


}

void model::Llama2Model::create_param_quant_layers()
{
}

void model::Llama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor &pos_tensor) const
{
}

void model::Llama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor &input) const
{
}

void model::Llama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor &input) const
{
}

void model::Llama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor &pos_tensor) const
{
}

void model::Llama2Model::cls_logits(const tensor::Tensor &input) const
{
}

int32_t model::Llama2Model::post_processing(const tensor::Tensor &pos, bool is_prompt) const
{
    return 0;
}
