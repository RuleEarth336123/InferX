#include "models/qwen2.h"
#include "cpu/rope_kernel.h"
#include <iostream>

model::Qwen2Model::Qwen2Model(base::TokenizerType tokenizer_type,string token_path, \
    string model_path, bool is_quant_model) 
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),std::move(model_path), is_quant_model){
}

base::Status model::Qwen2Model::init(base::DeviceType device_type)
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
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                    get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                    get_buffer(ModelBufferType::kCosCache).ptr<float>());
    } else {

    }
    sampler_ = std::make_unique<ArgmaxSampler>(device_type_);

    return error::Success();
}

base::Status model::Qwen2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                    bool is_prompt, int& next) const
{
    auto status = forward(input,pos_tensor,next);
    if(!status){
        std::cerr << "predict fail!" <<std::endl;
    }
    next = post_processing(pos_tensor, is_prompt);
    return base::error::Success();
}

base::Status model::Qwen2Model::forward(const tensor::Tensor &input, const tensor::Tensor &pos_tensor, int &next) const
{
    if(input.is_empty()){
        return base::error::InvalidArgument("the input tensor is empty.");
    }

    if(device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_){
        return base::error::InternalError("Unsupported int8 quant in the cpu device");
    }

    for(int32_t layer_idx = 0;layer_idx < config_->layer_num_;layer_idx++){
        attention_rms(layer_idx,input);
        attention_qkv(layer_idx,pos_tensor);
        attention_mha(layer_idx, pos_tensor);
        feed_forward(layer_idx,input);
    }

    cls_logits(input);

    return base::error::Success();
}

std::vector<int32_t> model::Qwen2Model::encode(const string &sentence) const
{
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->encode(sentence);
}

std::string model::Qwen2Model::decode(int32_t token_idx) const
{
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->decode(token_idx);
}

int32_t model::Qwen2Model::get_eos() const
{
    // CHECK(this->encode_layer_ != nullptr);
    // return encode_layer_->eos();
    return 0;
}

std::pair<tensor::Tensor, tensor::Tensor> model::Qwen2Model::slice_kv_cache(int32_t layer_idx, int32_t token_pos) const
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

op::embedingOutput model::Qwen2Model::embedding(const std::vector<int> &tokens) const
{
    auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
    auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);

    if(input_tokens.size() != tokens.size()){
        input_tokens.reshape({static_cast<int32_t>(tokens.size())});
        input_embeddings.reshape({static_cast<int32_t>(tokens.size()),config_->dim_});
    }

    for(int32_t i = 0;i<tokens.size();i++){
        input_tokens.index<int32_t>(i) = tokens.at(i);
    }

    auto input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
    LOG_IF(FATAL, !qwen_layers_->embedding_layer_)<< "The embedding layer in the llama2 model is null pointer.";
    //qwen_layers_->embedding_layer_->forward(input_tokens,input_token_num,input_embeddings);
    STATUS_CHECK(
        qwen_layers_->embedding_layer_->forward(input_tokens,input_token_num,input_embeddings)
    );

    op::embedingOutput output;
    output.input_embedings = input_embeddings;
    output.input_tokens = input_tokens;
    output.input_token_num = input_token_num;

    return output;
}

tensor::Tensor model::Qwen2Model::fill_input(const tensor::Tensor &pos_tensor, const op::embedingOutput &embedding_output, bool is_prompt) const
{
    const int32_t pos = pos_tensor.index<int32_t>(0);
    auto [input_tokens,input_embeddings,input_token_num] = embedding_output;

    int32_t index = 0;
    if(is_prompt){
        index = pos;
    }
    std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
        config_->dim_ * sizeof(float), nullptr,\
        input_embeddings.ptr<float>(index * config_->dim_), true);

    tensor::Tensor input(base::DataType::kDataTypeFp32, config_->dim_);
    input.assign(input_emb_buffer);
    input.set_device_type(device_type_);
    return input;
}

bool model::Qwen2Model::is_sentence_ending(int32_t token_idx) const
{
    CHECK(this->encode_layer_ != nullptr);
    return this->encode_layer_->is_sentence_ending(token_idx);
}
void model::Qwen2Model::init_mem()
{
    using namespace tensor;
    std::shared_ptr<base::DeviceAllocator> alloc;
    if(device_type_ == base::DeviceType::kDeviceCPU){
        alloc = base::CPUDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
    CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));


    tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,true, alloc);
    tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,true, alloc);
    CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
    CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));


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

base::Status model::Qwen2Model::create_layers()
{
    using namespace base;
    if(!qwen_layers_){
        qwen_layers_ = std::make_unique<Qwen2layers>();
    }

    if(!is_quant_model_){
        create_param_layers();
    }else{
        create_param_quant_layers();
    }

    create_nonparam_layers();

    if(!qwen_layers_->embedding_layer_){
        std::cerr << "create the embedding layer for the llama model failed!" <<std::endl;
        return error::InternalError("create the embedding layer for the llama model failed!");
    }

    if(qwen_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1){
        std::cerr << "create the rmsnorm layers for the llama model failed!" << std::endl;
        return error::InternalError("create the rmsnorm layers for the llama model failed!" );
    }

    if(qwen_layers_->wq_layers_.size() != config_->layer_num_ ||
        qwen_layers_->wk_layers_.size() != config_->layer_num_ ||
        qwen_layers_->wv_layers_.size() != config_->layer_num_ ||
        qwen_layers_->wo_layers_.size() != config_->layer_num_ ){
        std::cerr << "Create the matmul layer in the attention and ffn attention layers failed! " << std::endl;
        return error::InternalError("Create the matmul layer in the attention and ffn attention layers failed! ");
    }

    for(int32_t i = 0;i < config_->layer_num_; i++){
        if(!qwen_layers_->wq_layers_.at(i) ||
        !qwen_layers_->wq_layers_.at(i) ||
        !qwen_layers_->wq_layers_.at(i) ||
        !qwen_layers_->wq_layers_.at(i) ){
            std::cerr << "Create the matmul layer in the attention and ffn attention layers failed! " << std::endl;
            return error::InternalError("Create the matmul layer in the attention and ffn attention layers failed! ");
        }
    }

    if(qwen_layers_->w1_layers_.size() != config_->layer_num_ ||
    qwen_layers_->w2_layers_.size() != config_->layer_num_ ||
    qwen_layers_->w3_layers_.size() != config_->layer_num_ ){
        std::cerr << "Create the matmul layer in the feedforward layers for the llama model failed! " << std::endl;
        return error::InternalError("Create the matmul layer in the feedforward layers for the llama model failed! ");

    }

    for(int32_t i = 0;i < config_->layer_num_; i++){
        if(!qwen_layers_->w1_layers_.at(i) ||
        !qwen_layers_->w2_layers_.at(i) ||
        !qwen_layers_->w3_layers_.at(i)){
            std::cerr << "Create the matmul layer in the feedforward layers for the llama model failed! " << std::endl;
            return error::InternalError("Create the matmul layer in the feedforward layers for the llama model failed! ");
        }
    }

    if(!qwen_layers_->rope_layer_){
        std::cerr << "Create the rope layer for the llama model failed! " << std::endl;
        return error::InternalError("Create the rope layer for the llama model failed! ");
    }

    if(!qwen_layers_->add_layer_){
        std::cerr << "Create the add layer for the llama model failed!" << std::endl;
        return error::InternalError("Create the add layer for the llama model failed!");
    }

    if(!qwen_layers_->mha_layer_){
        std::cerr << "Create the mha layer for the llama model failed!" << std::endl;
        return error::InternalError("Create the mha layer for the llama model failed!");
    }

    if(!qwen_layers_->swiglu_layer_){
        std::cerr << "Create the swiglu layer for the llama model failed!" << std::endl;
        return error::InternalError("Create the swiglu layer for the llama model failed!");
    }

    return error::Success();    
}

void model::Qwen2Model::create_param_layers()
{
    CHECK(!is_quant_model_);
    CHECK(qwen_layers_ != nullptr);

    auto cpu_device_type = base::DeviceType::kDeviceCPU;
    qwen_layers_->embedding_layer_ =
        std::make_shared<op::EmbeddingLayer>(
            device_type_,config_->dim_,config_->seq_len_,std::abs(config_->vocab_size_)
        );
    
    const void* weight_embeding = raw_model_data_->weight(0);
    qwen_layers_->embedding_layer_->set_weight(
        0,{std::abs(config_->vocab_size_),config_->dim_},weight_embeding,cpu_device_type);

    int32_t dim = config_->dim_;
    size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;

    //WeightQ
    for(int32_t i = 0;i < config_->layer_num_;i++){
        auto wq = std::make_shared<op::MatmulLayer>(device_type_,dim,dim,false,true);
        wq->set_weight(0,{dim,dim},raw_model_data_->weight(pos),cpu_device_type);
        pos += dim * dim;
        wq->set_bias(0,dim,raw_model_data_->weight(pos),cpu_device_type);
        pos += dim;
        qwen_layers_->wq_layers_.push_back(wq);
    }

    //WeightK
    for(int32_t i = 0;i < config_->layer_num_;i++){
        auto wk = std::make_shared<op::MatmulLayer>(device_type_,config_->kv_dim_,dim,false,true);
        wk->set_weight(0,{config_->kv_dim_,dim},raw_model_data_->weight(pos),cpu_device_type);
        pos += config_->kv_dim_ * dim;
        wk->set_bias(0,config_->kv_dim_,raw_model_data_->weight(pos),cpu_device_type);
        pos += config_->kv_dim_;
        qwen_layers_->wk_layers_.push_back(wk);
    } 

    //WeightV
    for(int32_t i = 0;i < config_->layer_num_;i++){
        auto wv = std::make_shared<op::MatmulLayer>(device_type_,config_->kv_dim_,dim,false,true);
        wv->set_weight(0,{config_->kv_dim_,dim},raw_model_data_->weight(pos),cpu_device_type);
        pos += config_->kv_dim_ * dim;
        wv->set_bias(0,config_->kv_dim_,raw_model_data_->weight(pos),cpu_device_type);
        pos += config_->kv_dim_;
        qwen_layers_->wv_layers_.push_back(wv);
    }

    //create weight matrix for output
    for(int32_t i = 0;i < config_->layer_num_;i++){
        auto wo = std::make_shared<op::MatmulLayer>(device_type_,dim,dim);
        wo->set_weight(0,{dim,dim},this->raw_model_data_->weight(pos),cpu_device_type);
        qwen_layers_->wo_layers_.push_back(wo);
        pos += dim * dim;
    }

    //skip ffn rmsnorm
    pos += config_->layer_num_ * dim;

    //全连接层
    //w1
    int32_t hidden_dim = config_->hidden_dim_;
    for(int32_t i = 0;i < config_->layer_num_;i++){
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_,hidden_dim,dim);
        w1->set_weight(0,{dim,hidden_dim},this->raw_model_data_->weight(pos),cpu_device_type);
        qwen_layers_->w1_layers_.push_back(w1);
        pos += dim * hidden_dim;
    }
    
    for(int32_t i = 0;i < config_->layer_num_;i++){
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_,hidden_dim,dim);
        w2->set_weight(0,{dim,hidden_dim},this->raw_model_data_->weight(pos),cpu_device_type);
        qwen_layers_->w2_layers_.push_back(w2);
        pos += dim * hidden_dim;
    }

    for(int32_t i = 0;i < config_->layer_num_;i++){
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_,hidden_dim,dim);
        w3->set_weight(0,{dim,hidden_dim},this->raw_model_data_->weight(pos),cpu_device_type);
        qwen_layers_->w3_layers_.push_back(w3);
        pos += dim * hidden_dim;
    }

    //skip 最后的rmsnorm
    pos += dim;
    // skip freqs_cos and freqs_sin weight
    pos += config_->seq_len_ * config_->head_size_;

    qwen_layers_->cls_layer_ = 
        std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
    if(config_->is_shared_weight_){
        //使用token embeding weight
        qwen_layers_->cls_layer_->set_weight(
            0,{config_->vocab_size_,dim},this->raw_model_data_->weight(0),cpu_device_type);
    }else{
        qwen_layers_->cls_layer_->set_weight(
            0,{config_->vocab_size_,dim},this->raw_model_data_->weight(pos),cpu_device_type);
    }

    //创建rmsnorm1
    size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

    for(int32_t i=0;i<config_->layer_num_;i++){
        std::shared_ptr<op::RmsnormLayer> rmsnorm_layer = 
            std::make_shared<op::RmsnormLayer>(device_type_,config_->dim_);
        const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
        rmsnorm_layer->set_weight(0,{config_->dim_},weight_rmsnorm,cpu_device_type);
        qwen_layers_->rmsnorm_layers_.push_back(rmsnorm_layer);
        rmsnorm_pos += config_->dim_;
    }

    //rmsnorm2
    //skip q、k、v、o
    rmsnorm_pos += config_->layer_num_ * (config_->dim_ * config_->dim_ + config_->dim_);
    rmsnorm_pos += config_->layer_num_ * (config_->dim_ * config_->kv_dim_ + config_->kv_dim_);
    rmsnorm_pos += config_->layer_num_ * (config_->dim_ * config_->kv_dim_ + config_->kv_dim_);
    rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

    for(int32_t i=0;i<config_->layer_num_;i++){
        std::shared_ptr<op::RmsnormLayer> rmsnorm_layer = 
            std::make_shared<op::RmsnormLayer>(device_type_,config_->dim_);
        const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
        rmsnorm_layer->set_weight(0,{config_->dim_},weight_rmsnorm,cpu_device_type);
        qwen_layers_->rmsnorm_layers_.push_back(rmsnorm_layer);
        rmsnorm_pos += config_->dim_;
    }

    //rmsnorm3
    //skip ffn.w1,ffn.w2,ffn.w3
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

    std::shared_ptr<op::RmsnormLayer> rmsnorm_final_layer = 
        std::make_shared<op::RmsnormLayer>(device_type_, config_->dim_);
    
    const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
    rmsnorm_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
    qwen_layers_->rmsnorm_layers_.push_back(rmsnorm_final_layer);
}

void model::Qwen2Model::create_nonparam_layers()
{
    CHECK(qwen_layers_ != nullptr);
    qwen_layers_->rope_layer_ = std::make_shared<op::RopeLayer>(
        device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);
    qwen_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
        device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,config_->head_size_);
    qwen_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
    qwen_layers_->swiglu_layer_ = std::make_shared<op::SwigluLayer>(device_type_, config_->hidden_dim_);

}

void model::Qwen2Model::create_param_quant_layers()
{
    
}

void model::Qwen2Model::attention_mha(int32_t layer_idx, const tensor::Tensor &pos_tensor) const
{
    using namespace tensor;
    CHECK(qwen_layers_ != nullptr);

    Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
    Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    Tensor query = get_buffer(ModelBufferType::kQuery);

    const auto& mha_layer = qwen_layers_->mha_layer_;
    CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";

    int pos = pos_tensor.index<int32_t>(0);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

    Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
    const auto& wo_layer = qwen_layers_->wo_layers_.at(layer_idx);
    CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
    STATUS_CHECK(wo_layer->forward(mha_output, attn_output));

}

void model::Qwen2Model::attention_rms(int32_t layer_idx, const tensor::Tensor &input) const
{
    CHECK(qwen_layers_ != nullptr);
    tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    std::shared_ptr<op::Layer> rmsnorm_layer = qwen_layers_->rmsnorm_layers_.at(layer_idx);
    if(!rmsnorm_layer){
        std::cerr << "The attention rmsnorm layer is a null pointer in the llama2 model" << std::endl;
    }
    STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));

}

void model::Qwen2Model::feed_forward(int32_t layer_idx, const tensor::Tensor &input) const
{
    using namespace tensor;
    CHECK(qwen_layers_ != nullptr);

    CHECK_NE(qwen_layers_->add_layer_,nullptr) << " The add layer in the feedforward block is null pointer";
    STATUS_CHECK(
        qwen_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input)
    );

    //ffn resnorm
    Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
    const auto& ffn_rmsnorm = qwen_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
    CHECK_NE(ffn_rmsnorm,nullptr) << "The final rmsnorm layer in the feedforward block is null pointer";
    STATUS_CHECK(
        ffn_rmsnorm->forward(input,ffn_norm_output);
    );

    //w1
    Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
    const auto& w1_layer = qwen_layers_->w1_layers_.at(layer_idx);
    CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
    w1_layer->forward(ffn_norm_output, w1_output);
    // STATUS_CHECK(
    //     w1_layer->forward(ffn_norm_output, w1_output)
    // );

    //w3
    Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    const auto& w3_layer = qwen_layers_->w3_layers_.at(layer_idx);
    CHECK_NE(w3_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
    STATUS_CHECK(
        w3_layer->forward(ffn_norm_output, w3_output)
    );

    //swiglu
    CHECK_NE(qwen_layers_->swiglu_layer_, nullptr) << "The swiglu layer in the feedforward block is null pointer";
    STATUS_CHECK(
        qwen_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output)
    );

    //w2
    Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
    const auto& w2_layer = qwen_layers_->w3_layers_.at(layer_idx);
    CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
    STATUS_CHECK(
        w2_layer->forward(w1_output, w2_output)
    );
 
    // residual add
    CHECK_NE(qwen_layers_->add_layer_, nullptr) << "The add layer in the feedforward block is null pointer";
    STATUS_CHECK(qwen_layers_->add_layer_->forward(input, w2_output, input));

}

void model::Qwen2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor &pos_tensor) const
{
    CHECK(qwen_layers_ != nullptr);
    tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
    int32_t pos = pos_tensor.index<int32_t>(0);

    // wq wk wv @ input
    const auto& [key, val] = slice_kv_cache(layer_idx, pos);
    // query
    const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
    CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";
    auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    query_layer->forward(rmsnorm_output, query);
    //STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
    // key
    const auto& key_layer = qwen_layers_->wk_layers_.at(layer_idx);
    CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
    // value
    const auto& value_layer = qwen_layers_->wv_layers_.at(layer_idx);
    CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
    STATUS_CHECK(value_layer->forward(rmsnorm_output, val));
    // rope
    CHECK_NE(qwen_layers_->rope_layer_, nullptr)
        << "The RoPE layer in the attention block is null pointer.";
    STATUS_CHECK(qwen_layers_->rope_layer_->forward(\
        query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));

}

void model::Qwen2Model::cls_logits(const tensor::Tensor &input) const
{
    CHECK(qwen_layers_ != nullptr);
    const auto& norm = qwen_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
    CHECK_NE(norm,nullptr);
    STATUS_CHECK(norm->forward(input,input));

    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    CHECK_NE(qwen_layers_->cls_layer_, nullptr);
    STATUS_CHECK(
        qwen_layers_->cls_layer_->forward(input, forward_output);
    );
}

int32_t model::Qwen2Model::post_processing(const tensor::Tensor &pos, bool is_prompt) const
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
