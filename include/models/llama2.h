#ifndef MODEL_H_LLAMA2_
#define MODEL_H_LLAMA2_

#include <string>
#include "models/model.h"
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <op/embeding.h>
#include <op/rope.h>
#include <op/swiglu.h>
#include <op/add.h>
#include <glog/logging.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "base/base.h"
#include "base/alloc.h"
#include "glog/logging.h"

namespace model{
    struct Llama2layers{
        std::shared_ptr<op::Layer> add_layer_; 
        std::shared_ptr<op::Layer> rope_layer_; 
        std::shared_ptr<op::Layer> swiglu_layer_; 
        std::shared_ptr<op::Layer> mha_layer_;
        std::shared_ptr<op::Layer> embeding_layer_;

        std::vector<std::shared_ptr<op::Layer>> wq_layers_;
        std::vector<std::shared_ptr<op::Layer>> wk_layers_;
        std::vector<std::shared_ptr<op::Layer>> wv_layers_;
        std::vector<std::shared_ptr<op::Layer>> wo_layers_;

        std::vector<std::shared_ptr<op::Layer>> w1_layers_;
        std::vector<std::shared_ptr<op::Layer>> w2_layers_;
        std::vector<std::shared_ptr<op::Layer>> w3_layers_;
        std::vector<std::shared_ptr<op::Layer>> resnorm_layers_;

        void to_cuda(){}
    };
}

namespace model{

    using std::string;


    class Llama2Model : public Model{
    public:
        explicit Llama2Model(string token_path,string model_path,bool is_quant_model);

        base::Status init(base::DeviceType device_type) override;
        base::Status predict(const tensor::Tensor& init,const tensor::Tensor& pos_tensor,bool is_prompt,int& next) const override;
        base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,int next) const override;

        std::vector<int32_t> encode(const string& sentence) const override;
        std::string decode(int32_t token_idx) const override;

        int32_t get_eos() const override;
        std::pair<tensor::Tensor,tensor::Tensor> slice_kv_cache(int32_t layer_idx,int32_t token_pos) const override;

        op::embedingOutput embedding(const std::vector<int>& tokens) const;

        tensor::Tensor fill_output(const tensor::Tensor& pos_tensor,const op::embedingOutput& ,bool is_prompt) const;

    private:

        void init_mem() override;
        base::Status create_layers() override;
        void create_param_layers() override; 
        void create_nonparam_layers() override;
        void create_param_quant_layers() override;

        void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
        void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;
        void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;
        void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;
        void cls_logits(const tensor::Tensor& input) const;
        int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

    private:

        std::unique_ptr<model::LLama2Layers> llama_layers_;

    };
}



#endif