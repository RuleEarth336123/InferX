#ifndef MODEL_H_MODEL_
#define MODEL_H_MODEL_

#include <map>
#include <string>
#include <models/config.h>
#include "models/raw_model_data.h"
#include "models/sampler.h"
#include "op/encode.h"
#include "op/layer.h"
#include "tensor/tensor.h"
#include "sentencepiece_processor.h"

namespace model{
    class Model{
    public:
        explicit Model(base::TokenizerType tokenizer_type,base::ModelType model_type,std::string token_path,std::string model_path,bool is_quant_modedl);
        
        virtual base::Status init(base::DeviceType device_type) = 0;

        virtual base::Status predict(\
            const tensor::Tensor& input ,const tensor::Tensor& pos_tensor ,bool is_prompt, int& next\
        ) const = 0;

        virtual base::Status forward(\
            const tensor::Tensor& input ,const tensor::Tensor& pos_tensor , int& next\
        ) const = 0;

        virtual int32_t get_eos() const = 0;

        virtual bool is_sentence_ending(int32_t token_idx) const = 0;

        base::ModelType model_type() const;

        const std::string& token_path() const;

        const std::string& model_path() const;
        /**
         * 获取指定索引的模型缓冲区。
         *
         * @param buffer_idx 要获取的模型缓冲区的索引。
         * @return 返回指定索引的模型缓冲区。
         */
        virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);
        /**
         * 获取指定索引的模型缓冲区。
         *
         * @param buffer_idx 要获取的模型缓冲区的索引。
         * @return 返回指定索引的模型缓冲区。
         */
        virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;
        /**
         * 使用给定的token_id，通过spe对象进行解码。
         *
         * @param token_id 需要被解码的token id。
         * @return 返回解码后的字符串。
         */
        virtual std::string decode(int32_t token_idx) const = 0;

    protected:
        /**
         * 将给定的张量插入到指定的缓冲区中。如果缓冲区已存在或张量为空，则返回错误。
         *
         * @param buffer_idx 要插入的缓冲区的索引。
         * @param tensor     要插入的张量。
         * @return 如果成功插入，则返回成功状态；否则，返回相应的错误信息。
         */
        virtual base::Status insert_buffer(ModelBufferType buffer_idx,const tensor::Tensor& tensor);
        /**
         * 读取模型文件。
         *
         * @return 如果成功，返回状态为Success()；如果失败，返回相应的错误信息。
         */
        virtual base::Status read_model_file();
        /**
         * 创建编码层。首先，加载SentencePieceProcessor，然后从模型文件中读取词汇表大小参数。如果词汇表大小参数小于等于0，则返回错误。最后，创建EncodeLayer并返回成功状态。
         *
         * @return 如果成功创建编码层，返回Success()；否则，返回相应的错误信息。
         */
        virtual base::Status create_encode_layer();
        /**
         * 从文件中生成模型。首先创建编码层，然后读取模型文件，最后创建模型的各层。
         *
         * @return 如果所有步骤都成功执行，返回error::Success；否则，返回相应的错误状态。
         */
        virtual base::Status gen_model_from_file();
        /**
         * 根据给定的模型配置生成模型信息。
         *
         * @param config 模型配置，包含模型的各种参数和设置。
         * @return 如果成功生成模型信息，返回base::error::Sucess()；如果词汇表大小与模型文件不匹配，返回base::error::ModelParseError("Vocabulary size mismatch between the model file and the token list.")。
         */
        virtual base::Status generate_model_infos(const ModelConfig& config) const; 

        virtual int32_t post_processing(const tensor::Tensor& pos,bool is_prompt) const = 0;
    
    private:

        virtual void init_mem() = 0;
        virtual base::Status create_layers() = 0;
        virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;
        virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                                        int32_t token_pos) const = 0;
        virtual void create_param_layers() = 0;
        virtual void create_nonparam_layers() = 0;
        virtual void create_param_quant_layers() = 0;

    protected:

        int32_t group_size_ = 1;
        bool is_quant_model_ = false;
        std::unique_ptr<TransformerConfig> config_;

        std::string token_path_;
        std::string model_path_;
        std::unique_ptr<op::EncodeLayerBase> encode_layer_;
        std::map<ModelBufferType, tensor::Tensor> buffers_;
        std::unique_ptr<model::Sampler> sampler_;
        std::shared_ptr<RawModelData> raw_model_data_;
        base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
        base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
        base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;

    };
}






#endif
