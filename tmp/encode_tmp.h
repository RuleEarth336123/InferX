#ifndef INCLUDE_OP_ENCODE_H_
#define INCLUDE_OP_ENCODE_H_

#include "op/layer.h"
#include "sentencepiece_processor.h"

namespace op{
    class EncodeLayer : public Layer{
    public:
        explicit EncodeLayer(base::DeviceType device_type);
        explicit EncodeLayer(\
            base::DeviceType device_type,bool has_bos,bool has_eos,\
            std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor\
        );
        /**
         * 使用给定的句子，通过特定的编码方式进行编码。
         *
         * @param sentence 需要进行编码的字符串句子。
         * @return 返回一个包含编码后整数的向量。如果设置了has_bos_，则在向量开始处插入特殊字符的id；如果设置了has_eos_，则在向量末尾添加特殊字符的id。
         */
        std::vector<int32_t> encode(const std::string& sentence) const;
        /**
         * 使用给定的token_id，通过spe对象进行解码。
         *
         * @param token_id 需要被解码的token id。
         * @return 返回解码后的字符串。
         */
        std::string decode(int32_t token_id) const;
        //获取编码层的结束符id
        int32_t eos() const;    


    private:

        bool has_bos_ = true;       //是否存在开始符号
        bool has_eos_ = false;      //是否存在结束符
        std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;
    };

    //#if define QWEN2_SUPPORT
    class BpeEncodeLayer : public EncodeLayer{
    public:
        explicit BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos);
        std::vector<int32_t> encode(const std::string& sentence) const override;
        std::string decode(int32_t token_id) const override;
        std::string decode(const std::vector<int32_t>& token_ids) const override;
        bool is_sentence_ending(int32_t token_id) const override;
        int32_t vocab_size() const override;
    protected:
        int32_t bos_id_ = -1;
        int32_t eos_id_ = -1;
        int32_t stop_token1_ = -1;
        int32_t stop_token2_ = -1;
        int32_t num_token_ = 0;
        std::unique_ptr<tiktoken::tiktoken> tiktoken_;
    };

}

#endif
