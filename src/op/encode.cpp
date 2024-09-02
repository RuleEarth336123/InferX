#include "op/encode.h"
#include "glog/logging.h"

op::EncodeLayer::EncodeLayer(base::DeviceType device_type)
    : Layer(device_type,LayerType::kLayerEncode,"Encode")
{
}

op::EncodeLayer::EncodeLayer(base::DeviceType device_type, bool has_bos, bool has_eos, std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor)
    : Layer(device_type,LayerType::kLayerEncode,"Encode"),
    has_bos_(has_bos),
    has_eos_(has_eos),
    spe(std::move(sentence_piece_processor))
{
}

std::vector<int32_t> op::EncodeLayer::encode(const std::string &sentence) const
{
    CHECK(spe != nullptr);
    std::vector<int32_t> input_ids = spe->EncodeAsIds(sentence);
    if(has_bos_){
        input_ids.insert(input_ids.begin(),spe->bos_id());
    }
    if(has_eos_){
        input_ids.push_back(spe->eos_id());
    }
    return input_ids;
}

std::string op::EncodeLayer::decode(int32_t token_id) const
{
    CHECK(spe != nullptr);
    std::vector<int32_t> token_ids(token_id);
    return this->spe->DecodeIds(token_ids);
}

int32_t op::EncodeLayer::eos() const
{
    CHECK(spe != nullptr);
    return this->spe->eos_id();
}
