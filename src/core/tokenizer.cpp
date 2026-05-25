#include "tokenizer.h"
#include "registry.h"

namespace zazaki_tokenizer {

Tokenizer::Tokenizer(const std::string& model_name)
    : model_name_(model_name) {}

Tokenizer Tokenizer::for_model(const std::string& model_name) {
    Registry::instance().get_encoder(model_name);
    return Tokenizer(model_name);
}

EncodingResult Tokenizer::encode(const std::string& text) const {
    auto* enc = Registry::instance().get_encoder(model_name_);
    auto ids = enc->encode(text);
    size_t count = ids.size();
    return {std::move(ids), count};
}

std::string Tokenizer::decode(const std::vector<uint32_t>& token_ids) const {
    auto* enc = Registry::instance().get_encoder(model_name_);
    return enc->decode(token_ids);
}

size_t Tokenizer::count_tokens(const std::string& text) const {
    auto* enc = Registry::instance().get_encoder(model_name_);
    return enc->count(text);
}

} // namespace zazaki_tokenizer
