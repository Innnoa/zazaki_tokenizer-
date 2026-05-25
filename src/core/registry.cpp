#include "registry.h"

#include <stdexcept>

extern const unsigned char cl100k_base_tiktoken[];
extern const unsigned int cl100k_base_tiktoken_len;

extern const unsigned char o200k_base_tiktoken[];
extern const unsigned int o200k_base_tiktoken_len;

namespace zazaki_tokenizer {

Registry::Registry() {
    auto cl100k_data = openai::parse_tiktoken(cl100k_base_tiktoken, cl100k_base_tiktoken_len);
    auto cl100k = std::make_shared<openai::Encoder>(std::move(cl100k_data));

    models_["gpt-4"] = cl100k;
    models_["gpt-4-turbo"] = cl100k;
    models_["gpt-3.5-turbo"] = cl100k;
    models_["text-embedding-ada-002"] = cl100k;

    auto o200k_data = openai::parse_tiktoken(o200k_base_tiktoken, o200k_base_tiktoken_len);
    auto o200k = std::make_shared<openai::Encoder>(std::move(o200k_data));

    models_["gpt-4o"] = o200k;
    models_["gpt-4o-mini"] = o200k;
    models_["o1"] = o200k;
    models_["o3"] = o200k;
}

Registry& Registry::instance() {
    static Registry reg;
    return reg;
}

const openai::Encoder* Registry::get_encoder(const std::string& model_name) const {
    auto it = models_.find(model_name);
    if (it == models_.end()) {
        throw std::invalid_argument("unknown model: " + model_name);
    }
    return it->second.get();
}

} // namespace zazaki_tokenizer
