#pragma once

#include "openai/encoder.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace zazaki_tokenizer {

using EncoderPtr = std::shared_ptr<openai::Encoder>;

class Registry {
public:
    static Registry& instance();

    const openai::Encoder* get_encoder(const std::string& model_name) const;

    void register_model(const std::string& model_name, EncoderPtr encoder);
    void register_from_file(const std::string& model_name,
                            const std::string& file_path);
    bool has_model(const std::string& model_name) const;

private:
    Registry();
    mutable std::unordered_map<std::string, EncoderPtr> models_;
};

} // namespace zazaki_tokenizer
