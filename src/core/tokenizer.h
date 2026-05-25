#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace zazaki_tokenizer {

struct EncodingResult {
    std::vector<uint32_t> token_ids;
    size_t token_count;
};

class Tokenizer {
public:
    static Tokenizer for_model(const std::string& model_name);

    EncodingResult encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& token_ids) const;
    size_t count_tokens(const std::string& text) const;

private:
    explicit Tokenizer(const std::string& model_name);
    std::string model_name_;
};

} // namespace zazaki_tokenizer
