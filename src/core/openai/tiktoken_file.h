#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace zazaki_tokenizer {
namespace openai {

struct TiktokenData {
    std::string regex_pattern;
    std::unordered_map<std::string, uint32_t> rank_map;
    std::unordered_map<std::string, uint32_t> special_tokens;
    std::vector<std::string> decode_map;
};

TiktokenData parse_tiktoken(const unsigned char* data, unsigned int len);

} // namespace openai
} // namespace zazaki_tokenizer
