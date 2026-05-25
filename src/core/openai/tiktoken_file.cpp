#include "openai/tiktoken_file.h"

#include <cstring>
#include <stdexcept>

namespace zazaki_tokenizer {
namespace openai {

namespace {

uint32_t read_u32(const unsigned char*& p, const unsigned char* end) {
    if (p + 4 > end) {
        throw std::runtime_error("tiktoken file truncated at u32 read");
    }
    uint32_t val = static_cast<uint32_t>(p[0])
                 | (static_cast<uint32_t>(p[1]) << 8)
                 | (static_cast<uint32_t>(p[2]) << 16)
                 | (static_cast<uint32_t>(p[3]) << 24);
    p += 4;
    return val;
}

} // namespace

TiktokenData parse_tiktoken(const unsigned char* data, unsigned int len) {
    const unsigned char* p = data;
    const unsigned char* end = data + len;

    TiktokenData result;

    uint32_t version = read_u32(p, end);
    if (version != 1) {
        throw std::runtime_error("unsupported tiktoken version: " + std::to_string(version));
    }

    // Regex pattern
    uint32_t regex_len = read_u32(p, end);
    if (p + regex_len > end) {
        throw std::runtime_error("tiktoken file truncated at regex");
    }
    result.regex_pattern.assign(reinterpret_cast<const char*>(p), regex_len);
    p += regex_len;

    // Mergeable ranks
    uint32_t num_ranks = read_u32(p, end);
    uint32_t max_rank = 0;
    result.rank_map.reserve(num_ranks);

    for (uint32_t i = 0; i < num_ranks; ++i) {
        uint32_t token_len = read_u32(p, end);
        if (p + token_len > end) {
            throw std::runtime_error("tiktoken file truncated at token bytes");
        }
        std::string token_bytes(reinterpret_cast<const char*>(p), token_len);
        p += token_len;

        uint32_t rank = read_u32(p, end);
        result.rank_map[std::move(token_bytes)] = rank;
        if (rank > max_rank) {
            max_rank = rank;
        }
    }

    // Build decode map
    result.decode_map.resize(static_cast<size_t>(max_rank) + 1);
    for (const auto& [token_bytes, rank] : result.rank_map) {
        result.decode_map[rank] = token_bytes;
    }

    // Special tokens
    uint32_t num_special = read_u32(p, end);
    result.special_tokens.reserve(num_special);

    for (uint32_t i = 0; i < num_special; ++i) {
        uint32_t name_len = read_u32(p, end);
        if (p + name_len > end) {
            throw std::runtime_error("tiktoken file truncated at special token name");
        }
        std::string name(reinterpret_cast<const char*>(p), name_len);
        p += name_len;

        uint32_t id = read_u32(p, end);
        result.special_tokens[std::move(name)] = id;
    }

    return result;
}

} // namespace openai
} // namespace zazaki_tokenizer
