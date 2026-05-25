#pragma once

#include "openai/tiktoken_file.h"

#include <memory>
#include <string>
#include <vector>

namespace re2 {
class RE2;
}

namespace zazaki_tokenizer {
namespace openai {

class Encoder {
public:
    explicit Encoder(TiktokenData data);
    ~Encoder();

    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& ids) const;
    size_t count(const std::string& text) const;

private:
    struct Piece {
        std::string bytes;
        uint32_t rank;
    };

    std::vector<std::string> split_by_pattern(const std::string& text) const;
    std::vector<uint32_t> bpe_merge(const std::string& piece_bytes) const;

    TiktokenData data_;
    std::unique_ptr<re2::RE2> pattern_;
};

} // namespace openai
} // namespace zazaki_tokenizer
