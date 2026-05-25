#include "openai/encoder.h"

#include <re2/re2.h>

#include <algorithm>
#include <limits>

namespace zazaki_tokenizer {
namespace openai {

Encoder::Encoder(TiktokenData data) : data_(std::move(data)) {
    pattern_ = std::make_unique<re2::RE2>(data_.regex_pattern);
    if (!pattern_->ok()) {
        throw std::runtime_error("invalid regex pattern: " + pattern_->error());
    }
}

std::vector<std::string> Encoder::split_by_pattern(const std::string& text) const {
    std::vector<std::string> pieces;
    re2::StringPiece input(text);
    re2::StringPiece match;

    size_t last_end = 0;
    while (re2::RE2::FindAndConsume(&input, *pattern_, &match)) {
        size_t match_start = match.data() - text.data();
        if (match_start > last_end) {
            pieces.emplace_back(text.substr(last_end, match_start - last_end));
        }
        pieces.emplace_back(match.as_string());
        last_end = match_start + match.size();
    }

    if (last_end < text.size()) {
        pieces.emplace_back(text.substr(last_end));
    }

    return pieces;
}

std::vector<uint32_t> Encoder::bpe_merge(const std::string& piece_bytes) const {
    if (piece_bytes.empty()) {
        return {};
    }

    std::vector<Piece> parts;
    parts.reserve(piece_bytes.size());

    for (unsigned char c : piece_bytes) {
        std::string byte_str(1, static_cast<char>(c));
        auto it = data_.rank_map.find(byte_str);
        if (it == data_.rank_map.end()) {
            throw std::runtime_error(
                "byte not found in vocabulary: " + std::to_string(static_cast<int>(c)));
        }
        parts.push_back({std::move(byte_str), it->second});
    }

    if (parts.size() == 1) {
        return {parts[0].rank};
    }

    while (true) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        size_t best_idx = 0;
        bool found = false;

        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            std::string merged_bytes = parts[i].bytes + parts[i + 1].bytes;
            auto it = data_.rank_map.find(merged_bytes);
            if (it != data_.rank_map.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
                found = true;
            }
        }

        if (!found) {
            break;
        }

        parts[best_idx].bytes += parts[best_idx + 1].bytes;
        parts[best_idx].rank = best_rank;
        parts.erase(parts.begin() + static_cast<long>(best_idx + 1));
    }

    std::vector<uint32_t> result;
    result.reserve(parts.size());
    for (const auto& p : parts) {
        result.push_back(p.rank);
    }
    return result;
}

std::vector<uint32_t> Encoder::encode(const std::string& text) const {
    auto pieces = split_by_pattern(text);
    std::vector<uint32_t> result;
    for (const auto& piece : pieces) {
        auto tokens = bpe_merge(piece);
        result.insert(result.end(), tokens.begin(), tokens.end());
    }
    return result;
}

std::string Encoder::decode(const std::vector<uint32_t>& ids) const {
    std::string result;
    for (uint32_t id : ids) {
        if (id >= data_.decode_map.size()) {
            throw std::out_of_range("token id out of range: " + std::to_string(id));
        }
        result += data_.decode_map[id];
    }
    return result;
}

size_t Encoder::count(const std::string& text) const {
    auto pieces = split_by_pattern(text);
    size_t total = 0;
    for (const auto& piece : pieces) {
        total += bpe_merge(piece).size();
    }
    return total;
}

} // namespace openai
} // namespace zazaki_tokenizer
