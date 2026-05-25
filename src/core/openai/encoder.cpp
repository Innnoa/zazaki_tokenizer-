#include "openai/encoder.h"

#include <re2/re2.h>

#include <algorithm>
#include <limits>
#include <queue>
#include <vector>

namespace zazaki_tokenizer {
namespace openai {

namespace {

struct MergeCandidate {
    uint32_t rank;
    size_t pos;
    uint32_t left_ver;
    uint32_t right_ver;

    bool operator>(const MergeCandidate& o) const {
        if (rank != o.rank) return rank > o.rank;
        return pos > o.pos;
    }
};

} // namespace

Encoder::Encoder(TiktokenData data) : data_(std::move(data)) {
    if (data_.regex_pattern.empty() || data_.regex_pattern == ".") {
        pattern_ = nullptr;
    } else {
        pattern_ = std::make_unique<re2::RE2>(data_.regex_pattern);
        if (!pattern_->ok()) {
            throw std::runtime_error("invalid regex pattern: " + pattern_->error());
        }
    }
}

Encoder::~Encoder() = default;

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
        pieces.emplace_back(std::string(match));
        last_end = match_start + match.size();
    }

    if (last_end < text.size()) {
        pieces.emplace_back(text.substr(last_end));
    }

    return pieces;
}

std::vector<uint32_t> Encoder::bpe_merge(const std::string& piece_bytes) const {
    size_t n = piece_bytes.size();
    if (n == 0) return {};
    if (n == 1) {
        std::string byte_str(1, piece_bytes[0]);
        auto it = data_.rank_map.find(byte_str);
        if (it == data_.rank_map.end()) {
            throw std::runtime_error(
                "byte not found in vocabulary: " +
                std::to_string(static_cast<int>(static_cast<unsigned char>(piece_bytes[0]))));
        }
        return {it->second};
    }

    // Initialize tokens
    std::vector<std::string> token_bytes(n);
    std::vector<uint32_t> token_prev(n);
    std::vector<uint32_t> token_next(n);
    std::vector<uint32_t> token_ver(n, 1);
    constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();

    for (size_t i = 0; i < n; ++i) {
        token_bytes[i] = std::string(1, piece_bytes[i]);
        token_prev[i] = (i > 0) ? static_cast<uint32_t>(i - 1) : kNone;
        token_next[i] = (i + 1 < n) ? static_cast<uint32_t>(i + 1) : kNone;
    }

    // Build initial heap
    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, std::greater<>> heap;
    auto try_push = [&](size_t left) {
        size_t right = token_next[left];
        if (right == kNone) return;
        std::string merged = token_bytes[left] + token_bytes[right];
        auto it = data_.rank_map.find(merged);
        if (it != data_.rank_map.end()) {
            heap.push({it->second, left, token_ver[left], token_ver[right]});
        }
    };

    for (size_t i = 0; i + 1 < n; ++i) {
        try_push(i);
    }

    // Merge loop
    while (!heap.empty()) {
        auto cand = heap.top();
        heap.pop();

        size_t left = cand.pos;
        size_t right = token_next[left];
        if (right == kNone) continue;
        if (cand.left_ver != token_ver[left]) continue;
        if (cand.right_ver != token_ver[right]) continue;

        // Perform the merge
        token_bytes[left] += token_bytes[right];
        token_ver[left]++;
        token_ver[right]++;

        size_t right_next = token_next[right];
        token_next[left] = right_next;
        if (right_next != kNone) {
            token_prev[right_next] = static_cast<uint32_t>(left);
        }
        token_bytes[right].clear();

        // Push new candidates involving the merged token
        size_t left_neighbor = token_prev[left];
        if (left_neighbor != kNone) {
            try_push(left_neighbor);
        }
        if (token_next[left] != kNone) {
            try_push(left);
        }
    }

    // Collect result in order
    std::vector<uint32_t> result;
    size_t cur = 0;
    while (cur != kNone) {
        auto it = data_.rank_map.find(token_bytes[cur]);
        if (it == data_.rank_map.end()) {
            throw std::runtime_error(
                "token bytes not found in vocabulary after merge");
        }
        result.push_back(it->second);
        cur = token_next[cur];
    }

    return result;
}

std::vector<uint32_t> Encoder::encode(const std::string& text) const {
    if (!pattern_) {
        return bpe_merge(text);
    }
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
    if (!pattern_) {
        return bpe_merge(text).size();
    }
    auto pieces = split_by_pattern(text);
    size_t total = 0;
    for (const auto& piece : pieces) {
        total += bpe_merge(piece).size();
    }
    return total;
}

} // namespace openai
} // namespace zazaki_tokenizer
