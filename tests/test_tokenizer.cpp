#include <gtest/gtest.h>

#include "tokenizer.h"

TEST(TokenizerTest, ForModelGpt4o) {
    auto t = zazaki_tokenizer::Tokenizer::for_model("gpt-4o");
    auto n = t.count_tokens("hello world");
    EXPECT_EQ(n, 2u);
}

TEST(TokenizerTest, ForModelGpt4) {
    auto t = zazaki_tokenizer::Tokenizer::for_model("gpt-4");
    auto n = t.count_tokens("hello world");
    EXPECT_EQ(n, 2u);
}

TEST(TokenizerTest, UnknownModel) {
    EXPECT_THROW(
        zazaki_tokenizer::Tokenizer::for_model("nonexistent-model"),
        std::invalid_argument
    );
}

TEST(TokenizerTest, EncodeReturnsCorrectSizes) {
    auto t = zazaki_tokenizer::Tokenizer::for_model("gpt-4o");
    auto result = t.encode("hello world");
    EXPECT_EQ(result.token_ids.size(), result.token_count);
    EXPECT_EQ(result.token_count, 2u);
}
