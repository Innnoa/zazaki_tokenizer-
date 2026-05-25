#include <gtest/gtest.h>

#include "openai/encoder.h"
#include "openai/tiktoken_file.h"

extern const unsigned char cl100k_base_tiktoken[];
extern const unsigned int cl100k_base_tiktoken_len;

class EncoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto data = zazaki_tokenizer::openai::parse_tiktoken(
            cl100k_base_tiktoken, cl100k_base_tiktoken_len);
        encoder_ = std::make_unique<zazaki_tokenizer::openai::Encoder>(std::move(data));
    }
    std::unique_ptr<zazaki_tokenizer::openai::Encoder> encoder_;
};

TEST_F(EncoderTest, EncodeEmptyString) {
    auto ids = encoder_->encode("");
    EXPECT_TRUE(ids.empty());
}

TEST_F(EncoderTest, EncodeHelloWorld) {
    auto ids = encoder_->encode("hello world");
    EXPECT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 15339u);
    EXPECT_EQ(ids[1], 1917u);
}

TEST_F(EncoderTest, CountHelloWorld) {
    EXPECT_EQ(encoder_->count("hello world"), 2u);
}

TEST_F(EncoderTest, DecodeRoundtrip) {
    std::string text = "hello world";
    auto ids = encoder_->encode(text);
    std::string decoded = encoder_->decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST_F(EncoderTest, CountVsEncodeSize) {
    std::string text = "The quick brown fox jumps over the lazy dog.";
    auto ids = encoder_->encode(text);
    EXPECT_EQ(encoder_->count(text), ids.size());
}

TEST_F(EncoderTest, EncodeChinese) {
    auto ids = encoder_->encode("\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c");
    EXPECT_GT(ids.size(), 0u);
    auto decoded = encoder_->decode(ids);
    EXPECT_EQ(decoded, "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c");
}

TEST_F(EncoderTest, DecodeOutOfRange) {
    EXPECT_THROW(encoder_->decode({999999999u}), std::out_of_range);
}
