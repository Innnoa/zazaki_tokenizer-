#include <gtest/gtest.h>

#include "registry.h"

TEST(RegistryTest, Cl100kModelsAvailable) {
    auto* enc = zazaki_tokenizer::Registry::instance().get_encoder("gpt-4");
    ASSERT_NE(enc, nullptr);
    EXPECT_EQ(enc->count("hello"), 1u);
}

TEST(RegistryTest, O200kModelsAvailable) {
    auto* enc = zazaki_tokenizer::Registry::instance().get_encoder("gpt-4o");
    ASSERT_NE(enc, nullptr);
    EXPECT_EQ(enc->count("hello"), 1u);
}

TEST(RegistryTest, Gpt4oMiniSharesO200k) {
    auto* a = zazaki_tokenizer::Registry::instance().get_encoder("gpt-4o");
    auto* b = zazaki_tokenizer::Registry::instance().get_encoder("gpt-4o-mini");
    EXPECT_EQ(a, b);
}
