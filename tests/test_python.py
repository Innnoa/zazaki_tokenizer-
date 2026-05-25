"""Python binding tests. Run with pytest."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "bindings"))

try:
    from _zazaki_tokenizer import Tokenizer, count_tokens
except ImportError:
    pytest.skip("Python bindings not built", allow_module_level=True)


class TestTokenizer:
    def test_default_model(self):
        n = count_tokens("hello world")
        assert n == 2

    def test_gpt4o_count(self):
        t = Tokenizer("gpt-4o")
        assert t.count("hello world") == 2

    def test_gpt4o_encode(self):
        t = Tokenizer("gpt-4o")
        result = t.encode("hello world")
        assert result.token_ids == [24912, 2375]
        assert result.token_count == 2

    def test_decode_roundtrip(self):
        t = Tokenizer("gpt-4o")
        text = "hello world"
        result = t.encode(text)
        assert t.decode(result.token_ids) == text

    def test_count_batch(self):
        t = Tokenizer("gpt-4o")
        counts = t.count_batch(["hello", "world", "test"])
        assert counts == [1, 1, 1]

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            Tokenizer("nonexistent-model")
