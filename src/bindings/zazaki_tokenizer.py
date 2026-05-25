"""Zazaki Tokenizer - fast token counting for OpenAI models."""

try:
    from ._zazaki_tokenizer import Tokenizer, count_tokens, EncodingResult
except ImportError:
    from _zazaki_tokenizer import Tokenizer, count_tokens, EncodingResult  # noqa: F811

__all__ = ["Tokenizer", "count_tokens", "EncodingResult"]
