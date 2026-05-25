"""Zazaki Tokenizer — fast token counting for OpenAI models."""

from ._zazaki_tokenizer import Tokenizer, count_tokens, EncodingResult

__all__ = ["Tokenizer", "count_tokens", "EncodingResult"]
