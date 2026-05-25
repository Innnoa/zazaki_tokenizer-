"""Zazaki Tokenizer — fast token counting for OpenAI models."""

from ._zazaki_tokenizer import (
    Tokenizer,
    count_tokens,
    EncodingResult,
    register_model,
    has_model,
)

__all__ = [
    "Tokenizer",
    "count_tokens",
    "EncodingResult",
    "register_model",
    "has_model",
]
