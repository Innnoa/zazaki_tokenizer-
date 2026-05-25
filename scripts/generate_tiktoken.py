#!/usr/bin/env python3
"""Generate .tiktoken binary files from the tiktoken Python package.

Binary format (all integers little-endian u32):
  u32: version (1)
  u32: regex_len
  u8[regex_len]: regex_pattern (utf-8, not null-terminated)
  u32: num_ranks
  For each rank:
    u32: token_bytes_len
    u8[token_bytes_len]: token_bytes
    u32: rank
  u32: num_special
  For each special:
    u32: name_len
    u8[name_len]: name (utf-8)
    u32: id
"""

import struct
import re
import sys
from pathlib import Path


def sanitize_pattern(pattern):
    """Remove regex features that RE2 does not support.

    Converts: ?+ -> ?, ++ -> +, *+ -> *, {n,m}+ -> {n,m}
    Removes: (?!...) negative lookahead assertions and their alternatives
    """
    # Remove possessive quantifiers
    pattern = re.sub(r'([?+*}])[+]', r'\1', pattern)
    # Remove negative/positive lookahead assertions and the preceding |
    # Pattern: |\s+(?!\S)  or similar constructs
    pattern = re.sub(r'\|?\s*\(\?[=!][^)]*\)', '', pattern)
    # Clean up double || or trailing |
    pattern = re.sub(r'\|+', '|', pattern)
    pattern = re.sub(r'\|$', '', pattern)
    return pattern


def write_u32(f, val):
    f.write(struct.pack("<I", val))


def load_encoding(enc_name):
    """Load vocabulary from tiktoken package."""
    import tiktoken
    enc = tiktoken.get_encoding(enc_name)
    ranks = enc._mergeable_ranks  # {bytes: rank}
    special = enc._special_tokens  # {str: id}
    pattern = enc._pat_str
    return ranks, special, pattern


def write_tiktoken(output_path, ranks, special, pattern):
    """Write a .tiktoken binary file."""
    with open(output_path, "wb") as f:
        # Version
        write_u32(f, 1)

        # Regex pattern
        pattern_bytes = pattern.encode("utf-8")
        write_u32(f, len(pattern_bytes))
        f.write(pattern_bytes)

        # Mergeable ranks
        write_u32(f, len(ranks))
        for token_bytes, rank in ranks.items():
            write_u32(f, len(token_bytes))
            f.write(token_bytes)
            write_u32(f, rank)

        # Special tokens
        write_u32(f, len(special))
        for name, token_id in special.items():
            name_bytes = name.encode("utf-8")
            write_u32(f, len(name_bytes))
            f.write(name_bytes)
            write_u32(f, token_id)


def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import tiktoken  # noqa: F401
    except ImportError:
        print("Error: tiktoken is not installed. Run: pip install tiktoken")
        sys.exit(1)

    encodings = [
        ("cl100k_base", data_dir / "cl100k_base.tiktoken"),
        ("o200k_base", data_dir / "o200k_base.tiktoken"),
    ]

    for enc_name, output_path in encodings:
        ranks, special, pattern = load_encoding(enc_name)
        pattern = sanitize_pattern(pattern)
        write_tiktoken(output_path, ranks, special, pattern)
        print(f"Generated {output_path} "
              f"({len(ranks)} ranks, {len(special)} special tokens)")


if __name__ == "__main__":
    main()
