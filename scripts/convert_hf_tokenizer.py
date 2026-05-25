#!/usr/bin/env python3
"""Convert HuggingFace tokenizer.json to zazaki .tiktoken format.

Supports BPE tokenizers (DeepSeek, Llama 3, Qwen, etc.).

Usage:
    python3 convert_hf_tokenizer.py deepseek-ai/DeepSeek-V3 --output data/deepseek_v3.tiktoken
    python3 convert_hf_tokenizer.py meta-llama/Meta-Llama-3-8B --output data/llama3.tiktoken
    python3 convert_hf_tokenizer.py Qwen/Qwen2.5-7B --output data/qwen25.tiktoken
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path


def write_u32(f, val):
    f.write(struct.pack("<I", val))


def sanitize_pattern(pattern):
    pattern = re.sub(r'([?+*}])[+]', r'\1', pattern)
    pattern = re.sub(r'\(\?[=!][^)]*\)', '', pattern)
    pattern = re.sub(r'\|+', '|', pattern)
    pattern = re.sub(r'\|$', '', pattern)
    return pattern


def download_tokenizer_json(model_id):
    """Download tokenizer.json from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model_id, "tokenizer.json")
        with open(path) as f:
            return json.load(f)
    except ImportError:
        import urllib.request
        url = f"https://huggingface.co/{model_id}/resolve/main/tokenizer.json"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())


def build_byte_map():
    """Build the standard HuggingFace ByteLevel byte-to-character mapping.

    Returns: dict mapping byte value (0-255) to its character representation.
    """
    # Bytes that keep their code point as representation
    direct = list(range(ord("!"), ord("~") + 1))  # 33-126
    direct += list(range(ord("\xa1"), ord("\xac") + 1))  # 161-172
    direct += list(range(ord("\xae"), ord("\xff") + 1))  # 174-255

    byte_map = {}
    for b in direct:
        byte_map[b] = chr(b)

    # Remaining bytes map to chr(256+n) sequentially
    n = 0
    for b in range(256):
        if b not in byte_map:
            byte_map[b] = chr(256 + n)
            n += 1

    return byte_map


def token_str_to_bytes(token_str, byte_map):
    """Convert a BPE token string to raw bytes using the byte map."""
    # Build reverse map: char -> byte
    char_to_byte = {c: b for b, c in byte_map.items()}
    result = bytearray()
    for ch in token_str:
        if ch in char_to_byte:
            result.append(char_to_byte[ch])
        else:
            result.extend(ch.encode("utf-8"))
    return bytes(result)


def extract_bpe_vocab(tok_json):
    """Extract vocabulary and special tokens from HuggingFace tokenizer JSON."""
    model = tok_json.get("model", {})
    vocab = model.get("vocab", {})

    if not vocab:
        raise ValueError("No BPE vocabulary found in tokenizer.json")

    byte_map = build_byte_map()

    rank_map = {}
    max_rank = 0
    for token_str, token_id in vocab.items():
        token_bytes = token_str_to_bytes(token_str, byte_map)
        rank_map[token_bytes] = token_id
        if token_id > max_rank:
            max_rank = token_id

    # Ensure all 256 byte values have a rank (required for BPE)
    for b in range(256):
        key = bytes([b])
        if key not in rank_map:
            max_rank += 1
            rank_map[key] = max_rank

    special_tokens = {}
    for entry in tok_json.get("added_tokens", []):
        name = entry.get("content", "")
        tid = entry.get("id", 0)
        if name and tid > max_rank:
            special_tokens[name] = tid

    # Determine if this is a byte-level tokenizer (no regex splitting needed)
    pattern = "."
    pre_tok = tok_json.get("pre_tokenizer", {})
    is_byte_level = False
    if pre_tok and pre_tok.get("type") == "Sequence":
        for pretok in pre_tok.get("pretokenizers", []):
            if pretok.get("type") == "ByteLevel":
                if not pretok.get("use_regex", True):
                    is_byte_level = True
    if is_byte_level:
        pattern = "."

    return rank_map, special_tokens, pattern


def write_tiktoken(output_path, rank_map, special_tokens, pattern):
    pattern = sanitize_pattern(pattern)
    with open(output_path, "wb") as f:
        write_u32(f, 1)
        pattern_bytes = pattern.encode("utf-8")
        write_u32(f, len(pattern_bytes))
        f.write(pattern_bytes)
        write_u32(f, len(rank_map))
        for token_bytes, rank in rank_map.items():
            write_u32(f, len(token_bytes))
            f.write(token_bytes)
            write_u32(f, rank)
        write_u32(f, len(special_tokens))
        for name, tid in special_tokens.items():
            nb = name.encode("utf-8")
            write_u32(f, len(nb))
            f.write(nb)
            write_u32(f, tid)


def main():
    parser = argparse.ArgumentParser(description="Convert HF tokenizer.json to .tiktoken")
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument("--output", "-o", required=True, help="Output .tiktoken file")
    parser.add_argument("--local", help="Use local tokenizer.json file instead of downloading")
    args = parser.parse_args()

    if args.local:
        with open(args.local) as f:
            tok_json = json.load(f)
    else:
        print(f"Downloading tokenizer.json for {args.model_id}...")
        tok_json = download_tokenizer_json(args.model_id)

    rank_map, special_tokens, pattern = extract_bpe_vocab(tok_json)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_tiktoken(output, rank_map, special_tokens, pattern)

    print(f"Generated {output}")
    print(f"  ranks: {len(rank_map)}")
    print(f"  special tokens: {len(special_tokens)}")
    print(f"  pattern: {pattern[:80]}...")


if __name__ == "__main__":
    main()
