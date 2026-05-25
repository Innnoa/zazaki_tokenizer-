#!/usr/bin/env python3
"""CLI entry point for zazaki_tokenizer."""

import argparse
import json
import sys
import os
from pathlib import Path

from zazaki_tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(
        prog="zazaki_tokenizer",
        description="Fast token counting for OpenAI models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # count
    p_count = sub.add_parser("count", help="Count tokens in text")
    p_count.add_argument("text", nargs="*", default=[], help="Text to tokenize")
    p_count.add_argument("-m", "--model", default="gpt-4o", help="Model name")
    p_count.add_argument("-f", "--file", help="Read input from file")
    p_count.add_argument("--json", action="store_true", help="Output JSON")

    # encode
    p_enc = sub.add_parser("encode", help="Encode text to token IDs")
    p_enc.add_argument("text", nargs="*", default=[], help="Text to encode")
    p_enc.add_argument("-m", "--model", default="gpt-4o", help="Model name")
    p_enc.add_argument("-f", "--file", help="Read input from file")
    p_enc.add_argument("--json", action="store_true", help="Output JSON")

    # decode
    p_dec = sub.add_parser("decode", help="Decode token IDs to text")
    p_dec.add_argument("ids", nargs="+", type=int, help="Token IDs to decode")
    p_dec.add_argument("-m", "--model", default="gpt-4o", help="Model name")
    p_dec.add_argument("--json", action="store_true", help="Output JSON")

    args = parser.parse_args()

    try:
        t = Tokenizer(args.model)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    def read_input():
        if hasattr(args, "file") and args.file:
            return Path(args.file).read_text()
        if hasattr(args, "text") and args.text:
            return " ".join(args.text)
        return sys.stdin.read()

    if args.command == "count":
        text = read_input()
        n = t.count(text)
        if args.json:
            print(json.dumps({"tokens": n}))
        else:
            print(f"{n} tokens")

    elif args.command == "encode":
        text = read_input()
        result = t.encode(text)
        if args.json:
            print(json.dumps({
                "token_ids": result.token_ids,
                "count": result.token_count,
            }))
        else:
            print(result.token_ids)

    elif args.command == "decode":
        text = t.decode(args.ids)
        if args.json:
            print(json.dumps({"text": text}))
        else:
            print(text)


if __name__ == "__main__":
    sys.exit(main())
