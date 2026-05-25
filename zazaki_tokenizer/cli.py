#!/usr/bin/env python3
"""CLI entry point for zazaki_tokenizer."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from zazaki_tokenizer import Tokenizer, register_model, has_model


KNOWN_MODELS = {
    "deepseek-chat": "deepseek-ai/DeepSeek-V3",
    "deepseek-reasoner": "deepseek-ai/DeepSeek-R1",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
}

CACHE_DIR = Path.home() / ".cache" / "zazaki_tokenizer"


def count_chat_messages(messages, model):
    """Count tokens in OpenAI-format chat messages.

    Includes per-message and per-role overhead tokens,
    matching the official OpenAI token counting formula.
    """
    t = Tokenizer(model)

    if model in ("gpt-3.5-turbo-0301",):
        tokens_per_message = 4
        tokens_per_name = -1
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    total = 0
    for msg in messages:
        total += tokens_per_message
        for key, value in msg.items():
            total += t.count(str(value))
            if key == "name":
                total += tokens_per_name
    total += 3  # assistant priming
    return total


def cmd_download(args):
    if not args.model or args.model == "list":
        print("Built-in models: gpt-4o, gpt-4, gpt-4-turbo, gpt-3.5-turbo")
        print()
        print("Downloadable models:")
        for name, hf_id in KNOWN_MODELS.items():
            status = "✓" if has_model(name) else " "
            print(f"  [{status}] {name}  ({hf_id})")
        print(f"\nCache: {CACHE_DIR}")
        return 0

    if args.hf_id:
        hf_id = args.hf_id
        name = args.model
    elif args.model in KNOWN_MODELS:
        hf_id = KNOWN_MODELS[args.model]
        name = args.model
    else:
        print(f"Unknown model: {args.model}", file=sys.stderr)
        print("Use --hf-id to specify a HuggingFace model ID", file=sys.stderr)
        return 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output = CACHE_DIR / f"{name}.tiktoken"

    script_dir = Path(__file__).resolve().parent.parent / "scripts"
    script = script_dir / "convert_hf_tokenizer.py"
    result = subprocess.run(
        [sys.executable, str(script), hf_id, "-o", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Download failed:\n{result.stderr}", file=sys.stderr)
        return 1

    register_model(name, str(output))
    print(f"✓ Registered: {name} -> {hf_id}")
    print(f"  Vocabulary: {output}")
    return 0


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
    p_count.add_argument("--stream", action="store_true",
                         help="Stream stdin line by line (cumulative count)")

    # chat
    p_chat = sub.add_parser(
        "chat", help="Count tokens in OpenAI chat messages (JSON array)")
    p_chat.add_argument("-m", "--model", default="gpt-4o", help="Model name")
    p_chat.add_argument("-f", "--file", help="Read messages from JSON file")
    p_chat.add_argument("--json", action="store_true", help="Output JSON")

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

    # download
    p_dl = sub.add_parser("download", help="Download vocabulary for a model")
    p_dl.add_argument("model", nargs="?", help="Model name or 'list'")
    p_dl.add_argument("--hf-id", help="HuggingFace model ID (if not in known list)")

    args = parser.parse_args()

    if args.command == "download":
        return cmd_download(args)

    if args.command != "chat":
        try:
            t = Tokenizer(args.model)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
        t = None

    def read_full():
        if hasattr(args, "file") and args.file:
            return Path(args.file).read_text()
        if hasattr(args, "text") and args.text:
            return " ".join(args.text)
        return sys.stdin.read()

    if args.command == "count":
        if args.stream and not args.file and not args.text:
            total = 0
            line_count = 0
            for line in sys.stdin:
                total += t.count(line)
                line_count += 1
                if args.json:
                    print(json.dumps({"line": line_count, "cumulative": total}),
                          flush=True)
                else:
                    print(f"{total} tokens (lines: {line_count})", flush=True)
        else:
            text = read_full()
            n = t.count(text)
            if args.json:
                print(json.dumps({"tokens": n}))
            else:
                print(f"{n} tokens")

    elif args.command == "chat":
        raw = args.file and Path(args.file).read_text() or sys.stdin.read()
        messages = json.loads(raw)
        n = count_chat_messages(messages, args.model)
        if args.json:
            print(json.dumps({"tokens": n, "messages": len(messages)}))
        else:
            print(f"{n} tokens ({len(messages)} messages)")

    elif args.command == "encode":
        text = read_full()
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
