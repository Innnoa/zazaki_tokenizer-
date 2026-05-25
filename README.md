# Zazaki Tokenizer

Fast token counting CLI and Python library for AI models. Built with C++17 and RE2.

## Quick Start

```bash
# Build from source
cmake -S . -B build -DBUILD_PYTHON=ON
cmake --build build -j$(nproc)
pip install -e .

# Or just install
pip install zazaki_tokenizer
```

## CLI Usage

```bash
# Count tokens
zazaki_tokenizer count "hello world"                     # 2 tokens
zazaki_tokenizer count -m gpt-4 "hello world"            # 2 tokens
zazaki_tokenizer count -f input.txt                      # from file
echo "text" | zazaki_tokenizer count                     # stdin

# Stream mode (real-time stdin)
tail -f log.txt | zazaki_tokenizer count --stream

# Encode / decode
zazaki_tokenizer encode "hello world"                    # [24912, 2375]
zazaki_tokenizer decode 24912 2375                      # hello world

# Chat messages (OpenAI API format)
zazaki_tokenizer chat -m gpt-4o <<< '[{"role":"user","content":"hi"}]'

# Download additional models
zazaki_tokenizer download deepseek-chat
zazaki_tokenizer download qwen2.5
zazaki_tokenizer download list
```

## Python API

```python
from zazaki_tokenizer import Tokenizer, count_tokens

# Quick count
count_tokens("hello world")                  # 2

# Full API
t = Tokenizer("gpt-4o")
t.count("hello world")                       # 2
t.encode("hello world").token_ids            # [24912, 2375]
t.decode([24912, 2375])                      # "hello world"
t.count_batch(["a", "b", "c"])               # [1, 1, 1]
```

## Supported Models

| Model | Status |
|-------|--------|
| gpt-4o / gpt-4o-mini / o1 / o3 | Built-in |
| gpt-4 / gpt-4-turbo / gpt-3.5-turbo | Built-in |
| deepseek-chat / deepseek-reasoner | `zazaki_tokenizer download` |
| qwen2.5 | `zazaki_tokenizer download` |
| llama3 | Manual HF download |

## Build from Source

Requirements: CMake 3.20+, C++17 compiler, Python 3.8+

```bash
# Configure (with tests)
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_PYTHON=ON

# Build
cmake --build build -j$(nproc)

# Test
ctest --test-dir build
python3 -m pytest tests/test_python.py -v
```

## Project Structure

```
zazaki_tokenizer/
├── zazaki_tokenizer/          # Python package
│   ├── __init__.py
│   └── cli.py
├── src/core/                  # C++ core (BPE encoder, registry)
│   └── openai/
├── src/cli/                   # C++ CLI binary
├── src/bindings/              # pybind11 Python bindings
├── scripts/                   # Vocab generation/conversion tools
├── data/                      # Embedded vocabulary files
├── tests/                     # C++ and Python tests
├── pyproject.toml
└── CMakeLists.txt
```

## License

MIT
