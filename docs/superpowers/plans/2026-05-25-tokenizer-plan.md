# Zazaki Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++17 token counting/encoding/decoding library for OpenAI models with CLI and Python bindings.

**Architecture:** Layered design — `libzazaki_tokenizer` static library (core parsing, BPE encoder, registry), linked by both CLI binary and pybind11 Python module. `.tiktoken` vocabulary files embedded at compile time via `xxd -i`.

**Tech Stack:** C++17, CMake, pybind11, RE2, GoogleTest (test only)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `CMakeLists.txt`
- Create: `src/core/CMakeLists.txt`
- Create: `src/cli/CMakeLists.txt`
- Create: `src/bindings/CMakeLists.txt`
- Create: `tests/CMakeLists.txt`
- Create: `scripts/generate_tiktoken.py`

- [ ] **Step 1: Create root CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.20)
project(zazaki_tokenizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_CLI "Build CLI tool" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" OFF)

# Fetch pybind11 if Python bindings enabled
if(BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Fetch RE2 for regex pre-tokenization
include(FetchContent)
FetchContent_Declare(
    re2
    GIT_REPOSITORY https://github.com/google/re2.git
    GIT_TAG 2024-07-02
)
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(re2)

# Fetch GoogleTest if tests enabled
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_subdirectory(src/core)
if(BUILD_CLI)
    add_subdirectory(src/cli)
endif()
if(BUILD_PYTHON)
    add_subdirectory(src/bindings)
endif()
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

- [ ] **Step 2: Create src/core/CMakeLists.txt**

```cmake
# Embed .tiktoken files as C arrays
set(TIKTOKEN_FILES
    ${CMAKE_SOURCE_DIR}/data/cl100k_base.tiktoken
    ${CMAKE_SOURCE_DIR}/data/o200k_base.tiktoken
)

foreach(tik_file ${TIKTOKEN_FILES})
    get_filename_component(tik_name ${tik_file} NAME_WE)
    set(embed_file ${CMAKE_CURRENT_BINARY_DIR}/${tik_name}_data.c)
    add_custom_command(
        OUTPUT ${embed_file}
        COMMAND xxd -i ${tik_file} ${embed_file}
        DEPENDS ${tik_file}
        COMMENT "Embedding ${tik_name}.tiktoken"
    )
    list(APPEND EMBED_SOURCES ${embed_file})
endforeach()

add_library(zazaki_tokenizer STATIC
    tokenizer.cpp
    registry.cpp
    openai/tiktoken_file.cpp
    openai/encoder.cpp
    ${EMBED_SOURCES}
)

target_include_directories(zazaki_tokenizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(zazaki_tokenizer PUBLIC re2::re2)
```

- [ ] **Step 3: Create src/cli/CMakeLists.txt**

```cmake
add_executable(zazaki_tokenizer main.cpp)
target_link_libraries(zazaki_tokenizer PRIVATE zazaki_tokenizer)
```

- [ ] **Step 4: Create src/bindings/CMakeLists.txt**

```cmake
pybind11_add_module(_zazaki_tokenizer python_bindings.cpp)
target_link_libraries(_zazaki_tokenizer PRIVATE zazaki_tokenizer)
```

- [ ] **Step 5: Create tests/CMakeLists.txt**

```cmake
add_executable(test_zazaki_tokenizer
    test_encoder.cpp
    test_tokenizer.cpp
    test_registry.cpp
)
target_link_libraries(test_zazaki_tokenizer PRIVATE zazaki_tokenizer GTest::gtest GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_zazaki_tokenizer)
```

- [ ] **Step 6: Create directory structure**

Run:
```bash
mkdir -p /home/zazaki/Projects/zazaki_tokenizer/src/core/openai
mkdir -p /home/zazaki/Projects/zazaki_tokenizer/src/cli
mkdir -p /home/zazaki/Projects/zazaki_tokenizer/src/bindings
mkdir -p /home/zazaki/Projects/zazaki_tokenizer/tests
mkdir -p /home/zazaki/Projects/zazaki_tokenizer/data
mkdir -p /home/zazaki/Projects/zazaki_tokenizer/scripts
```

Expected: directories created without error.

- [ ] **Step 7: Create .gitignore**

Write `/.gitignore`:
```
build/
*.pyc
__pycache__/
.pytest_cache/
```

- [ ] **Step 8: Commit**

```bash
cd /home/zazaki/Projects/zazaki_tokenizer && git init && git add -A && git commit -m "chore: project scaffolding with CMake build system"
```

---

### Task 2: Generate .tiktoken Vocabulary Files

**Files:**
- Create: `scripts/generate_tiktoken.py`
- Create: `data/cl100k_base.tiktoken` (generated)
- Create: `data/o200k_base.tiktoken` (generated)

- [ ] **Step 1: Write the generation script**

Write `scripts/generate_tiktoken.py`:

```python
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
import sys
from pathlib import Path


def write_u32(f, val):
    f.write(struct.pack("<I", val))


def load_encoding(enc_name):
    """Load vocabulary from tiktoken package."""
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

    # Check if tiktoken is installed
    try:
        import tiktoken
    except ImportError:
        print("Error: tiktoken is not installed. Run: pip install tiktoken")
        sys.exit(1)

    encodings = [
        ("cl100k_base", data_dir / "cl100k_base.tiktoken"),
        ("o200k_base", data_dir / "o200k_base.tiktoken"),
    ]

    for enc_name, output_path in encodings:
        ranks, special, pattern = load_encoding(enc_name)
        write_tiktoken(output_path, ranks, special, pattern)
        print(f"Generated {output_path} "
              f"({len(ranks)} ranks, {len(special)} special tokens)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run generation script**

Run:
```bash
pip install tiktoken && python3 /home/zazaki/Projects/zazaki_tokenizer/scripts/generate_tiktoken.py
```

Expected: Two files created (`data/cl100k_base.tiktoken`, `data/o200k_base.tiktoken`), each ~1-2 MB.

- [ ] **Step 3: Verify generated files**

Run:
```bash
python3 -c "
import struct
with open('/home/zazaki/Projects/zazaki_tokenizer/data/cl100k_base.tiktoken','rb') as f:
    ver = struct.unpack('<I', f.read(4))[0]
    regex_len = struct.unpack('<I', f.read(4))[0]
    regex = f.read(regex_len).decode('utf-8')
    num_ranks = struct.unpack('<I', f.read(4))[0]
    print(f'version={ver}, regex_len={regex_len}, num_ranks={num_ranks}')
    print(f'regex: {regex[:80]}...')
"
```

Expected output like:
```
version=1, regex_len=..., num_ranks=100256
regex: '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\...
```

- [ ] **Step 4: Commit**

```bash
git add scripts/ data/ && git commit -m "feat: add tiktoken vocabulary generation script and data files"
```

---

### Task 3: Tiktoken File Parser (C++)

**Files:**
- Create: `src/core/openai/tiktoken_file.h`
- Create: `src/core/openai/tiktoken_file.cpp`

- [ ] **Step 1: Write the header**

Write `src/core/openai/tiktoken_file.h`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace zazaki_tokenizer {
namespace openai {

struct TiktokenData {
    std::string regex_pattern;
    std::unordered_map<std::string, uint32_t> rank_map;
    std::unordered_map<std::string, uint32_t> special_tokens;
    std::vector<std::string> decode_map;
};

// Parse a .tiktoken binary blob (embedded via xxd -i).
// The blob is (const unsigned char*, unsigned int) matching xxd output.
TiktokenData parse_tiktoken(const unsigned char* data, unsigned int len);

} // namespace openai
} // namespace zazaki_tokenizer
```

- [ ] **Step 2: Write the implementation**

Write `src/core/openai/tiktoken_file.cpp`:

```cpp
#include "openai/tiktoken_file.h"
#include <cstring>
#include <stdexcept>

namespace zazaki_tokenizer {
namespace openai {

namespace {

uint32_t read_u32(const unsigned char*& p, const unsigned char* end) {
    if (p + 4 > end) {
        throw std::runtime_error("tiktoken file truncated at u32 read");
    }
    uint32_t val = static_cast<uint32_t>(p[0])
                 | (static_cast<uint32_t>(p[1]) << 8)
                 | (static_cast<uint32_t>(p[2]) << 16)
                 | (static_cast<uint32_t>(p[3]) << 24);
    p += 4;
    return val;
}

} // namespace

TiktokenData parse_tiktoken(const unsigned char* data, unsigned int len) {
    const unsigned char* p = data;
    const unsigned char* end = data + len;

    TiktokenData result;

    uint32_t version = read_u32(p, end);
    if (version != 1) {
        throw std::runtime_error("unsupported tiktoken version: " + std::to_string(version));
    }

    // Regex pattern
    uint32_t regex_len = read_u32(p, end);
    if (p + regex_len > end) {
        throw std::runtime_error("tiktoken file truncated at regex");
    }
    result.regex_pattern.assign(reinterpret_cast<const char*>(p), regex_len);
    p += regex_len;

    // Mergeable ranks
    uint32_t num_ranks = read_u32(p, end);
    result.decode_map.resize(num_ranks);
    for (uint32_t i = 0; i < num_ranks; ++i) {
        uint32_t token_len = read_u32(p, end);
        if (p + token_len > end) {
            throw std::runtime_error("tiktoken file truncated at token bytes");
        }
        std::string token_bytes(reinterpret_cast<const char*>(p), token_len);
        p += token_len;

        uint32_t rank = read_u32(p, end);
        result.rank_map[std::move(token_bytes)] = rank;
        if (rank < result.decode_map.size()) {
            result.decode_map[rank] = result.rank_map.rbegin()->first;
        }
    }

    // Fix decode_map: use the reverse lookup from rank_map
    result.decode_map.clear();
    result.decode_map.resize(result.rank_map.size());
    for (const auto& [token_bytes, rank] : result.rank_map) {
        if (rank < result.decode_map.size()) {
            result.decode_map[rank] = token_bytes;
        }
    }

    // Special tokens
    uint32_t num_special = read_u32(p, end);
    for (uint32_t i = 0; i < num_special; ++i) {
        uint32_t name_len = read_u32(p, end);
        if (p + name_len > end) {
            throw std::runtime_error("tiktoken file truncated at special token name");
        }
        std::string name(reinterpret_cast<const char*>(p), name_len);
        p += name_len;

        uint32_t id = read_u32(p, end);
        result.special_tokens[std::move(name)] = id;
    }

    return result;
}

} // namespace openai
} // namespace zazaki_tokenizer
```

- [ ] **Step 3: Commit**

```bash
git add src/core/openai/ && git commit -m "feat: add .tiktoken binary file parser"
```

---

### Task 4: BPE Encoder

**Files:**
- Create: `src/core/openai/encoder.h`
- Create: `src/core/openai/encoder.cpp`

- [ ] **Step 1: Write the header**

Write `src/core/openai/encoder.h`:

```cpp
#pragma once

#include "openai/tiktoken_file.h"
#include <memory>
#include <string>
#include <vector>

namespace re2 {
class RE2;
}

namespace zazaki_tokenizer {
namespace openai {

class Encoder {
public:
    explicit Encoder(TiktokenData data);

    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& ids) const;
    size_t count(const std::string& text) const;

private:
    struct Piece {
        std::string bytes;
        uint32_t rank;
    };

    std::vector<std::string> split_by_pattern(const std::string& text) const;
    std::vector<uint32_t> bpe_merge(const std::string& piece_bytes) const;

    TiktokenData data_;
    std::unique_ptr<re2::RE2> pattern_;
};

} // namespace openai
} // namespace zazaki_tokenizer
```

- [ ] **Step 2: Write the implementation**

Write `src/core/openai/encoder.cpp`:

```cpp
#include "openai/encoder.h"

#include <re2/re2.h>
#include <algorithm>
#include <limits>

namespace zazaki_tokenizer {
namespace openai {

Encoder::Encoder(TiktokenData data) : data_(std::move(data)) {
    pattern_ = std::make_unique<re2::RE2>(data_.regex_pattern);
    if (!pattern_->ok()) {
        throw std::runtime_error("invalid regex pattern: " + pattern_->error());
    }
}

std::vector<std::string> Encoder::split_by_pattern(const std::string& text) const {
    std::vector<std::string> pieces;
    re2::StringPiece input(text);
    re2::StringPiece match;

    size_t last_end = 0;
    while (re2::RE2::FindAndConsume(&input, *pattern_, &match)) {
        // Any text between the last match and this match goes as a separate piece?
        // The tiktoken regex pattern is designed to consume everything —
        // it shouldn't produce gaps. But we handle gaps just in case.
        size_t match_start = match.data() - text.data();
        if (match_start > last_end) {
            pieces.emplace_back(text.substr(last_end, match_start - last_end));
        }
        pieces.emplace_back(match.as_string());
        last_end = match_start + match.size();
    }

    if (last_end < text.size()) {
        pieces.emplace_back(text.substr(last_end));
    }

    return pieces;
}

std::vector<uint32_t> Encoder::bpe_merge(const std::string& piece_bytes) const {
    if (piece_bytes.empty()) {
        return {};
    }

    // Each element: (bytes, rank)
    std::vector<Piece> parts;
    parts.reserve(piece_bytes.size());

    for (unsigned char c : piece_bytes) {
        std::string byte_str(1, static_cast<char>(c));
        auto it = data_.rank_map.find(byte_str);
        if (it == data_.rank_map.end()) {
            throw std::runtime_error("byte not found in vocabulary: " + std::to_string(c));
        }
        parts.push_back({std::move(byte_str), it->second});
    }

    if (parts.size() == 1) {
        return {parts[0].rank};
    }

    while (true) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        size_t best_idx = 0;

        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            std::string merged_bytes = parts[i].bytes + parts[i + 1].bytes;
            auto it = data_.rank_map.find(merged_bytes);
            if (it != data_.rank_map.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_rank == std::numeric_limits<uint32_t>::max()) {
            break;
        }

        parts[best_idx].bytes += parts[best_idx + 1].bytes;
        parts[best_idx].rank = best_rank;
        parts.erase(parts.begin() + static_cast<long>(best_idx + 1));
    }

    std::vector<uint32_t> result;
    result.reserve(parts.size());
    for (const auto& p : parts) {
        result.push_back(p.rank);
    }
    return result;
}

std::vector<uint32_t> Encoder::encode(const std::string& text) const {
    auto pieces = split_by_pattern(text);
    std::vector<uint32_t> result;
    for (const auto& piece : pieces) {
        auto tokens = bpe_merge(piece);
        result.insert(result.end(), tokens.begin(), tokens.end());
    }
    return result;
}

std::string Encoder::decode(const std::vector<uint32_t>& ids) const {
    std::string result;
    for (uint32_t id : ids) {
        if (id >= data_.decode_map.size()) {
            throw std::out_of_range("token id out of range: " + std::to_string(id));
        }
        result += data_.decode_map[id];
    }
    return result;
}

size_t Encoder::count(const std::string& text) const {
    auto pieces = split_by_pattern(text);
    size_t total = 0;
    for (const auto& piece : pieces) {
        total += bpe_merge(piece).size();
    }
    return total;
}

} // namespace openai
} // namespace zazaki_tokenizer
```

- [ ] **Step 3: Commit**

```bash
git add src/core/openai/encoder.h src/core/openai/encoder.cpp && git commit -m "feat: BPE encoder with regex pre-tokenization"
```

---

### Task 5: Registry

**Files:**
- Create: `src/core/registry.h`
- Create: `src/core/registry.cpp`

- [ ] **Step 1: Write the header**

Write `src/core/registry.h`:

```cpp
#pragma once

#include "openai/encoder.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace zazaki_tokenizer {

using EncoderPtr = std::shared_ptr<openai::Encoder>;

class Registry {
public:
    static Registry& instance();

    const openai::Encoder* get_encoder(const std::string& model_name) const;

private:
    Registry();
    std::unordered_map<std::string, EncoderPtr> models_;
};

} // namespace zazaki_tokenizer
```

- [ ] **Step 2: Write the implementation**

Write `src/core/registry.cpp`:

```cpp
#include "registry.h"
#include <stdexcept>

extern const unsigned char cl100k_base_tiktoken[];
extern const unsigned int cl100k_base_tiktoken_len;

extern const unsigned char o200k_base_tiktoken[];
extern const unsigned int o200k_base_tiktoken_len;

namespace zazaki_tokenizer {

Registry::Registry() {
    auto cl100k_data = openai::parse_tiktoken(cl100k_base_tiktoken, cl100k_base_tiktoken_len);
    auto cl100k = std::make_shared<openai::Encoder>(std::move(cl100k_data));

    models_["gpt-4"] = cl100k;
    models_["gpt-4-turbo"] = cl100k;
    models_["gpt-3.5-turbo"] = cl100k;
    models_["text-embedding-ada-002"] = cl100k;

    auto o200k_data = openai::parse_tiktoken(o200k_base_tiktoken, o200k_base_tiktoken_len);
    auto o200k = std::make_shared<openai::Encoder>(std::move(o200k_data));

    models_["gpt-4o"] = o200k;
    models_["gpt-4o-mini"] = o200k;
    models_["o1"] = o200k;
    models_["o3"] = o200k;
}

Registry& Registry::instance() {
    static Registry reg;
    return reg;
}

const openai::Encoder* Registry::get_encoder(const std::string& model_name) const {
    auto it = models_.find(model_name);
    if (it == models_.end()) {
        throw std::invalid_argument("unknown model: " + model_name);
    }
    return it->second.get();
}

} // namespace zazaki_tokenizer
```

- [ ] **Step 3: Commit**

```bash
git add src/core/registry.h src/core/registry.cpp && git commit -m "feat: model registry with embedded vocabulary loading"
```

---

### Task 6: Tokenizer Public API

**Files:**
- Create: `src/core/tokenizer.h`
- Create: `src/core/tokenizer.cpp`

- [ ] **Step 1: Write the header**

Write `src/core/tokenizer.h`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace zazaki_tokenizer {

struct EncodingResult {
    std::vector<uint32_t> token_ids;
    size_t token_count;
};

class Tokenizer {
public:
    static Tokenizer for_model(const std::string& model_name);

    EncodingResult encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& token_ids) const;
    size_t count_tokens(const std::string& text) const;

private:
    explicit Tokenizer(const std::string& model_name);
    std::string model_name_;
};

} // namespace zazaki_tokenizer
```

- [ ] **Step 2: Write the implementation**

Write `src/core/tokenizer.cpp`:

```cpp
#include "tokenizer.h"
#include "registry.h"

namespace zazaki_tokenizer {

Tokenizer::Tokenizer(const std::string& model_name)
    : model_name_(model_name) {}

Tokenizer Tokenizer::for_model(const std::string& model_name) {
    // Validate model exists by looking it up
    Registry::instance().get_encoder(model_name);
    return Tokenizer(model_name);
}

EncodingResult Tokenizer::encode(const std::string& text) const {
    auto* enc = Registry::instance().get_encoder(model_name_);
    auto ids = enc->encode(text);
    return {std::move(ids), ids.size()};
}

std::string Tokenizer::decode(const std::vector<uint32_t>& token_ids) const {
    auto* enc = Registry::instance().get_encoder(model_name_);
    return enc->decode(token_ids);
}

size_t Tokenizer::count_tokens(const std::string& text) const {
    auto* enc = Registry::instance().get_encoder(model_name_);
    return enc->count(text);
}

} // namespace zazaki_tokenizer
```

- [ ] **Step 3: Commit**

```bash
git add src/core/tokenizer.h src/core/tokenizer.cpp && git commit -m "feat: Tokenizer public API with encode/decode/count"
```

---

### Task 7: CLI Application

**Files:**
- Create: `src/cli/main.cpp`

- [ ] **Step 1: Write CLI implementation**

Write `src/cli/main.cpp`:

```cpp
#include "tokenizer.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(std::ostream& os, const char* prog) {
    os << "Usage: " << prog << " <command> [options] [text]\n\n"
       << "Commands:\n"
       << "  count    Count tokens in text\n"
       << "  encode   Encode text to token IDs\n"
       << "  decode   Decode token IDs to text\n\n"
       << "Options:\n"
       << "  -m, --model <name>   Model name (default: gpt-4o)\n"
       << "  -f, --file <path>    Read input from file\n"
       << "  --json               Output in JSON format\n"
       << "  -h, --help           Show this help\n";
}

std::string read_input(const std::string& text_arg, const std::string& file_path) {
    if (!file_path.empty()) {
        std::ifstream f(file_path);
        if (!f) {
            throw std::runtime_error("cannot open file: " + file_path);
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }
    if (!text_arg.empty()) {
        return text_arg;
    }
    // Read from stdin
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char* argv[]) try {
    if (argc < 2) {
        print_usage(std::cerr, argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "-h" || command == "--help" || command == "help") {
        print_usage(std::cout, argv[0]);
        return 0;
    }

    std::string model = "gpt-4o";
    std::string file_path;
    std::string text_arg;
    bool json_output = false;

    // Parse options
    int i = 2;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "error: --model requires an argument\n";
                return 1;
            }
            model = argv[++i];
        } else if (arg == "-f" || arg == "--file") {
            if (i + 1 >= argc) {
                std::cerr << "error: --file requires an argument\n";
                return 1;
            }
            file_path = argv[++i];
        } else if (arg == "--json") {
            json_output = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(std::cout, argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            // Remaining args are text
            for (int j = i; j < argc; ++j) {
                if (!text_arg.empty()) text_arg += " ";
                text_arg += argv[j];
            }
            break;
        } else {
            std::cerr << "error: unknown option: " << arg << "\n";
            return 1;
        }
        ++i;
    }

    std::string input = read_input(text_arg, file_path);
    auto tokenizer = zazaki_tokenizer::Tokenizer::for_model(model);

    if (command == "count") {
        size_t n = tokenizer.count_tokens(input);
        if (json_output) {
            std::cout << "{\"tokens\":" << n << "}\n";
        } else {
            std::cout << n << " tokens\n";
        }
    } else if (command == "encode") {
        auto result = tokenizer.encode(input);
        if (json_output) {
            std::cout << "{\"token_ids\":[";
            for (size_t j = 0; j < result.token_ids.size(); ++j) {
                if (j > 0) std::cout << ",";
                std::cout << result.token_ids[j];
            }
            std::cout << "],\"count\":" << result.token_count << "}\n";
        } else {
            std::cout << "[";
            for (size_t j = 0; j < result.token_ids.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << result.token_ids[j];
            }
            std::cout << "]\n";
        }
    } else if (command == "decode") {
        // Parse token IDs from remaining args
        std::vector<uint32_t> ids;
        if (!text_arg.empty()) {
            std::istringstream iss(text_arg);
            uint32_t id;
            while (iss >> id) {
                ids.push_back(id);
            }
        }
        if (ids.empty()) {
            std::cerr << "error: decode requires token IDs as arguments\n";
            return 1;
        }
        std::string decoded = tokenizer.decode(ids);
        if (json_output) {
            // Basic JSON escaping
            std::cout << "{\"text\":\"";
            for (char c : decoded) {
                if (c == '"') std::cout << "\\\"";
                else if (c == '\\') std::cout << "\\\\";
                else if (c == '\n') std::cout << "\\n";
                else if (c == '\r') std::cout << "\\r";
                else if (c == '\t') std::cout << "\\t";
                else std::cout << c;
            }
            std::cout << "\"}\n";
        } else {
            std::cout << decoded << "\n";
        }
    } else {
        std::cerr << "error: unknown command: " << command << "\n";
        print_usage(std::cerr, argv[0]);
        return 1;
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/cli/main.cpp && git commit -m "feat: CLI application with count/encode/decode commands"
```

---

### Task 8: Python Bindings

**Files:**
- Create: `src/bindings/python_bindings.cpp`

- [ ] **Step 1: Write Python bindings**

Write `src/bindings/python_bindings.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tokenizer.h"

namespace py = pybind11;

PYBIND11_MODULE(_zazaki_tokenizer, m) {
    m.doc() = "Zazaki Tokenizer - fast token counting for OpenAI models";

    m.def("count_tokens", [](const std::string& text) -> size_t {
        return zazaki_tokenizer::Tokenizer::for_model("gpt-4o").count_tokens(text);
    }, py::arg("text"), "Count tokens using default model (gpt-4o)");

    py::class_<zazaki_tokenizer::EncodingResult>(m, "EncodingResult")
        .def_readonly("token_ids", &zazaki_tokenizer::EncodingResult::token_ids)
        .def_readonly("token_count", &zazaki_tokenizer::EncodingResult::token_count);

    py::class_<zazaki_tokenizer::Tokenizer>(m, "Tokenizer")
        .def(py::init([](const std::string& model_name) {
            return zazaki_tokenizer::Tokenizer::for_model(model_name);
        }), py::arg("model_name") = "gpt-4o")
        .def_static("for_model", &zazaki_tokenizer::Tokenizer::for_model,
                    py::arg("model_name"))
        .def("encode", &zazaki_tokenizer::Tokenizer::encode,
             py::arg("text"), "Encode text to token IDs")
        .def("decode", &zazaki_tokenizer::Tokenizer::decode,
             py::arg("token_ids"), "Decode token IDs to text")
        .def("count", &zazaki_tokenizer::Tokenizer::count_tokens,
             py::arg("text"), "Count tokens in text")
        .def("count_tokens", &zazaki_tokenizer::Tokenizer::count_tokens,
             py::arg("text"), "Count tokens in text (alias)")
        .def("count_batch", [](const zazaki_tokenizer::Tokenizer& self,
                                const std::vector<std::string>& texts) {
            std::vector<size_t> results;
            results.reserve(texts.size());
            for (const auto& t : texts) {
                results.push_back(self.count_tokens(t));
            }
            return results;
        }, py::arg("texts"), "Count tokens for a batch of texts");
}
```

- [ ] **Step 2: Create Python package wrapper**

Write `src/bindings/zazaki_tokenizer.py`:

```python
"""Zazaki Tokenizer - fast token counting for OpenAI models."""

try:
    from ._zazaki_tokenizer import Tokenizer, count_tokens, EncodingResult
except ImportError:
    from _zazaki_tokenizer import Tokenizer, count_tokens, EncodingResult  # noqa: F811

__all__ = ["Tokenizer", "count_tokens", "EncodingResult"]
```

- [ ] **Step 3: Commit**

```bash
git add src/bindings/ && git commit -m "feat: Python bindings via pybind11"
```

---

### Task 9: Tests

**Files:**
- Create: `tests/test_encoder.cpp`
- Create: `tests/test_tokenizer.cpp`
- Create: `tests/test_registry.cpp`
- Create: `tests/test_python.py`

- [ ] **Step 1: Write encoder tests**

Write `tests/test_encoder.cpp`:

```cpp
#include <gtest/gtest.h>
#include "openai/encoder.h"
#include "openai/tiktoken_file.h"

// Extern from embedded data
extern const unsigned char cl100k_base_tiktoken[];
extern const unsigned int cl100k_base_tiktoken_len;

class EncoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto data = zazaki_tokenizer::openai::parse_tiktoken(
            cl100k_base_tiktoken, cl100k_base_tiktoken_len);
        encoder_ = std::make_unique<zazaki_tokenizer::openai::Encoder>(std::move(data));
    }
    std::unique_ptr<zazaki_tokenizer::openai::Encoder> encoder_;
};

TEST_F(EncoderTest, EncodeEmptyString) {
    auto ids = encoder_->encode("");
    EXPECT_TRUE(ids.empty());
}

TEST_F(EncoderTest, EncodeHelloWorld) {
    auto ids = encoder_->encode("hello world");
    EXPECT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 15339u);
    EXPECT_EQ(ids[1], 1917u);
}

TEST_F(EncoderTest, CountHelloWorld) {
    EXPECT_EQ(encoder_->count("hello world"), 2u);
}

TEST_F(EncoderTest, DecodeRoundtrip) {
    std::string text = "hello world";
    auto ids = encoder_->encode(text);
    std::string decoded = encoder_->decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST_F(EncoderTest, CountVsEncodeSize) {
    std::string text = "The quick brown fox jumps over the lazy dog.";
    auto ids = encoder_->encode(text);
    EXPECT_EQ(encoder_->count(text), ids.size());
}

TEST_F(EncoderTest, EncodeChinese) {
    auto ids = encoder_->encode("你好世界");
    EXPECT_GT(ids.size(), 0u);
    auto decoded = encoder_->decode(ids);
    EXPECT_EQ(decoded, "你好世界");
}

TEST_F(EncoderTest, DecodeOutOfRange) {
    EXPECT_THROW(encoder_->decode({999999999u}), std::out_of_range);
}
```

- [ ] **Step 2: Write tokenizer tests**

Write `tests/test_tokenizer.cpp`:

```cpp
#include <gtest/gtest.h>
#include "tokenizer.h"

TEST(TokenizerTest, ForModelGpt4o) {
    auto t = zazaki_tokenizer::Tokenizer::for_model("gpt-4o");
    auto n = t.count_tokens("hello world");
    EXPECT_EQ(n, 2u);
}

TEST(TokenizerTest, ForModelGpt4) {
    auto t = zazaki_tokenizer::Tokenizer::for_model("gpt-4");
    auto n = t.count_tokens("hello world");
    EXPECT_EQ(n, 2u);
}

TEST(TokenizerTest, UnknownModel) {
    EXPECT_THROW(
        zazaki_tokenizer::Tokenizer::for_model("nonexistent-model"),
        std::invalid_argument
    );
}

TEST(TokenizerTest, EncodeReturnsCorrectSizes) {
    auto t = zazaki_tokenizer::Tokenizer::for_model("gpt-4o");
    auto result = t.encode("hello world");
    EXPECT_EQ(result.token_ids.size(), result.token_count);
    EXPECT_EQ(result.token_count, 2u);
}
```

- [ ] **Step 3: Write registry tests**

Write `tests/test_registry.cpp`:

```cpp
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
```

- [ ] **Step 4: Write Python tests**

Write `tests/test_python.py`:

```python
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
        ids = t.encode("hello world")
        assert ids == [15339, 1917]

    def test_decode_roundtrip(self):
        t = Tokenizer("gpt-4o")
        text = "hello world"
        ids = t.encode(text)
        assert t.decode(ids) == text

    def test_count_batch(self):
        t = Tokenizer("gpt-4o")
        counts = t.count_batch(["hello", "world", "test"])
        assert counts == [1, 1, 1]

    def test_unknown_model(self):
        with pytest.raises(RuntimeError):
            Tokenizer("nonexistent-model")
```

- [ ] **Step 5: Commit**

```bash
git add tests/ && git commit -m "test: encoder, tokenizer, registry, and Python binding tests"
```

---

### Task 10: Build and Verify

- [ ] **Step 1: Configure build (without tests first)**

Run:
```bash
mkdir -p /home/zazaki/Projects/zazaki_tokenizer/build && cmake -S /home/zazaki/Projects/zazaki_tokenizer -B /home/zazaki/Projects/zazaki_tokenizer/build -DBUILD_TESTS=ON -DBUILD_PYTHON=ON
```

Expected: CMake configures successfully, finds pybind11, RE2, GoogleTest.

- [ ] **Step 2: Build**

Run:
```bash
cmake --build /home/zazaki/Projects/zazaki_tokenizer/build -j$(nproc)
```

Expected: All targets compile without errors.

- [ ] **Step 3: Run C++ tests**

Run:
```bash
cd /home/zazaki/Projects/zazaki_tokenizer/build && ctest --output-on-failure
```

Expected: All tests pass.

- [ ] **Step 4: Run Python tests**

Run:
```bash
PYTHONPATH=/home/zazaki/Projects/zazaki_tokenizer/build/src/bindings python3 -m pytest /home/zazaki/Projects/zazaki_tokenizer/tests/test_python.py -v
```

Expected: All Python tests pass.

- [ ] **Step 5: Manual CLI smoke test**

Run:
```bash
/home/zazaki/Projects/zazaki_tokenizer/build/src/cli/zazaki_tokenizer count "hello world"
```

Expected output: `2 tokens`

- [ ] **Step 6: Cross-validate with Python tiktoken**

Run:
```bash
python3 -c "
import tiktoken
import subprocess
import json

# Compare count for various texts
texts = ['hello world', '你好世界', 'OpenAI is amazing!', 'a' * 1000]
tiktoken_enc = tiktoken.encoding_for_model('gpt-4o')

for text in texts:
    expected = len(tiktoken_enc.encode(text))
    result = subprocess.run(['/home/zazaki/Projects/zazaki_tokenizer/build/src/cli/zazaki_tokenizer', 'count', '--json', text],
                           capture_output=True, text=True)
    actual = json.loads(result.stdout)['tokens']
    status = 'PASS' if expected == actual else 'FAIL'
    print(f'{status}: \"{text[:30]}...\" expected={expected} actual={actual}')
"
```

Expected: All checks PASS.

- [ ] **Step 7: Commit if any build fixes made**

```bash
git status && git diff
```

If changes exist, commit them.

---

### Plan Summary

| Task | Component | Est. Time |
|------|-----------|-----------|
| 1 | Project scaffolding | 10 min |
| 2 | Generate vocabulary files | 5 min |
| 3 | Tiktoken file parser | 10 min |
| 4 | BPE encoder | 15 min |
| 5 | Registry | 5 min |
| 6 | Tokenizer API | 5 min |
| 7 | CLI application | 10 min |
| 8 | Python bindings | 10 min |
| 9 | Tests | 10 min |
| 10 | Build and verify | 10 min |
