# Zazaki Tokenizer — 设计文档

## 概述

一个用于计算 AI 对话/文本 token 消耗的命令行工具及 Python 库。支持 OpenAI 系列模型（GPT-4o、GPT-4、GPT-3.5 等）的 token 计数与编解码。

## 动机

- 当前只有 Python/Rust 生态有现成的 token 计数工具
- 需要一个 C++ 实现，作为独立 CLI 和 Python 可导入库使用
- 架构预留扩展接口，后续可加入 Llama、Qwen、DeepSeek 等模型

## 技术选型

| 项目 | 选择 |
|------|------|
| 语言 | C++17 |
| 构建系统 | CMake |
| Python 绑定 | pybind11 |
| 词表数据 | 编译期嵌入 `.tiktoken` 文件 |

## 架构

分层架构，核心库 + CLI + Python 绑定分离：

```
src/
├── core/                     (libzazaki_tokenizer 静态库)
│   ├── tokenizer.h / .cpp    (公共接口)
│   ├── openai/               (OpenAI 编码器)
│   │   ├── tiktoken_file.h / .cpp  (.tiktoken 文件解析)
│   │   └── encoder.h / .cpp        (BPE 编码核心)
│   └── registry.h / .cpp     (模型名 → 编码器注册表)
├── cli/                      (CLI 可执行文件)
│   └── main.cpp
├── bindings/                 (pybind11 Python 模块)
│   └── python_bindings.cpp
└── data/                     (编译期嵌入的 .tiktoken 词表文件)
```

- `libzazaki_tokenizer`：静态库，包含所有核心逻辑
- CLI 和 Python 绑定分别链接该库，共享同一套实现
- `registry` 维护模型名到编码配置的映射，新增模型只需在注册表中加一行

## 核心 API

```cpp
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
};

}
```

- `for_model()` 按模型名查找注册表，自动加载对应 `.tiktoken` 词表
- `count_tokens()` 是 encode 的快路径，不分配 token_ids 向量
- 模型名映射：`gpt-4o`/`gpt-4o-mini` → `o200k_base`，`gpt-4`/`gpt-3.5-turbo` → `cl100k_base`

## CLI 接口

```
zazaki_tokenizer count [OPTIONS] [TEXT]
zazaki_tokenizer encode [OPTIONS] [TEXT]
zazaki_tokenizer encode -d [TOKEN_IDS...]
```

| 选项 | 说明 |
|------|------|
| `-m, --model <name>` | 指定模型（默认 `gpt-4o`） |
| `-f, --file <path>` | 从文件读取输入 |
| `--json` | 输出 JSON 格式 |
| `-h, --help` | 帮助 |

支持管道输入（stdin）。

## Python 绑定

```python
from zazaki_tokenizer import Tokenizer, count_tokens

t = Tokenizer("gpt-4o")
n = t.count("hello world")              # int
ids = t.encode("hello world")           # list[int]
text = t.decode([15339, 1917])          # str
counts = t.count_batch(["a", "b"])      # list[int]

n = count_tokens("hello world")         # 便捷函数，默认 gpt-4o
```

## 支持模型

### 初期（v1）

| 模型 | 编码 |
|------|------|
| `gpt-4o`, `gpt-4o-mini`, `o1`, `o3` | `o200k_base` |
| `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo` | `cl100k_base` |
| `text-embedding-ada-002` | `cl100k_base` |

### 扩展接口

在 `core/` 下按厂商添加子目录（如 `core/openai/`、`core/llama/`、`core/deepseek/`），各自实现统一的 `Encoder` 虚接口，在 `registry` 中注册即可。

## 词表数据

- `.tiktoken` 文件通过 `xxd -i` 或 CMake `target_embed` 转换为 C 字节数组，编译期嵌入二进制
- 运行时在注册表初始化时解析

## 错误处理

- 不存在的模型名：`for_model()` 抛出 `std::invalid_argument`
- 无效 token ID：`decode()` 抛出 `std::out_of_range`
- 文件读取失败：CLI 输出 `stderr` 错误信息，返回非零退出码
- Python 层：上述 C++ 异常自动转为 Python 异常（pybind11 内置）

## 测试策略

- 核心库单元测试（GoogleTest）：BPE 编码/解码正确性、token 计数
- 基准对比：与 Python `tiktoken` 库结果交叉验证
- CLI 集成测试：管道、文件、JSON 输出格式
- Python 绑定测试：pytest 覆盖基础 API

## 非目标（v1 不包含）

- 非 OpenAI 模型支持（架构预留但不实现）
- 训练/合并 BPE 词表
- HTTP API 服务
- 图片/音频 token 计算
