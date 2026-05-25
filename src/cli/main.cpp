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
