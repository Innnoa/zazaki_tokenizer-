#include "registry.h"
#include "openai/tiktoken_file.h"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

extern const unsigned char cl100k_base_tiktoken[];
extern const unsigned int cl100k_base_tiktoken_len;

extern const unsigned char o200k_base_tiktoken[];
extern const unsigned int o200k_base_tiktoken_len;

namespace zazaki_tokenizer {

namespace {

std::string cache_path(const std::string& model_name) {
    const char* home = std::getenv("HOME");
    if (!home) return "";
    namespace fs = std::filesystem;
    fs::path dir = fs::path(home) / ".cache" / "zazaki_tokenizer";
    return (dir / (model_name + ".tiktoken")).string();
}

} // namespace

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
    static std::once_flag init_flag;
    static Registry* reg = nullptr;
    std::call_once(init_flag, [] {
        static Registry instance;
        reg = &instance;
    });
    return *reg;
}

const openai::Encoder* Registry::get_encoder(const std::string& model_name) const {
    auto it = models_.find(model_name);
    if (it == models_.end()) {
        // Try auto-loading from cache
        std::string path = cache_path(model_name);
        if (!path.empty()) {
            std::ifstream f(path);
            if (f.good()) {
                f.close();
                try {
                    const_cast<Registry*>(this)->register_from_file(model_name, path);
                    return models_.at(model_name).get();
                } catch (...) {
                    // Fall through to error
                }
            }
        }
        throw std::invalid_argument("unknown model: " + model_name);
    }
    return it->second.get();
}

void Registry::register_model(const std::string& model_name, EncoderPtr encoder) {
    models_[model_name] = std::move(encoder);
}

void Registry::register_from_file(const std::string& model_name,
                                   const std::string& file_path) {
    std::ifstream f(file_path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("cannot open vocabulary file: " + file_path);
    }
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<unsigned char> buffer(size);
    f.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size));

    auto data = openai::parse_tiktoken(buffer.data(), static_cast<unsigned int>(size));
    auto encoder = std::make_shared<openai::Encoder>(std::move(data));
    register_model(model_name, std::move(encoder));
}

bool Registry::has_model(const std::string& model_name) const {
    return models_.find(model_name) != models_.end();
}

} // namespace zazaki_tokenizer
