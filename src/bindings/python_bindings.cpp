#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "registry.h"
#include "tokenizer.h"

namespace py = pybind11;

PYBIND11_MODULE(_zazaki_tokenizer, m) {
    m.doc() = "Zazaki Tokenizer - fast token counting for OpenAI models";

    m.def("count_tokens", [](const std::string& text) -> size_t {
            return zazaki_tokenizer::Tokenizer::for_model("gpt-4o").count_tokens(text);
        },
        py::arg("text"), "Count tokens using default model (gpt-4o)");

    m.def("register_model", [](const std::string& model_name,
                                const std::string& file_path) {
            zazaki_tokenizer::Registry::instance().register_from_file(
                model_name, file_path);
        },
        py::arg("model_name"), py::arg("file_path"),
        "Register a model from a .tiktoken vocabulary file");

    m.def("has_model", [](const std::string& model_name) -> bool {
            return zazaki_tokenizer::Registry::instance().has_model(model_name);
        },
        py::arg("model_name"), "Check if a model is registered");

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
