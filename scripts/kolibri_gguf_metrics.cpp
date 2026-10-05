// Frozen-token likelihood and full-vocabulary measurements through the public llama.cpp API.
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;

static double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

static void write_json(const std::string & path, const json & value) {
    std::ofstream stream(path + ".tmp");
    stream << value.dump(2) << "\n";
    stream.close();
    if (!stream) throw std::runtime_error("Cannot write measurement JSON");
    std::filesystem::rename(path + ".tmp", path);
}

static double log_normalizer(const float * logits, int vocab) {
    double maximum = -INFINITY;
    for (int i = 0; i < vocab; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("Nonfinite model logits");
        maximum = std::max(maximum, double(logits[i]));
    }
    double sum = 0;
    for (int i = 0; i < vocab; ++i) sum += std::exp(double(logits[i]) - maximum);
    return maximum + std::log(sum);
}

static int greedy(const float * logits, int vocab) {
    return int(std::max_element(logits, logits + vocab) - logits);
}

static void fill_batch(llama_batch & batch, const std::vector<llama_token> & tokens,
                       int start, int stop, bool all_logits) {
    batch.n_tokens = stop - start;
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[start + i];
        batch.pos[i] = start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = all_logits || i + 1 == batch.n_tokens;
    }
}

static void decode(llama_context * context, llama_batch batch) {
    if (llama_decode(context, batch) != 0) throw std::runtime_error("llama_decode failed");
}

static std::vector<llama_token> generate(llama_context * context, const llama_vocab * vocab,
                                         llama_batch & batch, const std::vector<llama_token> & prompt,
                                         int count, bool ignore_eos) {
    llama_memory_clear(llama_get_memory(context), true);
    for (int start = 0; start < int(prompt.size()); start += 256) {
        fill_batch(batch, prompt, start, std::min(start + 256, int(prompt.size())), false);
        decode(context, batch);
    }
    std::vector<llama_token> generated;
    const int vocabulary = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < count; ++i) {
        const auto token = greedy(llama_get_logits_ith(context, -1), vocabulary);
        if (!ignore_eos && llama_vocab_is_eog(vocab, token)) break;
        generated.push_back(token);
        if (i + 1 < count) {
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = int(prompt.size()) + i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            decode(context, batch);
        }
    }
    return generated;
}

static std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string text;
    for (auto token : tokens) {
        std::vector<char> buffer(256);
        int length = llama_token_to_piece(vocab, token, buffer.data(), int(buffer.size()), 0, true);
        if (length < 0) {
            buffer.resize(-length);
            length = llama_token_to_piece(vocab, token, buffer.data(), int(buffer.size()), 0, true);
        }
        if (length < 0) throw std::runtime_error("Token piece does not fit its requested buffer");
        text.append(buffer.data(), length);
    }
    return text;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: kolibri-metrics MODEL PROTOCOL OUTPUT_JSON CONTEXT");
        const std::string output = argv[3];
        json protocol;
        std::ifstream(argv[2]) >> protocol;
        ggml_backend_load_all();
        llama_backend_init();
        if (!llama_supports_gpu_offload()) throw std::runtime_error("GPU backend was not loaded");
        auto parameters = llama_model_default_params();
        parameters.n_gpu_layers = 99;
        parameters.load_mode = LLAMA_LOAD_MODE_NONE;
        parameters.lazy_mode = LLAMA_LAZY_MODE_OFF;
        const auto started = clock_type::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], parameters), llama_model_free);
        if (!model) throw std::runtime_error("Cannot load the complete GGUF");
        const auto * vocab = llama_model_get_vocab(model.get());
        const int vocabulary = llama_vocab_n_tokens(vocab);
        auto context_parameters = llama_context_default_params();
        context_parameters.n_ctx = std::stoul(argv[4]);
        context_parameters.n_batch = 256;
        context_parameters.n_ubatch = 256;
        context_parameters.n_seq_max = 1;
        context_parameters.n_threads = 4;
        context_parameters.n_threads_batch = 4;
        context_parameters.type_k = GGML_TYPE_BF16;
        context_parameters.type_v = GGML_TYPE_BF16;
        context_parameters.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_parameters), llama_free);
        if (!context) throw std::runtime_error("Cannot create the GGUF inference context");
        llama_batch batch = llama_batch_init(256, 0, 1);
        json report = {{"label", "gguf-mxfp4"}, {"vocabulary", vocabulary},
                       {"load_seconds", seconds(started)}, {"windows", json::array()},
                       {"behavior", json::array()}, {"throughput", json::array()}};
        std::ofstream matrix(output + ".logprobs.f32", std::ios::binary);
        if (!matrix) throw std::runtime_error("Cannot open the full-vocabulary matrix");
        for (const auto & window : protocol["windows"]) {
            const auto tokens = window["token_ids"].get<std::vector<llama_token>>();
            for (auto token : tokens) if (token < 0 || token >= vocabulary) throw std::runtime_error("Invalid frozen token ID");
            llama_memory_clear(llama_get_memory(context.get()), true);
            double nll = 0;
            int count = 0;
            bool prefix_written = false;
            for (int start = 0; start < int(tokens.size()); start += 256) {
                const int stop = std::min(start + 256, int(tokens.size()));
                fill_batch(batch, tokens, start, stop, true);
                decode(context.get(), batch);
                for (int i = 0; i < batch.n_tokens; ++i) {
                    const auto * logits = llama_get_logits_ith(context.get(), i);
                    const double normalizer = log_normalizer(logits, vocabulary);
                    const int position = start + i;
                    if (position + 1 < int(tokens.size())) {
                        nll += normalizer - logits[tokens[position + 1]];
                        ++count;
                    }
                    if (position + 1 == window["kl_prefix_tokens"].get<int>()) {
                        std::vector<float> row(vocabulary);
                        for (int j = 0; j < vocabulary; ++j) row[j] = float(double(logits[j]) - normalizer);
                        matrix.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
                        prefix_written = true;
                    }
                }
            }
            if (!prefix_written || count != int(tokens.size()) - 1) throw std::runtime_error("Incomplete frozen measurement");
            report["windows"].push_back({{"id", window["id"]}, {"domain", window["domain"]},
                {"token_ids_sha256", window["token_ids_sha256"]}, {"scored_tokens", count},
                {"nll", nll}, {"mean_nll", nll / count}});
            write_json(output + ".partial.json", report);
            std::cout << "Completed passage " << report["windows"].size() << "/" << protocol["windows"].size() << std::endl;
        }
        matrix.close();
        if (!matrix) throw std::runtime_error("Cannot flush the full-vocabulary matrix");
        for (const auto & item : protocol["behavior_cases"]) {
            const auto prompt = item["token_ids"].get<std::vector<llama_token>>();
            const auto tokens = generate(context.get(), vocab, batch, prompt, item.value("max_tokens", 128), false);
            report["behavior"].push_back({{"id", item["id"]}, {"prompt_tokens", prompt.size()},
                {"token_ids", tokens}, {"text", detokenize(vocab, tokens)}});
            write_json(output + ".partial.json", report);
            std::cout << "Completed behavior " << item["id"] << std::endl;
        }
        auto prompt = protocol["windows"][0]["token_ids"].get<std::vector<llama_token>>();
        prompt.resize(std::min(size_t(512), prompt.size()));
        generate(context.get(), vocab, batch, prompt, 16, true);
        for (int repetition = 0; repetition < 3; ++repetition) {
            const auto before = clock_type::now();
            const auto tokens = generate(context.get(), vocab, batch, prompt, 128, true);
            const double elapsed = seconds(before);
            if (tokens.size() != 128) throw std::runtime_error("Incomplete forced throughput output");
            report["throughput"].push_back({{"concurrency", 1}, {"repetition", repetition},
                {"prompt_tokens_per_request", prompt.size()}, {"output_tokens_per_request", 128},
                {"seconds", elapsed}, {"aggregate_output_tokens_per_second", 128 / elapsed}});
        }
        report["seconds"] = seconds(started);
        write_json(output, report);
        llama_batch_free(batch);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
