#include "VectorSearch.hpp"
#include "VectorMath.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <print>
#include <utility>
#include <cstddef>
#include <string>
#include <vector>

void VectorSearch::addFile(const std::filesystem::path& path) {
    File newFile(path);
    Files.push_back(std::move(newFile));
}

void VectorSearch::embedAll(Socket& socket) {
    for (auto& file : Files) {
        file.embed(socket);
    }
}

std::vector<VectorSearch::SearchResult> VectorSearch::rank(std::span<const float> query,
                                                           std::size_t k) const {
    if (k == 0) {
        return {};
    }

    struct Candidate {
        std::size_t file_index;
        std::size_t chunk_index;
        float score;
    };

    std::size_t total_chunks = 0;
    for (const auto& file : Files) {
        total_chunks += file.chunk_count();
    }

    std::vector<Candidate> candidates;
    candidates.reserve(total_chunks);
    for (std::size_t f = 0; f < Files.size(); ++f) {
        const auto embs = Files[f].embeddings();
        for (std::size_t i = 0; i < embs.size(); ++i) {
            candidates.push_back(
                {f, i, VectorMath::dot_product(query, embs[i])});
        }
    }

    auto by_score = [](const Candidate& left, const Candidate& right) {
        return left.score > right.score;
    };

    if (candidates.size() > k) {
        std::partial_sort(candidates.begin(),
                          candidates.begin() + static_cast<std::ptrdiff_t>(k),
                          candidates.end(), by_score);
        candidates.resize(k);
    } else {
        std::sort(candidates.begin(), candidates.end(), by_score);
    }

    std::vector<SearchResult> results;
    results.reserve(candidates.size());
    for (const auto& c : candidates) {
        const auto& file = Files[c.file_index];
        results.push_back({
            file.chunk_id(c.chunk_index),
            c.score,
            file.get_text(c.chunk_index),
            &file.path(),
        });
    }
    return results;
}

std::vector<VectorSearch::SearchResult> VectorSearch::search(Socket& socket,
                                                             const std::string& query,
                                                             std::size_t k) const {
    const auto t_start = std::chrono::steady_clock::now();

    std::array<float, EMBEDDING_DIM> query_embedding{};
    socket.send(query);
    socket.receive(query_embedding.data(), query_embedding.size());

    const auto t_embedded = std::chrono::steady_clock::now();

    auto results = rank(query_embedding, k);

    std::size_t total_chunks = 0;
    for (const auto& file : Files) {
        total_chunks += file.chunk_count();
    }

    const auto t_done = std::chrono::steady_clock::now();
    const auto ms = [](auto from, auto to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };
    std::println("  scanned {} chunks in {:.3f} ms  (query embedding: {:.2f} ms)",
                 total_chunks, ms(t_embedded, t_done), ms(t_start, t_embedded));

    return results;
}
