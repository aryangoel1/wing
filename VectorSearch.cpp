#include "VectorSearch.hpp"
#include "Chunk.hpp"
#include "Socket.hpp"
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

std::vector<VectorSearch::SearchResult> VectorSearch::search(Socket& socket,const std::string& query, std::size_t k) const {
    if (k == 0) {
        return {};
    }
    const auto t_start = std::chrono::steady_clock::now();

    // Embed the query using the socket
    std::array<float, EMBEDDING_DIM> query_embedding{};
    socket.send(query);
    socket.receive(query_embedding.data(), query_embedding.size());

    // Split the timing here: everything before this point is one network round trip
    // plus a model forward pass, everything after is pure local compute.
    const auto t_embedded = std::chrono::steady_clock::now();

    // Count the chunks up front so results allocates exactly once. Without this the
    // vector doubles its way to the final size, copying everything it already holds
    // on each reallocation.
    std::size_t total_chunks = 0;
    for (const auto& file : Files) {
        total_chunks += file.getChunks().size();
    }

    std::vector<SearchResult> results;
    results.reserve(total_chunks);
    for (const auto& file : Files) {
        for (const auto& chunk : file.getChunks()) {
            results.push_back({ &chunk, VectorMath::dot_product(query_embedding, chunk.embedding) });
        }
    }
    auto by_score = [](const SearchResult& left, const SearchResult& right) {
        return left.score > right.score;
    };

    if (results.size() > k) {
        // Sort the first k results of the results vector using the specified lambda function
        // ptrdiff_t stores the result of subtracting 2 pointers
        std::partial_sort(results.begin(), results.begin() + static_cast<std::ptrdiff_t>(k), results.end(), by_score);
        results.resize(k);
    } else {
        std::sort(results.begin(), results.end(), by_score);
    }
    const auto t_done = std::chrono::steady_clock::now();
    const auto ms = [](auto from, auto to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };
    std::println("  scanned {} chunks in {:.3f} ms  (query embedding: {:.2f} ms)",
                 total_chunks, ms(t_embedded, t_done), ms(t_start, t_embedded));

    return results;
}