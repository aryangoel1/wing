#ifndef DA424F61_5C04_4CC5_9639_987F564D318E
#define DA424F61_5C04_4CC5_9639_987F564D318E

#include "File.hpp"
#include "Socket.hpp"
#include "VectorMath.hpp"
#include <algorithm>
#include <filesystem>
#include <array>
#include <string>
#include <utility>
#include <vector>
#include <cstddef>

// This will be the Vector Database class
class VectorSearch {
private:
    std::vector<File> Files;
public:
    struct SearchResult {
        const Chunk* chunk;
        float score;
    };

    void addFile(const std::filesystem::path& path) {
        File newFile(path);
        Files.push_back(std::move(newFile));
    }
    void embedAll(Socket& socket) {
        for (auto& file : Files) {
            file.embed(socket); 
        }
    }
    // Top-K results
    std::vector<SearchResult> search(Socket& socket, const std::string& query, std::size_t k) const {
        
        if (k == 0) {
            return {};
        }
        // Embed the query using the socket
        std::array<float, EMBEDDING_DIM> query_embedding{};
        socket.send(query);
        socket.receive(query_embedding.data(), query_embedding.size());

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
            std::partial_sort(results.begin(), results.begin() + static_cast<std::ptrdiff_t>(k), results.end(), by_score);
            results.resize(k);
        } else {
            std::sort(results.begin(), results.end(), by_score);
        }
        return results;
    }
};
#endif /* DA424F61_5C04_4CC5_9639_987F564D318E */