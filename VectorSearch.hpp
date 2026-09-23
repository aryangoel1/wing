#ifndef DA424F61_5C04_4CC5_9639_987F564D318E
#define DA424F61_5C04_4CC5_9639_987F564D318E

#include "File.hpp"
#include "Socket.hpp"
#include <cstddef>
#include <filesystem>
#include <string>
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

    void addFile(const std::filesystem::path& path); 
    void embedAll(Socket& socket);
    // Top-K results
    std::vector<SearchResult> search(Socket& socket, const std::string& query, std::size_t k) const;
};
#endif /* DA424F61_5C04_4CC5_9639_987F564D318E */