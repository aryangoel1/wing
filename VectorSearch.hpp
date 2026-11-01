#ifndef DA424F61_5C04_4CC5_9639_987F564D318E
#define DA424F61_5C04_4CC5_9639_987F564D318E

#include "Chunk.hpp"
#include "File.hpp"
#include "Socket.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

class VectorSearch {
private:
    std::vector<File> Files;

public:
    struct SearchResult {
        std::uint64_t id;
        float score;
        std::string_view text;
        const std::filesystem::path* file_path;
    };

    void addFile(const std::filesystem::path& path);
    void embedAll(Socket& socket);
    File& file_at(std::size_t index) { return Files.at(index); }
    const File& file_at(std::size_t index) const { return Files.at(index); }
    std::size_t file_count() const noexcept { return Files.size(); }

    std::vector<SearchResult> rank(std::span<const float> query, std::size_t k) const;
    std::vector<SearchResult> search(Socket& socket, const std::string& query, std::size_t k) const;
};

#endif /* DA424F61_5C04_4CC5_9639_987F564D318E */
