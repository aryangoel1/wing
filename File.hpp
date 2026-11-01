#ifndef EEB45E86_ABC9_4246_9506_71F1A4BA48D8
#define EEB45E86_ABC9_4246_9506_71F1A4BA48D8

#include "Chunk.hpp"
#include "Socket.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

class File {
private:
    std::vector<std::array<float, EMBEDDING_DIM>> embeddings_;
    std::string text_buffer_;
    std::vector<std::uint64_t> text_offsets_;
    std::vector<std::uint64_t> chunk_ids_;
    std::filesystem::path file_path_;

public:
    explicit File(const std::filesystem::path& path);

    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&&) noexcept = default;
    File& operator=(File&&) noexcept = default;

    void displayFileContents() const;
    void displayFormattedContents() const;
    void embed(Socket& socket);

    std::string_view get_text(std::size_t index) const;
    std::size_t chunk_count() const noexcept { return embeddings_.size(); }
    std::uint64_t chunk_id(std::size_t index) const { return chunk_ids_[index]; }
    const std::filesystem::path& path() const noexcept { return file_path_; }
    std::size_t text_offset_count() const noexcept { return text_offsets_.size(); }

    std::span<const std::array<float, EMBEDDING_DIM>> embeddings() const noexcept {
        return embeddings_;
    }
    std::span<std::array<float, EMBEDDING_DIM>> embeddings() noexcept { return embeddings_; }
};

static_assert(sizeof(std::uint64_t) == 8, "text offsets must be 64-bit");

#endif /* EEB45E86_ABC9_4246_9506_71F1A4BA48D8 */
