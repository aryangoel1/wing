#ifndef EEB45E86_ABC9_4246_9506_71F1A4BA48D8
#define EEB45E86_ABC9_4246_9506_71F1A4BA48D8

// This file will represent a simple file which is a collection of Chunks
#include "Chunk.hpp"
#include "Socket.hpp"
#include <filesystem>
#include <span>
#include <vector>

class File {
private:
    std::vector<Chunk> chunks;

public:
    // Now, we need to actually populate each chunk with the correct metadata inside of the body of the constructor
    explicit File(const std::filesystem::path& path);

    // No copy constructor since copying chunks can linearly scale with the size of the file
    // as that would mean allocating a new std::vector and then explicitly copying each chunk into it
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&&) noexcept = default;

    // noexcept here matters since std::vector prefers
    // moving elements during reallocation only when their custom
    // type promises to not throw, otherwise it may copy
    File& operator=(File&&) noexcept = default;

    void displayFileContents() const;
    void displayFormattedContents() const;
    void embed(Socket& socket);

    // Return a span of chunks belonging to the file, however one cannot modify the chunks
    // pointer + length where length is the number of elements (not number of bytes)
    std::span<const Chunk> getChunks() const noexcept { return chunks; } 
};

#endif /* EEB45E86_ABC9_4246_9506_71F1A4BA48D8 */
