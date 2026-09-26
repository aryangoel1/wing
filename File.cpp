#include "File.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <cstddef>
#include <sys/socket.h>
#include <utility> // std::move inside the utility header will cast the specific object to an rvalue reference
#include <print>
#include <numeric>
#include <vector>

File::File(const std::filesystem::path& path) {
    // If this variable was outside the body of the constructor, then it would be shared across all
    // File objects, essentially making it static
    std::uint64_t line_number = 1; // This makes it so that it is not shared across every File object
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    // Count the lines up front so that chunks allocates exactly once.
    // Without this, std::vector grows by doubling, and every reallocation has to relocate
    // every Chunk built so far. Now that Chunk holds its embedding inline it is ~1.6 KB,
    // so that relocation moves real bytes rather than a handful of pointers.
    // istreambuf_iterator reads raw characters straight from the stream buffer,
    // skipping the formatting layer that operator>> would go through.
    const auto newline_count = std::count(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
        '\n'
    );

    // The counting pass consumed the stream and set the EOF flag, so clear the flag
    // and seek back to byte 0 before the real pass. clear() must come first:
    // seekg() on a stream in a failed state is a no-op.
    input.clear();
    input.seekg(0);

    // An upper bound: this counts blank lines too, and those get skipped below.
    // Reserving slightly over is free, whereas reserving under costs a reallocation.
    // The +1 covers a final line with no trailing newline.
    chunks.reserve(static_cast<std::size_t>(newline_count) + 1);

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            ++line_number;
            continue; 
        }
        Chunk chunk{};
        chunk.text = line;
        chunk.id = line_number++;
        chunk.file_path = path;

        // std::move casts chunk to an rvalue reference so push_back moves instead of copies.
        // Note this is now weaker than it used to be: moving a Chunk still steals the two
        // strings' buffers, but the std::array embedding has no pointer to steal, so its
        // 1536 bytes get copied either way. Constructing in place with emplace_back would
        // avoid that copy entirely.
        chunks.push_back(std::move(chunk)); // std::move to not make a copy
    }
}

void File::displayFileContents() const {
    for (const auto& line : chunks) {
        std::print("{}", line.text);
    }
    std::println();
}

void File::displayFormattedContents() const {
    for (const auto& chunk : chunks) {
        std::println(
            "ID: {} | Chunk: {} | File: {}",
            chunk.id,
            chunk.text,
            chunk.file_path
        );
    }
}

void File::embed(Socket& socket) {
    // This function will send a batch size of 42 - 42 lines total
    // However, to reduce padding within all-minilm, we need to send the data with padding minimized
    // This means, we sort the file by whichever line is longest in length
    // First, we sort the file by which chunks have the biggest length
    // Swapping an array would mean copying O(n) operation 

    // Use the index to access the chunk directly from there rather than rearranging the chunks
    // Rearranging the Chunks can cause 1536 bytes to get copied per swap versus rearranging the
    // indices. std::sort expects the same type as the template instantiation in the parameter list
    // [&] allows the lambda to recieve access to local variables by reference
    std::vector<std::size_t> order(chunks.size()); // auto-initializes all elements to 0 
    std::iota(order.begin(), order.end(), 0);  // Fills a range with sequentially incrementing values
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return chunks[a].text.length() > chunks[b].text.length(); });

    constexpr std::size_t batch_size = 42; // Directly baked into the binary - treated as a raw number
 
    for (std::size_t start = 0; start < chunks.size(); start += batch_size) {
        const std::size_t end = std::min(start + batch_size, chunks.size());
        for (std::size_t i = start; i < end; ++i) {
            socket.send(chunks[order[i]].text); // Send by order index
        }
        for (std::size_t i = start; i < end; ++i) {
            socket.receive(chunks[order[i]].embedding.data(), chunks[order[i]].embedding.size()); 
        }
    }
}    