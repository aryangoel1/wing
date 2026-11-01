#include "File.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <numeric>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>

File::File(const std::filesystem::path& path) : file_path_(path) {
    std::uint64_t line_number = 1;
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    const auto newline_count = std::count(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
        '\n'
    );

    input.clear();
    input.seekg(0);

    const auto upper_bound = static_cast<std::size_t>(newline_count) + 1;
    embeddings_.reserve(upper_bound);
    chunk_ids_.reserve(upper_bound);
    text_offsets_.reserve(upper_bound + 1);

    std::error_code ec;
    const auto bytes = std::filesystem::file_size(path, ec);
    if (!ec) {
        text_buffer_.reserve(static_cast<std::size_t>(bytes));
    }

    text_offsets_.push_back(0);

    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            ++line_number;
            continue;
        }

        text_buffer_.append(line);
        text_offsets_.push_back(static_cast<std::uint64_t>(text_buffer_.size()));
        chunk_ids_.push_back(line_number++);
        embeddings_.emplace_back();
    }
}

std::string_view File::get_text(std::size_t index) const {
    const auto start = text_offsets_[index];
    const auto end = text_offsets_[index + 1];
    if (end == start) {
        return {};
    }
    return std::string_view(
        text_buffer_.data() + start,
        static_cast<std::size_t>(end - start)
    );
}

void File::displayFileContents() const {
    for (std::size_t i = 0; i < chunk_count(); ++i) {
        std::print("{}", get_text(i));
    }
    std::println();
}

void File::displayFormattedContents() const {
    for (std::size_t i = 0; i < chunk_count(); ++i) {
        std::println(
            "ID: {} | Chunk: {} | File: {}",
            chunk_ids_[i],
            get_text(i),
            file_path_.string()
        );
    }
}

void File::embed(Socket& socket) {
    std::vector<std::size_t> order(chunk_count());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) {
                  return get_text(a).size() > get_text(b).size();
              });

    constexpr std::size_t batch_size = 42;

    for (std::size_t start = 0; start < order.size(); start += batch_size) {
        const std::size_t end = std::min(start + batch_size, order.size());
        for (std::size_t i = start; i < end; ++i) {
            socket.send(get_text(order[i]));
        }
        for (std::size_t i = start; i < end; ++i) {
            auto& embedding = embeddings_[order[i]];
            socket.receive(embedding.data(), embedding.size());
        }
    }
}
