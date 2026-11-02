#include "File.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iterator>
#include <numeric>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>

File::File(const std::filesystem::path& path, LoadMethod method) : file_path_(path) {
    const auto start = std::chrono::steady_clock::now();
    if (method == LoadMethod::Mmap) {
        load_mmap(path);
    } else {
        load_stream(path);
    }
    load_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void File::reserve_for(std::size_t newline_count, std::uintmax_t bytes) {
    const auto upper_bound = newline_count + 1;
    embeddings_.reserve(upper_bound);
    chunk_ids_.reserve(upper_bound);
    text_offsets_.reserve(upper_bound + 1);
    text_buffer_.reserve(static_cast<std::size_t>(bytes));
    text_offsets_.push_back(0);
}

// One line parser for both backends, so any timing difference is purely I/O.
void File::append_line(std::string_view line, std::uint64_t& line_number) {
    if (!line.empty() && line.back() == '\r') {
        line.remove_suffix(1);
    }
    if (line.empty()) {
        ++line_number;
        return;
    }
    text_buffer_.append(line);
    text_offsets_.push_back(static_cast<std::uint64_t>(text_buffer_.size()));
    chunk_ids_.push_back(line_number++);
    embeddings_.emplace_back();
}

// read() copies page cache -> filebuf's buffer, getline copies that into `line`,
// then append copies it into text_buffer_. The file is also read twice (count + parse).
void File::load_stream(const std::filesystem::path& path) {
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

    std::error_code ec; // error code is a struct that contains the error code and the message
    const auto bytes = std::filesystem::file_size(path, ec);
    reserve_for(static_cast<std::size_t>(newline_count), ec ? 0 : bytes);

    std::uint64_t line_number = 1;
    std::string line;
    while (std::getline(input, line)) {
        append_line(line, line_number);
    }
}

// mmap maps the page cache straight into our address space: no read() copy and no
// intermediate std::string. The only copy left is into text_buffer_, which owns the
// text so the mapping can be dropped when the constructor returns.
void File::load_mmap(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY); // file descriptor: an integer that identifies an open file in the file system
    if (fd < 0) {
        throw std::runtime_error("Could not open file: " + path.string() + ": " + std::strerror(errno));
    }

    struct stat st{};
    if (::fstat(fd, &st) < 0) {
        const int err = errno;
        ::close(fd);
        throw std::runtime_error("fstat failed: " + path.string() + ": " + std::strerror(err));
    }
    const auto size = static_cast<std::size_t>(st.st_size); // st_size: the size of the file in bytes
    if (size == 0) { // mmap rejects length 0 with EINVAL
        ::close(fd); // closing a file descriptor means that the file is no longer accessible
        text_offsets_.push_back(0);
        return;
    }

    void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int map_err = errno;
    ::close(fd); // the mapping keeps the file alive; the descriptor is no longer needed
    if (mapped == MAP_FAILED) {
        throw std::runtime_error("mmap failed: " + path.string() + ": " + std::strerror(map_err));
    }

    // Unmaps on every exit path, including a bad_alloc thrown from append_line.
    struct Unmap {
        void* p;
        std::size_t n;
        ~Unmap() { ::munmap(p, n); }
    } guard{mapped, size};

    ::madvise(mapped, size, MADV_SEQUENTIAL); // hint: read ahead aggressively

    const char* const data = static_cast<const char*>(mapped);
    const char* const end = data + size;
    reserve_for(static_cast<std::size_t>(std::count(data, end, '\n')), size);

    std::uint64_t line_number = 1;
    const char* cursor = data;
    while (cursor < end) {
        const auto* nl = static_cast<const char*>(
            std::memchr(cursor, '\n', static_cast<std::size_t>(end - cursor)));
        const char* line_end = nl ? nl : end; // last line may lack a trailing '\n'
        append_line(std::string_view(cursor, static_cast<std::size_t>(line_end - cursor)), line_number);
        cursor = nl ? nl + 1 : end;
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
