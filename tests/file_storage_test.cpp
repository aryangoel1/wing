#include "Chunk.hpp"
#include "File.hpp"
#include "VectorMath.hpp"
#include "VectorSearch.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int g_failed = 0;
int g_passed = 0;

void expect(bool cond, const char* expr, const char* file, int line) {
    if (cond) {
        ++g_passed;
        return;
    }
    ++g_failed;
    std::cerr << "FAIL " << file << ":" << line << "  " << expr << '\n';
}

#define CHECK(expr) expect(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

std::filesystem::path write_temp(const std::string& name, std::string_view contents) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("could not write temp file");
    }
    out << contents;
    return path;
}

void test_short_lines() {
    std::string body;
    for (int i = 0; i < 200; ++i) {
        body += "L" + std::to_string(i) + "\n";
    }
    const auto path = write_temp("wing_short.txt", body);
    File file(path);
    CHECK(file.chunk_count() == 200);
    CHECK(file.text_offset_count() == 201);
    CHECK(file.get_text(0) == "L0");
    CHECK(file.get_text(199) == "L199");
    CHECK(file.chunk_id(0) == 1);
    CHECK(file.chunk_id(199) == 200);
    CHECK(file.path() == path);
    std::filesystem::remove(path);
}

void test_long_line() {
    const std::string long_line(8000, 'x');
    const auto path = write_temp("wing_long.txt", long_line + "\nshort\n");
    File file(path);
    CHECK(file.chunk_count() == 2);
    CHECK(file.get_text(0).size() == 8000);
    CHECK(file.get_text(0) == long_line);
    CHECK(file.get_text(1) == "short");
    std::filesystem::remove(path);
}

void test_empty_file() {
    const auto path = write_temp("wing_empty.txt", "");
    File file(path);
    CHECK(file.chunk_count() == 0);
    CHECK(file.text_offset_count() == 1);
    CHECK(file.embeddings().empty());
    std::filesystem::remove(path);
}

void test_empty_lines_preserve_ids() {
    const auto path = write_temp("wing_blanks.txt", "alpha\n\nbeta\n\n\ngamma\n");
    File file(path);
    CHECK(file.chunk_count() == 3);
    CHECK(file.get_text(0) == "alpha");
    CHECK(file.get_text(1) == "beta");
    CHECK(file.get_text(2) == "gamma");
    CHECK(file.chunk_id(0) == 1);
    CHECK(file.chunk_id(1) == 3);
    CHECK(file.chunk_id(2) == 6);
    std::filesystem::remove(path);
}

void test_crlf_matches_lf() {
    const auto lf = write_temp("wing_lf.txt", "one\ntwo\n\nthree\n");
    const auto crlf = write_temp("wing_crlf.txt", "one\r\ntwo\r\n\r\nthree\r\n");
    File a(lf);
    File b(crlf);
    CHECK(a.chunk_count() == b.chunk_count());
    CHECK(a.chunk_count() == 3);
    for (std::size_t i = 0; i < a.chunk_count(); ++i) {
        CHECK(a.get_text(i) == b.get_text(i));
        CHECK(a.chunk_id(i) == b.chunk_id(i));
        CHECK(a.get_text(i).find('\r') == std::string_view::npos);
    }
    std::filesystem::remove(lf);
    std::filesystem::remove(crlf);
}

void test_offset_and_mapping() {
    const auto path = write_temp("wing_map.txt", "ab\ncd\nef\n");
    File file(path);
    CHECK(file.text_offset_count() == file.chunk_count() + 1);
    CHECK(file.embeddings().size() == file.chunk_count());
    CHECK(file.get_text(0) == "ab");
    CHECK(file.get_text(1) == "cd");
    CHECK(file.get_text(2) == "ef");
    CHECK(file.chunk_id(0) == 1);
    CHECK(file.chunk_id(1) == 2);
    CHECK(file.chunk_id(2) == 3);
    CHECK(file.embeddings()[0].size() == EMBEDDING_DIM);
    std::filesystem::remove(path);
}

void test_no_persistence_sidecar() {
    const auto path = write_temp("wing_nopersist.txt", "only line\n");
    File file(path);
    CHECK(file.chunk_count() == 1);
    CHECK(!std::filesystem::exists(path.string() + ".idx"));
    CHECK(!std::filesystem::exists(path.string() + ".wing"));
    CHECK(!std::filesystem::exists(path.string() + ".index"));
    std::filesystem::remove(path);
}

void test_search_ranking() {
    const auto path = write_temp("wing_search.txt", "red apple\ngreen pear\nblue berry\n");
    VectorSearch db;
    db.addFile(path);
    CHECK(db.file_count() == 1);
    auto& file = db.file_at(0);

    auto unit = [](std::size_t hot) {
        std::array<float, EMBEDDING_DIM> e{};
        e[hot] = 1.0f;
        return e;
    };

    file.embeddings()[0] = unit(0);
    file.embeddings()[1] = unit(1);
    file.embeddings()[2] = unit(2);

    const auto hits = db.rank(unit(1), 2);
    CHECK(hits.size() == 2);
    CHECK(hits[0].text == "green pear");
    CHECK(hits[0].id == 2);
    CHECK(hits[0].file_path != nullptr);
    CHECK(*hits[0].file_path == path);
    CHECK(hits[0].score > hits[1].score);
    CHECK(hits[0].score > 0.9f);

    const float expected = VectorMath::dot_product(unit(1), file.embeddings()[1]);
    CHECK(hits[0].score == expected);

    std::filesystem::remove(path);
}

void test_uint64_offsets() {
    CHECK(sizeof(std::uint64_t) == 8);
    const auto path = write_temp("wing_offsets.txt", "a\nb\n");
    File file(path);
    CHECK(file.text_offset_count() == 3);
    std::filesystem::remove(path);
}

} // namespace

int main() {
    test_short_lines();
    test_long_line();
    test_empty_file();
    test_empty_lines_preserve_ids();
    test_crlf_matches_lf();
    test_offset_and_mapping();
    test_no_persistence_sidecar();
    test_search_ranking();
    test_uint64_offsets();

    std::cout << "passed=" << g_passed << " failed=" << g_failed << '\n';
    return g_failed == 0 ? 0 : 1;
}
