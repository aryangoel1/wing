#include "Chunk.hpp"
#include "File.hpp"
#include "VectorMath.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::atomic<std::uint64_t> g_alloc_calls{0};
std::atomic<std::uint64_t> g_alloc_bytes{0};

void* counted_malloc(std::size_t n) {
    g_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
    if (n == 0) {
        n = 1;
    }
    void* p = std::malloc(n);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

} // namespace

void* operator new(std::size_t n) { return counted_malloc(n); }
void* operator new[](std::size_t n) { return counted_malloc(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try {
        return counted_malloc(n);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try {
        return counted_malloc(n);
    } catch (...) {
        return nullptr;
    }
}

void* operator new(std::size_t n, std::align_val_t a) {
    g_alloc_calls.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
    void* p = nullptr;
    if (posix_memalign(&p, static_cast<std::size_t>(a), n == 0 ? 1 : n) != 0) {
        throw std::bad_alloc();
    }
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {

constexpr const char* kLayout = "SoA";
constexpr std::size_t kCorpusChunks = 100000;
constexpr std::size_t kTopK = 5;
constexpr int kSearchQueries = 80;
constexpr int kSearchWarmup = 8;

struct AllocSnapshot {
    std::uint64_t calls;
    std::uint64_t bytes;
};

AllocSnapshot snapshot_allocs() {
    return {g_alloc_calls.load(std::memory_order_relaxed),
            g_alloc_bytes.load(std::memory_order_relaxed)};
}

malloc_statistics_t zone_stats() {
    malloc_statistics_t stats{};
    malloc_zone_statistics(malloc_default_zone(), &stats);
    return stats;
}

struct Rss {
    std::uint64_t resident_bytes;
    std::uint64_t phys_footprint_bytes;
};

Rss process_rss() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
        KERN_SUCCESS) {
        return {0, 0};
    }
    return {static_cast<std::uint64_t>(info.resident_size),
            static_cast<std::uint64_t>(info.phys_footprint)};
}

std::filesystem::path write_corpus(std::size_t lines) {
    const auto path = std::filesystem::temp_directory_path() / "wing_bench_corpus.txt";
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write benchmark corpus");
    }
    for (std::size_t i = 0; i < lines; ++i) {
        out << "chunk-" << i << " the quick brown fox jumps over lazy dogs " << (i % 97) << '\n';
    }
    return path;
}

void fill_embeddings(File& file) {
    auto embs = file.embeddings();
    for (std::size_t i = 0; i < embs.size(); ++i) {
        auto& e = embs[i];
        for (std::size_t d = 0; d < EMBEDDING_DIM; ++d) {
            e[d] = static_cast<float>((i + 17 * d) % 23) * 0.037f;
        }
        const float mag = VectorMath::vectorMagnitude(e);
        if (mag > 0.0f) {
            const float inv = 1.0f / mag;
            for (float& v : e) {
                v *= inv;
            }
        }
    }
}

std::array<float, EMBEDDING_DIM> make_query(int q) {
    std::array<float, EMBEDDING_DIM> query{};
    for (std::size_t d = 0; d < EMBEDDING_DIM; ++d) {
        query[d] = static_cast<float>((static_cast<unsigned>(q) * 13u + 11u * d) % 23) * 0.037f;
    }
    const float mag = VectorMath::vectorMagnitude(query);
    if (mag > 0.0f) {
        const float inv = 1.0f / mag;
        for (float& v : query) {
            v *= inv;
        }
    }
    return query;
}

struct Candidate {
    std::size_t index;
    float score;
};

void rank_topk(const File& file, std::span<const float> query, std::size_t k,
               std::vector<Candidate>& results) {
    const auto embs = file.embeddings();
    results.clear();
    results.reserve(embs.size());
    for (std::size_t i = 0; i < embs.size(); ++i) {
        results.push_back({i, VectorMath::dot_product(query, embs[i])});
    }
    auto by_score = [](const Candidate& a, const Candidate& b) { return a.score > b.score; };
    if (results.size() > k) {
        std::partial_sort(results.begin(),
                          results.begin() + static_cast<std::ptrdiff_t>(k), results.end(),
                          by_score);
        results.resize(k);
    } else {
        std::sort(results.begin(), results.end(), by_score);
    }
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double idx = p / 100.0 * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

} // namespace

int main() {
    std::cout << "layout=" << kLayout << '\n';
    std::cout << "corpus_chunks=" << kCorpusChunks << '\n';
    std::cout << "embedding_bytes=" << sizeof(std::array<float, EMBEDDING_DIM>) << '\n';

    const auto corpus = write_corpus(kCorpusChunks);

    {
        const auto before_alloc = snapshot_allocs();
        const auto before_zone = zone_stats();
        const auto t0 = std::chrono::steady_clock::now();
        File ingested(corpus);
        const auto t1 = std::chrono::steady_clock::now();
        const auto after_alloc = snapshot_allocs();
        const auto after_zone = zone_stats();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        const std::size_t n = ingested.chunk_count();
        std::cout << '\n' << "== ingestion (File constructor, no embed) ==\n";
        std::cout << "chunks=" << n << '\n';
        std::cout << "seconds=" << secs << '\n';
        std::cout << "chunks_per_second=" << (secs > 0.0 ? static_cast<double>(n) / secs : 0.0)
                  << '\n';
        std::cout << "alloc_calls=" << (after_alloc.calls - before_alloc.calls) << '\n';
        std::cout << "alloc_bytes_requested=" << (after_alloc.bytes - before_alloc.bytes) << '\n';
        std::cout << "zone_blocks_delta="
                  << (static_cast<long long>(after_zone.blocks_in_use) -
                      static_cast<long long>(before_zone.blocks_in_use))
                  << '\n';
        std::cout << "zone_size_in_use_delta="
                  << (static_cast<long long>(after_zone.size_in_use) -
                      static_cast<long long>(before_zone.size_in_use))
                  << '\n';
    }

    File file(corpus);
    fill_embeddings(file);
    const std::size_t n = file.chunk_count();

    {
        std::vector<Candidate> results;
        for (int q = 0; q < kSearchWarmup; ++q) {
            rank_topk(file, make_query(q), kTopK, results);
        }

        std::vector<double> lat_ms;
        lat_ms.reserve(static_cast<std::size_t>(kSearchQueries));
        const auto t0 = std::chrono::steady_clock::now();
        for (int q = 0; q < kSearchQueries; ++q) {
            const auto q0 = std::chrono::steady_clock::now();
            rank_topk(file, make_query(q + 1000), kTopK, results);
            const auto q1 = std::chrono::steady_clock::now();
            lat_ms.push_back(std::chrono::duration<double, std::milli>(q1 - q0).count());
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double total_s = std::chrono::duration<double>(t1 - t0).count();
        const double mean = std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) /
                            static_cast<double>(lat_ms.size());

        std::cout << '\n' << "== similarity search (scan + top-k, in-process embeddings) ==\n";
        std::cout << "queries=" << kSearchQueries << '\n';
        std::cout << "k=" << kTopK << '\n';
        std::cout << "queries_per_second="
                  << (total_s > 0.0 ? static_cast<double>(kSearchQueries) / total_s : 0.0) << '\n';
        std::cout << "latency_mean_ms=" << mean << '\n';
        std::cout << "latency_p50_ms=" << percentile(lat_ms, 50.0) << '\n';
        std::cout << "latency_p99_ms=" << percentile(lat_ms, 99.0) << '\n';
    }

    const auto rss = process_rss();
    const auto zone = zone_stats();
    std::cout << '\n' << "== memory (after ingest + filled embeddings) ==\n";
    std::cout << "chunks=" << n << '\n';
    std::cout << "rss_bytes=" << rss.resident_bytes << '\n';
    std::cout << "phys_footprint_bytes=" << rss.phys_footprint_bytes << '\n';
    std::cout << "zone_size_in_use=" << zone.size_in_use << '\n';
    std::cout << "zone_blocks_in_use=" << zone.blocks_in_use << '\n';
    std::cout << "rss_bytes_per_chunk="
              << (n ? static_cast<double>(rss.resident_bytes) / static_cast<double>(n) : 0.0)
              << '\n';
    std::cout << "zone_bytes_per_chunk="
              << (n ? static_cast<double>(zone.size_in_use) / static_cast<double>(n) : 0.0)
              << '\n';
    std::cout << "phys_footprint_bytes_per_chunk="
              << (n ? static_cast<double>(rss.phys_footprint_bytes) / static_cast<double>(n) : 0.0)
              << '\n';

    std::filesystem::remove(corpus);
    return 0;
}
