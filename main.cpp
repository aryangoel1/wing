#include "VectorSearch.hpp"
#include "Chunk.hpp"
#include "Socket.hpp"
#include <chrono>
#include <print>
#include <string>

namespace {

// ANSI escape codes. "\033[" opens the sequence, the number selects a style, "m" ends it.
// 1 = bold, 2 = dim, 36 = cyan, 0 = reset everything back to normal. Terminals that do not
// understand them print nothing, so this degrades quietly rather than spraying garbage.
constexpr const char* BOLD  = "\033[1m";
constexpr const char* DIM   = "\033[2m";
constexpr const char* CYAN  = "\033[36m";
constexpr const char* RESET = "\033[0m";

void printBanner() {
    std::println();
    std::println("{}{}  ██     ██ ██ ███    ██  ██████ {}", BOLD, CYAN, RESET);
    std::println("{}{}  ██  █  ██ ██ ████   ██ ██      {}", BOLD, CYAN, RESET);
    std::println("{}{}  ██ ███ ██ ██ ██ ██  ██ ██   ███{}", BOLD, CYAN, RESET);
    std::println("{}{}  ████ ████ ██ ██  ██ ██ ██    ██{}", BOLD, CYAN, RESET);
    std::println("{}{}  ███   ███ ██ ██   ████  ██████ {}", BOLD, CYAN, RESET);
    std::println();
    std::println("  {}local semantic code search{}", BOLD, RESET);
    std::println("  {}all-MiniLM-L6-v2 · {}-dim · NEON SIMD · C++23{}", DIM, EMBEDDING_DIM, RESET);
    std::println();
}

void printUsage(const char* program) {
    std::println("  {}Usage{}", BOLD, RESET);
    std::println("    {}{} add{}    <path>           {}index a file and report timing{}",
                 CYAN, program, RESET, DIM, RESET);
    std::println("    {}{} search{} <path> <query>   {}index a file, then rank its lines{}",
                 CYAN, program, RESET, DIM, RESET);
    std::println();
    std::println("  {}Examples{}", BOLD, RESET);
    std::println("    {}{} add VectorMath.hpp{}", DIM, program, RESET);
    std::println("    {}{} search VectorMath.hpp \"SIMD dot product\"{}", DIM, program, RESET);
    std::println();
}

} // namespace

int main(int argc, char* argv[]) {
    // argc = argument count - the program-name is the first argument
    // argv = array of strings containing the arguments - argv[0] will hold the name of the program
    printBanner();
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    const std::string command = argv[1];

    try {
        Socket socket;      // one connection for the whole run
        VectorSearch db;
        if (command == "add") {
            if (argc < 3) {
                printUsage(argv[0]);
                return 1;
            }
            const auto start = std::chrono::steady_clock::now();
            db.addFile(argv[2]);
            db.embedAll(socket);
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::println("  indexed {}{}{} in {}{:.2f}s{}", BOLD, argv[2], RESET, BOLD, secs, RESET);
            std::println();

        } else if (command == "search") {
            if (argc < 4) {
                printUsage(argv[0]);
                return 1;
            }
            db.addFile(argv[2]);
            db.embedAll(socket);

            const auto hits = db.search(socket, argv[3], 5);
            std::println("  {}\"{}\"{}", BOLD, argv[3], RESET);
            std::println();
            for (const auto& hit : hits) {
                std::println("  {}{:+.4f}{}  {}{}:{}{}  {}",
                             CYAN, hit.score, RESET,
                             DIM, hit.chunk->file_path, hit.chunk->id, RESET,
                             hit.chunk->text);
            }
            std::println();
        } else {
            std::println("  unknown command: {}{}{}", BOLD, command, RESET);
            std::println();
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::println("  {}Error:{} {}", BOLD, RESET, e.what());
        return 1;
    }
    return 0;
}
