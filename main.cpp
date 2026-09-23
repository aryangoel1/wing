#include "VectorSearch.hpp"
#include "Chunk.hpp"
#include "Socket.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <mach-o/dyld.h>   // _NSGetExecutablePath, macOS only
#include <print>
#include <stdexcept>
#include <string>
#include <vector>


// These things are only visible inside of main.cpp
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

// Where the binary itself lives. argv[0] is just "wing" when it is found on PATH,
// so it cannot tell us that. canonical() follows the ~/.local/bin/wing symlink back to
// the real build/wing, which is what lets a bare filename resolve no matter which
// directory the user ran from.
std::filesystem::path executableDir() {
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);      // first call just asks for the length
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return std::filesystem::current_path();
    }
    return std::filesystem::canonical(buffer.c_str()).parent_path();
}

// Turn whatever the user typed into a real file. A path that already points at a file
// is used as-is, so absolute and relative paths keep working. Otherwise we look for a
// file of that name in a few sensible places, which is what makes "wing search
// cooking.txt" work instead of requiring "samples/cooking.txt".
std::filesystem::path resolvePath(const std::string& name) {
    namespace fs = std::filesystem;

    // is_regular_file rather than exists(): a directory "exists" and ifstream will
    // happily open one, then read nothing and hand back an empty index.
    if (fs::is_regular_file(name)) {
        return name;
    }

    const fs::path here = fs::current_path(); 
    const fs::path root = executableDir().parent_path();    // build/ -> project root

    // Deduplicated because running from the project directory makes here == root,
    // which would otherwise list every location twice in the error message.
    std::vector<fs::path> search_path{ here, here / "samples", root, root / "samples" };
    std::sort(search_path.begin(), search_path.end());
    search_path.erase(std::unique(search_path.begin(), search_path.end()), search_path.end());

        for (const auto& base : search_path) {
        if (const fs::path candidate = base / name; fs::is_regular_file(candidate)) {
            return candidate;
        }
    }

    std::string message = "No file named '" + name + "'. Looked in:";
    for (const auto& base : search_path) {
        message += "\n    " + base.string();
    }
    throw std::runtime_error(message);
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
            const auto path = resolvePath(argv[2]);
            const auto start = std::chrono::steady_clock::now();
            db.addFile(path);
            db.embedAll(socket);
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::println("  indexed {}{}{} in {}{:.2f}s{}", BOLD, path.string(), RESET, BOLD, secs, RESET);
            std::println();

        } else if (command == "search") {
            if (argc < 4) {
                printUsage(argv[0]);
                return 1;
            }
            db.addFile(resolvePath(argv[2]));
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
