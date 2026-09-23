cmake_minimum_required(VERSION 3.20)
project(wing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # -std=c++23, not -std=gnu++23

# Writes build/compile_commands.json, which is how clangd and VS Code learn the
# include paths and flags. Without it, editor autocomplete half-guesses.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Plain `cmake -B build` with no build type would otherwise compile unoptimized.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_executable(wing
    main.cpp
    Socket.cpp
    File.cpp
    VectorSearch.cpp
)

target_compile_options(wing PRIVATE -Wall -Wextra)

# Debug builds turn on the sanitizers that caught the heap-use-after-free in
# SearchResult. They have to reach the linker as well as the compiler, or the
# runtime library never gets linked in.
target_compile_options(wing PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined -fno-omit-frame-pointer>)
target_link_options(wing    PRIVATE $<$<CONFIG:Debug>:-fsanitize=address,undefined>)

# `cmake --install build --prefix ~/.local` copies the binary to <prefix>/bin.
# A symlink into ~/.local/bin is better while developing though, since it always
# points at the newest build instead of needing a reinstall after every change.
install(TARGETS wing RUNTIME DESTINATION bin)
