#ifndef B2069796_5717_4F87_816B_9C9E2F02C162
#define B2069796_5717_4F87_816B_9C9E2F02C162

#include <cstddef>

// all-MiniLM-L6-v2 produces 384-dimensional embeddings.
// Baked into the type, so swapping models means changing this and recompiling.
//
// inline + constexpr allows a global variable to be evaluated at compile time
// while also allowing it to be defined in a header file without violating the One Definition Rule (ODR)
inline constexpr std::size_t EMBEDDING_DIM = 384;

#endif /* B2069796_5717_4F87_816B_9C9E2F02C162 */
