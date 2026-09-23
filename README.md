# wing

A semantic code search engine in C++23. Ask a question in plain English; get back the
lines that *mean* the same thing, even when they share no words with your query.

```console
$ wing search cooking.txt "how do I get bread to rise"

  scanned 15 chunks in 0.002 ms  (query embedding: 6.42 ms)
  "how do I get bread to rise"

  +0.5136  samples/cooking.txt:1   Let the dough rest in a warm spot until it has doubled in size.
  +0.3706  samples/cooking.txt:7   Toast whole spices in a dry pan to wake up their oils before grinding.
  +0.3163  samples/cooking.txt:2   Sear the meat over high heat first to develop a deep brown crust.
```

The top hit contains no "bread", no "rise", no "get". `grep` returns nothing here.

It works on source code too:

```console
$ wing search Socket.cpp "close the file descriptor"

  +0.4613  Socket.cpp:46   close(m_fd); // Close the socket when the object is destroyed
  +0.3488  Socket.cpp:35   ::close(m_fd); // The constructor is throwing, so ~Socket() will not run
  +0.2075  Socket.cpp:32   // it also acts as a bookmark, allowing us to call send() and recv()
```

## What it is (and isn't)

wing is a **vector search engine**, not a vector database. It holds embeddings in memory
and scans all of them exactly — the equivalent of Faiss's `IndexFlatIP`. There is no
persistence, no delete, and no approximate-nearest-neighbour index. Those are deliberate
omissions, explained under [Design notes](#design-notes).

## How it works

```
  ┌─────────────────────────── C++23 ────────────────────────────┐
  │                                                              │
  │   File ──▶ Chunk[]  ────────▶  Socket  ────────▶             │   UDP     ┌── Python ──┐
  │   one chunk per line          RAII fd owner,   ──────────────┼──────────▶│ all-MiniLM │
  │                               connect()ed UDP                │  text     │   -L6-v2   │
  │                                                              │◀──────────│            │
  │   VectorSearch ◀── 384 floats stored inline in each Chunk    │ 1536 B    └────────────┘
  │        │                                                     │
  │        └──▶ VectorMath::dot_product  ──▶  top-k              │
  │             hand-written ARM NEON                            │
  └──────────────────────────────────────────────────────────────┘
```

1. `File` splits a text file into one `Chunk` per non-blank line.
2. `Socket` ships each line to a Python sidecar over a connected UDP socket.
3. The sidecar runs `all-MiniLM-L6-v2` and returns 384 normalised `float32`s.
4. `VectorMath::dot_product` scores the query against every chunk using ARM NEON
   intrinsics. Because the vectors are unit-length, the dot product *is* cosine similarity.
5. `VectorSearch` ranks with `std::partial_sort` and returns the top *k*.

## Build

Requires a C++23 compiler (tested with Apple clang 21), CMake 3.20+, and Python 3.9+.

```bash
# C++ side
cmake -B build
cmake --build build

# Python side
python3 -m venv .venv
.venv/bin/pip install sentence-transformers
```

Optionally put it on your `PATH`:

```bash
ln -s "$PWD/build/wing" ~/.local/bin/wing
```

A debug build with AddressSanitizer and UBSan:

```bash
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-debug
```

## Usage

Start the embedding server first — it takes about ten seconds to load the model:

```bash
.venv/bin/python embedding_server.py
```

Then, in another terminal:

```bash
wing add    <path>           # index a file and report timing
wing search <path> <query>   # index a file, then rank its lines
```

A bare filename is resolved against the current directory and `samples/`, so
`wing search ocean.txt "animal camouflage"` works from anywhere.

## Performance

All measured on an M3 Pro, `-O3`.

**Hand-written NEON vs. the naive scalar loop** — 200,000 chunks × 384 dims:

| | time | |
|---|---|---|
| NEON, 4 accumulators, 16 floats/iteration | 9.86 ms | **5.75×** |
| naive scalar loop | 56.70 ms | |

Four independent accumulators break the dependency chain between fused multiply-adds, so
the CPU can keep several in flight. clang does not autovectorise the scalar version to
equivalent quality.

**Memory layout** — storing embeddings as `std::array<float, 384>` *inside* `Chunk`
rather than `std::vector<float>`, measured over 50,000 chunks:

| layout | `sizeof(Chunk)` | scan |
|---|---|---|
| `std::vector<float>` member | 80 B | 1.77 ms |
| `std::array<float, 384>` member | 1592 B | **1.19 ms** |

A `std::vector` member stores a *pointer*; the floats live in a separate heap allocation,
so a scan chases one pointer per chunk and the prefetcher can't predict the pattern.
Inline storage turns the scan into a fixed-stride walk. Counting lines up front and
calling `reserve()` made ingestion a further **4.2×** faster.

**Scan time vs. corpus size:**

| chunks | index size | scan |
|---|---|---|
| 10,000 | 15 MB | 0.5 ms |
| 100,000 | 147 MB | 2.9 ms |
| 1,000,000 | 1.5 GB | 22 ms |

## Design notes

**Why no ANN index.** Brute-force search over 100,000 chunks takes 2.9 ms, while indexing
runs at roughly 6–11 ms *per line*. Search is four to five orders of magnitude away from
being the bottleneck, so HNSW would optimise the wrong thing. Exact search also means no
recall loss.

**Why a Python sidecar.** The model ecosystem lives in Python. Keeping inference in a
separate process means the C++ side stays a pure systems exercise, and the model could be
moved to another machine without touching it. The cost is a round trip per line.

**Why UDP.** One datagram is one message — the kernel preserves message boundaries, so no
length-prefix framing or partial-read loops are needed. An embedding is 1536 bytes, well
inside the limit. `connect()` on the UDP socket sends nothing but registers the peer, so a
dead server returns `ECONNREFUSED` in ~10 ms instead of blocking forever.

The trade-off: macOS caps datagrams at 9216 bytes (`net.inet.udp.maxdgram`), which limits
batching to six embeddings per reply and rejects any source line longer than ~9 KB.
Batching properly would require TCP.

## Limitations

- **No persistence.** The index is rebuilt on every run, so `search` takes a path.
- **One chunk per line.** Simple, but individual lines carry little context. Sliding
  windows with overlap would retrieve better.
- **Single file per invocation.** No directory traversal yet.
- **Sequential embedding.** ~6–11 ms per line with no batching and no retry, so indexing a
  large file is slow and a single dropped datagram aborts the run.
- **Apple Silicon and macOS only.** The kernel uses ARM NEON intrinsics, and path
  resolution uses `_NSGetExecutablePath`.
- **CRLF files** are mishandled: `std::getline` leaves the `\r`, so blank lines become
  chunks and every line carries a trailing carriage return.

## Layout

| file | |
|---|---|
| `Chunk.hpp` | one line of text plus its 384-float embedding, stored inline |
| `VectorMath.hpp` | NEON dot product |
| `Socket.hpp/.cpp` | RAII file-descriptor owner, connected UDP, timeouts |
| `File.hpp/.cpp` | reads a file into chunks, embeds them |
| `VectorSearch.hpp/.cpp` | holds the files, scores and ranks a query |
| `main.cpp` | CLI |
| `embedding_server.py` | the sidecar |
| `samples/` | text files chosen to show semantic rather than keyword matching |
