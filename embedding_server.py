import socket
from sentence_transformers import SentenceTransformer
import torch

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000
DIMENSION = 384  # Dimension of the embedding vector for 'all-MiniLM-L6-v2'

# The largest UDP payload that can ever be delivered. Sizing the buffer to this means a
# long source line is never silently truncated -- recvfrom discards whatever does not fit
# and gives no indication it happened.
MAX_DATAGRAM = 65535
# Matches batch_size in File::embed. The client sends this many lines before it reads
# any replies, so up to this many can be sitting in the socket at once.
MAX_BATCH = 42

model = SentenceTransformer('all-MiniLM-L6-v2', device="mps") # Use Metal GPU


class Data:
    # One socket, not two. A UDP socket is bidirectional: the same socket that receives a
    # request is the one that replies to it. The previous two-socket split sent replies to
    # a fixed port 5000 -- the server's own listening port -- so the embedding came back to
    # this process instead of reaching the client.
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create a UDP socket
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 10485760) # Request a 10mb buffer for the python socket   
        self.sock.bind((ip, port)) # Bind the socket to the specified address and port

    def receive_batch(self):
        # Each datagram is one line, and recvfrom only ever returns one datagram. So block
        # for the first, then drain whatever else is already queued without waiting. There
        # is no "end of batch" marker on the wire: a search query arrives alone, and a file's
        # last batch is short, so waiting for a full MAX_BATCH would hang on both.
        data, addr = self.sock.recvfrom(MAX_DATAGRAM)
        requests = [(data, addr)]
        self.sock.setblocking(False)
        try:
            while len(requests) < MAX_BATCH:
                requests.append(self.sock.recvfrom(MAX_DATAGRAM))
        except BlockingIOError:
            pass  # Queue is empty -- embed what we have
        finally:
            self.sock.setblocking(True)
        return requests

    def embed_texts(self, texts):
        # One encode call for the whole batch. The per-call overhead (Python, tokenizer,
        # tensor setup) is paid once instead of once per line, which is where the speedup is.
        # normalize_embeddings=True scales each vector to unit length, which is what makes a
        # plain dot product on the C++ side equal cosine similarity.
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=MAX_BATCH)
        return [embedding.astype('float32').tobytes() for embedding in embeddings]

    def serve_batch(self):
        requests = self.receive_batch()
        texts = [data.decode('utf-8', errors='replace') for data, _ in requests]
        # Replies go out in arrival order, one per request. The client matches replies to
        # chunks by order, so this must not be reordered.
        for embedding, (_, addr) in zip(self.embed_texts(texts), requests):
            self.sock.sendto(embedding, addr)
        return texts

    def serve_forever(self):
        print(f"listening on {SERVER_IP}:{SERVER_PORT}", flush=True)
        while True:
            try:
                texts = self.serve_batch()
                preview = texts[0][:60].replace("\n", " ")
                print(f"  embedded batch of {len(texts)}: {preview!r}", flush=True)
            except (UnicodeDecodeError, OSError) as e:
                # One malformed request should not take the server down mid-index.
                print(f"  request failed: {e}", flush=True)


if __name__ == "__main__":
    actual = model.get_sentence_embedding_dimension()
    if actual != DIMENSION:
        raise SystemExit(
            f"Model dimension is {actual} but DIMENSION is {DIMENSION}. "
            f"Update DIMENSION here and EMBEDDING_DIM in Chunk.hpp to match."
        )
    print(f"model ready, dim = {actual}", flush=True)
    Data(SERVER_IP, SERVER_PORT).serve_forever()