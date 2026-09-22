import socket
from sentence_transformers import SentenceTransformer

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000
DIMENSION = 384  # Dimension of the embedding vector for 'all-MiniLM-L6-v2'

# The largest UDP payload that can ever be delivered. Sizing the buffer to this means a
# long source line is never silently truncated -- recvfrom discards whatever does not fit
# and gives no indication it happened.
MAX_DATAGRAM = 65535

model = SentenceTransformer('all-MiniLM-L6-v2')

class Data:
    # One socket, not two. A UDP socket is bidirectional: the same socket that receives a
    # request is the one that replies to it. The previous two-socket split sent replies to
    # a fixed port 5000 -- the server's own listening port -- so the embedding came back to
    # this process instead of reaching the client.
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create a UDP socket
        self.sock.bind((ip, port)) # Bind the socket to the specified address and port

    def receive_data(self):
        data, addr = self.sock.recvfrom(MAX_DATAGRAM) # Receive data from the socket
        # addr is the client's (ip, port). The C++ client never binds, so the kernel gives
        # it a random ephemeral port -- this tuple is the only way to know where to reply.
        return data.decode('utf-8', errors='replace'), addr # Decode the received bytes to a string

    def embed_text(self, text):
        # normalize_embeddings=True scales the vector to unit length, which is what makes a
        # plain dot product on the C++ side equal cosine similarity. Without it the clamp in
        # VectorMath::dot_product would be silently clipping real values.
        embedding = model.encode(text, normalize_embeddings=True) # Generate the embedding
        return embedding.astype('float32').tobytes() # Convert to raw float32 bytes for sending

    def send_data(self):
        text, addr = self.receive_data()
        self.sock.sendto(self.embed_text(text), addr) # Reply to whoever asked
        return text, addr

    def serve_forever(self):
        print(f"listening on {SERVER_IP}:{SERVER_PORT}", flush=True)
        while True:
            try:
                text, addr = self.send_data()
                preview = text[:60].replace("\n", " ")
                print(f"  {addr} -> embedded {len(text)} chars: {preview!r}", flush=True)
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
