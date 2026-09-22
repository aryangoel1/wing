#include "Chunk.hpp"
#include "Socket.hpp"
#include "VectorMath.hpp"
#include <array>
#include <print>
#include <string>

int main() {
    try {
        Socket socket; // Connects to the Python embedding server on 127.0.0.1:5000

        const std::string text =
            "Aryan Goel is a software engineer with a passion for C++ and high-performance computing. ";

        socket.send(text); // Ship the text over as one UDP datagram

        std::array<float, EMBEDDING_DIM> embedding{};
        socket.receive(embedding.data(), embedding.size()); // Fill it with the reply

        std::println("Sent:     \"{}\"", text);
        std::println("Received: {} floats", embedding.size());
        std::print("First 20:  ");
        for (std::size_t i = 0; i < 20; ++i) {
            std::print("{:+.5f} ", embedding[i]);
        }
        std::println("...");

        // A normalized vector dotted with itself is 1.0. If this prints 1.0 then the bytes
        // survived the trip intact and the SIMD dot product agrees with them.
        std::println("Norm^2:   {:.6f}", VectorMath::dot_product(embedding, embedding));
    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }
    return 0;
}
