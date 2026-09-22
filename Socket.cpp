#include "Socket.hpp"
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

// Socket calls report failure by returning -1 and setting errno. This turns errno into
// readable text so an exception says what went wrong, not just where.
static std::runtime_error os_error(const std::string& what) {
    return std::runtime_error(what + ": " + std::strerror(errno));
}

Socket::Socket() : m_fd{ socket(AF_INET, SOCK_DGRAM, 0) } {
    if (m_fd < 0) {
        throw os_error("Failed to create socket");
    }

    sockaddr_in server{};
    server.sin_family = AF_INET; // IPv4
    server.sin_port = htons(5000); // htons converts to network byte order (big endian)
    inet_pton(AF_INET, "127.0.0.1", &server.sin_addr); // Localhost address

    // connect() on a UDP socket sends nothing -- UDP has no handshake. It just records the
    // peer address, which lets us use send/recv instead of sendto/recvfrom, and makes
    // "nothing is listening" come back as ECONNREFUSED instead of blocking forever.
    if (::connect(m_fd, reinterpret_cast<sockaddr*>(&server), sizeof(server)) < 0) {
        ::close(m_fd); // The constructor is throwing, so ~Socket() will not run
        throw os_error("Failed to connect to 127.0.0.1:5000");
    }

    // UDP never retransmits, so a dropped reply would block recv forever without this.
    timeval timeout{};
    timeout.tv_sec = 5;
    setsockopt(m_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
}

Socket::~Socket() {
    close(m_fd); // Close the socket when the object is destroyed - prevent resource leaks
}

void Socket::send(const std::string& message) {
    // ::send, not sendto, because connect() already told the kernel where to aim.
    // The leading :: reaches the global C function past this class's own send().
    if (::send(m_fd, message.c_str(), message.size(), 0) < 0) {
        throw os_error("Failed to send message");
    }
}

void Socket::receive(float* embedding, std::size_t size) {
    const std::size_t expected = size * sizeof(float);
    ssize_t bytes_received = ::recv(m_fd, embedding, expected, 0);
    if (bytes_received < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) { // SO_RCVTIMEO expired
            throw std::runtime_error("Timed out. Is embedding_server.py running?");
        }
        throw os_error("Failed to receive message");
    }
    if (static_cast<std::size_t>(bytes_received) != expected) {
        throw std::runtime_error("Received " + std::to_string(bytes_received) +
                                 " bytes, expected " + std::to_string(expected));
    }
}
