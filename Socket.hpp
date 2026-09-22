#ifndef CDB476FF_8020_462E_815F_B6DEB12AFEE5
#define CDB476FF_8020_462E_815F_B6DEB12AFEE5

#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
// A Network socket is an endpoint for sending or receiving data across a computer network
// A File Descriptor is a unique identifier for a file or socket in Unix-like operating systems
class Socket {
private:
    int m_fd;
public:
    // This uses a socket sys call to create a socket using UDP protocol
    // AF_NET is the address family for IPv4
    // 0 means the default protocol for the given socket type (SOCK_DGRAM for UDP)
    Socket() : m_fd{ socket(AF_INET, SOCK_DGRAM, 0) } {
        if (m_fd < 0) {
            throw std::runtime_error("Failed to create socket");
        }
        sockaddr_in server{}; // server is a struct that holds information about the server
        server.sin_family = AF_INET; // IPv4
        server.sin_port = htons(5000); // Port 5000, htons converts to network byte order

        inet_pton(AF_INET, "127.0.0.1", &server.sin_addr); // Localhost address

        // Server:
        // - IPV4
        // - IP: 127.0.0.1
        // - Port: 5000

        // Now, we connect the socket to the server.
        // reinterpret_cast is used to reinterpret the bits of one type as another type.
        // Here, we are casting sockaddr_in to sockaddr
        // connect is a syscall that connects the socket referred to by the file descriptor m_fd
        // to the address specified by server
        // connect returns an integer: 0 on success, -1 on error
        if (connect(m_fd, reinterpret_cast<sockaddr*>(&server), sizeof(server)) < 0) {
            throw std::runtime_error("Failed to connect to server");
        } else {
            std::cout << "Connected to server at PORT 5000!" << "\n";
            // ssize_t is a signed version of size_t, used for functions that return a count of bytes or an error code (-1)
            ssize_t bytes_sent = send(m_fd, "Hello, Server!", 14, 0); // Send a message to the server
            if (bytes_sent < 0) {
                throw std::runtime_error("Failed to send message to server");
            } 
        }
    }
};

// Test socket
int main() {
    try {
        Socket socket; // Create a socket object, which will connect to the server
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; // Print any errors that occur
        return 1; // Return a non-zero value to indicate failure
    }
    return 0; // Return zero to indicate success
}

#endif /* CDB476FF_8020_462E_815F_B6DEB12AFEE5 */
