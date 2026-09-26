#ifndef CDB476FF_8020_462E_815F_B6DEB12AFEE5
#define CDB476FF_8020_462E_815F_B6DEB12AFEE5

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>

// A Network socket is an endpoint for sending or receiving data across a computer network
// A File Descriptor is a unique identifier for a file or socket in Unix-like operating systems
// stdin - o; stdout - 1; stderr - 2
class Socket {
private:
    int m_fd;
public:
    // This uses a socket sys call to create a socket using UDP protocol
    // AF_INET is the address family for IPv4
    // 0 means the default protocol for the given socket type (SOCK_DGRAM for UDP)
    Socket();
    ~Socket(); 
    Socket(const Socket&) = delete; // Disable copy constructor - we dont want 2 sockets pointing to the same file descriptor
    Socket& operator=(const Socket&) = delete; // Disable copy assignment - we dont two sockets - singleton

    // The send function here will actually send the text belonging to a chunk to the server
    void send(const std::string& message);

    // We now need a function that receives the embedding from the server and then stores it in the
    // chunk. float* embedding is where the data will be stored
    // size
    void receive(float* embedding, std::size_t size);
    int get_fd() const { return m_fd; }
};

#endif /* CDB476FF_8020_462E_815F_B6DEB12AFEE5 */