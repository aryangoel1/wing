import socket

# 1. Create a UDP socket (SOCK_DGRAM)
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 2. Bind to the local address and port
server.bind(('127.0.0.1', 5000))
print('UDP server up and listening')

# 3. Receive data and client address without .listen() or .accept()
while True:
  # recvfrom returns a tuple (data, address)
  data, addr = server.recvfrom(1024)
  print(f'Received message from {addr}: {data.decode()}')

  # 4. Send a reply using sendto
  server.sendto(b'Hello from UDP server', addr)
