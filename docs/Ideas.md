# Ideas

## Client-Server Architecture

While the main client application is responsible for rendering and physics simulation,
There can be another 2 applications, one is the server, which is bootstrapped by the client if it can't
find a compatible server already running

- how do you know compatible server? Once you set the domain/address and port, you send a identification
message

The other 2 applications are the server itself, which is responsible to handling downloads and more
time consuming, non real-time tasks, and send commands to the client application received by the second
executable, the command line interface

Client And server should communicate with *protobuf* version 3

## External Libraries

- Boost Fibers -> User Level threading library
- Yoga -> Layout from HTML/CSS like files
- Volk, GPUOpen's Vulkan Memory Allocator
