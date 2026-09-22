# ARM Linux Edge AI Real-time Inference Runtime

A Linux/C++17 real-time edge AI inference runtime targeting x86-64 and ARM64 Linux platforms.

## Development Environment

- Ubuntu 24.04 LTS
- GCC 13
- CMake 3.28+
- C++17

## Build

### Debug

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug
./build/debug/edge_runtime