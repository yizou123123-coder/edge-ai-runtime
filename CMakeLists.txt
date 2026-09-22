#这个工程要求使用 CMake 3.20 或更高版本来解释它
cmake_minimum_required(VERSION 3.20)

# Project name: edge_ai_runtime     
# Language: C++
project(edge_ai_runtime LANGUAGES CXX)

# 请用 app/main.cpp构建一个edge_runtime 可执行程序(二进制文件)
add_executable(edge_runtime
    app/main.cpp
)

# edge_runtime 这个 target 需要 C++17
target_compile_features(edge_runtime PRIVATE cxx_std_17)