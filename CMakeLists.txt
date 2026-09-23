#这个工程要求使用 CMake 3.20 或更高版本来解释它
cmake_minimum_required(VERSION 3.20)

# Project name: edge_ai_runtime     
# Language: C++
project(edge_ai_runtime LANGUAGES CXX)

# 请用 app/main.cpp构建一个edge_runtime 可执行程序(二进制文件)
add_executable(edge_runtime
    app/main.cpp
)

# 让 CMake 在当前系统中寻找已经安装的 OpenCV，并读取它提供的构建信息
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    videoio
)

# edge_runtime 这个 target 需要 C++17
target_compile_features(edge_runtime PRIVATE cxx_std_17)

# 编译器去哪寻找 OpenCV 头文件
target_include_directories(edge_runtime PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# 最终链接 edge_runtime 时，需要链接哪些 OpenCV library
target_link_libraries(edge_runtime PRIVATE
    ${OpenCV_LIBS}
)