cmake_minimum_required(VERSION 3.20)

project(edge_ai_runtime LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    videoio
)


# 1. 创建核心库 target
add_library(edge_runtime_core
    src/logger.cpp
    src/video_source.cpp
    src/runtime.cpp
)


# 2. 配置核心库
target_compile_features(edge_runtime_core PUBLIC cxx_std_17)

target_include_directories(edge_runtime_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(edge_runtime_core
    PUBLIC
        ${OpenCV_LIBS}
)


# 3. 创建可执行程序
add_executable(edge_runtime
    app/main.cpp
)


# 4. 配置可执行程序
target_compile_features(edge_runtime PRIVATE cxx_std_17)

target_link_libraries(edge_runtime
    PRIVATE
        edge_runtime_core
)