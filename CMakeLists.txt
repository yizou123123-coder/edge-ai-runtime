cmake_minimum_required(VERSION 3.20)
project(edge_ai_runtime LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 寻找外部第三方依赖
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    videoio
)
find_package(Threads REQUIRED)

set(ONNXRUNTIME_ROOT
    ${PROJECT_SOURCE_DIR}/third_party/onnxruntime
)


# 打包核心功能库（edge_runtime_core） 
# 1. 创建核心库（把这3个.cpp打包成一个库）
add_library(edge_runtime_core
    src/logger.cpp
    src/video_source.cpp
    src/runtime.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/detector.cpp
)

# 2. 规定编译必须启用 C++17 特性
target_compile_features(edge_runtime_core PUBLIC cxx_std_17)

# 3. 指定去哪里找头文件（include 目录以及 OpenCV 的头文件目录）
target_include_directories(edge_runtime_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

# 4. 把外部依赖链接进这个核心库
target_link_libraries(edge_runtime_core
    PUBLIC
        ${OpenCV_LIBS}
        Threads::Threads
        onnxruntime
)


# 构建正式运行程序（edge_runtime）
add_executable(edge_runtime
    app/main.cpp
)
target_compile_features(edge_runtime PRIVATE cxx_std_17)
target_link_libraries(edge_runtime
    PRIVATE
        edge_runtime_core
)


# 构建两个独立的测试程序
add_executable(blocking_queue_test
    tests/blocking_queue_test.cpp
)
target_compile_features(
    blocking_queue_test
    PRIVATE
        cxx_std_17
)
target_include_directories(
    blocking_queue_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(
    blocking_queue_test
    PRIVATE
        Threads::Threads
)


add_executable(runtime_lifecycle_test
    tests/runtime_lifecycle_test.cpp
)
target_compile_features(
    runtime_lifecycle_test
    PRIVATE
        cxx_std_17
)
target_link_libraries(runtime_lifecycle_test
    PRIVATE
        edge_runtime_core
)

# onnx测试程序
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(
    onnxruntime
    PROPERTIES
        IMPORTED_LOCATION
            ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES
            ${ONNXRUNTIME_ROOT}/include

)

add_executable(onnx_session_test
    tests/onnx_session_test.cpp
)
target_compile_features(onnx_session_test
    PRIVATE
        cxx_std_17
)
target_link_libraries(onnx_session_test
    PRIVATE
        onnxruntime 
)
set_target_properties(onnx_session_test
    PROPERTIES
        BUILD_RPATH
            ${ONNXRUNTIME_ROOT}/lib
)

# 预处理测试程序
add_executable(preprocess_test
    tests/preprocess_test.cpp
)

target_compile_features(preprocess_test
    PRIVATE
        cxx_std_17
)

target_link_libraries(preprocess_test
    PRIVATE
        edge_runtime_core
)

# 测试run
add_executable(onnx_inference_test
    tests/onnx_inference_test.cpp
)
target_compile_features(onnx_inference_test
    PRIVATE
        cxx_std_17
)
target_link_libraries(onnx_inference_test
    PRIVATE
        edge_runtime_core
        onnxruntime
)
set_target_properties(onnx_inference_test
    PROPERTIES
        BUILD_RPATH
            ${ONNXRUNTIME_ROOT}/lib
)

# 测试后处理
add_executable(postprocess_test
    tests/postprocess_test.cpp
)

target_compile_features(postprocess_test
    PRIVATE cxx_std_17
)

target_link_libraries(postprocess_test
    PRIVATE edge_runtime_core
)
# 3.6 test
add_executable(
    coordinate_restore_test
    tests/coordinate_restore_test.cpp
)

target_link_libraries(
    coordinate_restore_test
    PRIVATE edge_runtime_core
)
# 3.7 test
add_executable(
    inference_benchmark
    tests/inference_benchmark.cpp
)

target_link_libraries(
    inference_benchmark
    PRIVATE edge_runtime_core
)