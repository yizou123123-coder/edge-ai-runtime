cmake_minimum_required(VERSION 3.20)

project(edge_ai_runtime LANGUAGES CXX)
enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# 1. Build Options
# ============================================================

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer: ENABLED")

    add_compile_options(
        -fsanitize=address
        -fno-omit-frame-pointer
    )

    add_link_options(
        -fsanitize=address
    )
endif()

# ASan 和 TSan 不能同时开启
if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ASan and TSan cannot be enabled together")
endif()

if(ENABLE_TSAN)
    message(STATUS "ThreadSanitizer: ENABLED")

    add_compile_options(
        -fsanitize=thread
        -fno-omit-frame-pointer
    )

    add_link_options(
        -fsanitize=thread
    )
endif()


# ============================================================
# 2. Dependencies
# ============================================================

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    videoio
)

find_package(Threads REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
# Compatibility with older yaml-cpp CMake packages
if(NOT TARGET yaml-cpp::yaml-cpp)
    if(TARGET yaml-cpp)
        add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
    else()
        message(FATAL_ERROR "yaml-cpp target not found")
    endif()
endif()

set(ONNXRUNTIME_ROOT
    "${PROJECT_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "ONNX Runtime installation directory"
)

add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION
        "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"

    INTERFACE_INCLUDE_DIRECTORIES
        "${ONNXRUNTIME_ROOT}/include"
)


# ============================================================
# 3. Core Library
# ============================================================

add_library(edge_runtime_core
    src/logger.cpp
    src/video_source.cpp
    src/camera_source.cpp
    src/runtime.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/detector.cpp
    src/frame_source_factory.cpp
    src/config_loader.cpp
)

target_compile_features(edge_runtime_core
    PUBLIC cxx_std_17
)

target_include_directories(edge_runtime_core
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(edge_runtime_core
    PUBLIC
        ${OpenCV_LIBS}
        Threads::Threads
        onnxruntime

    PRIVATE
        yaml-cpp::yaml-cpp
)


# ============================================================
# 4. Main Application
# ============================================================

add_executable(edge_runtime
    app/main.cpp
)

target_link_libraries(edge_runtime
    PRIVATE edge_runtime_core
)


# ============================================================
# 5. Test Helper
# ============================================================
# 测试视频路径，由 CMake 配置参数提供
set(TEST_VIDEO_PATH "" CACHE FILEPATH "Video path for CTest")

# 需要视频路径参数的测试
set(VIDEO_ARG_TESTS
    inference_benchmark
    video_source_pacing_test
    runtime_frame_source_test
    frame_source_factory_test
    detector_threshold_test
)
# 适用于链接 edge_runtime_core 的测试
function(add_core_test target_name)

    add_executable(${target_name}
        "tests/${target_name}.cpp"
    )

    target_link_libraries(${target_name}
        PRIVATE edge_runtime_core
    )


    if(target_name IN_LIST VIDEO_ARG_TESTS)
        add_test(
            NAME ${target_name}
            COMMAND $<TARGET_FILE:${target_name}> "${TEST_VIDEO_PATH}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        )
    else()
        add_test(
            NAME ${target_name}
            COMMAND $<TARGET_FILE:${target_name}>
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        )
    endif()
endfunction()


# ============================================================
# 6. Core Tests
# ============================================================

foreach(test_name IN ITEMS
    runtime_lifecycle_test
    preprocess_test
    onnx_inference_test
    postprocess_test
    coordinate_restore_test
    inference_benchmark
    video_source_pacing_test
    drop_oldest_test
    latest_frame_test
    queue_policy_concurrency_test
    camera_source_test
    runtime_frame_source_test
    runtime_live_source_failure_test
    config_loader_test
    config_validation_test
    config_startup_test
    frame_source_factory_test
    runtime_queue_capacity_test
    runtime_parameters_test
    detector_threshold_test
    display_queue_test
)
    add_core_test(${test_name})
endforeach()

# 保留原有测试程序的 ONNX Runtime RPATH
set_target_properties(onnx_inference_test PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)


# ============================================================
# 7. Standalone Tests
# ============================================================

# BlockingQueue: 不依赖 Core Library
add_executable(blocking_queue_test
    tests/blocking_queue_test.cpp
)

target_compile_features(blocking_queue_test
    PRIVATE cxx_std_17
)

target_include_directories(blocking_queue_test
    PRIVATE "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(blocking_queue_test
    PRIVATE Threads::Threads
)


# ONNX Runtime Session
add_executable(onnx_session_test
    tests/onnx_session_test.cpp
)

target_compile_features(onnx_session_test
    PRIVATE cxx_std_17
)

target_link_libraries(onnx_session_test
    PRIVATE onnxruntime
)

set_target_properties(onnx_session_test PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)


# YAML Parser
add_executable(yaml_smoke_test
    tests/yaml_smoke_test.cpp
)

target_compile_features(yaml_smoke_test
    PRIVATE cxx_std_17
)

target_link_libraries(yaml_smoke_test
    PRIVATE yaml-cpp::yaml-cpp
)

# P6-02 portable ARM64 RUNPATH
if(CMAKE_CROSSCOMPILING AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    get_property(_p6_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_p6_target IN LISTS _p6_targets)
        get_target_property(_p6_type "${_p6_target}" TYPE)
        if(_p6_type STREQUAL "EXECUTABLE")
            set_target_properties("${_p6_target}" PROPERTIES
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH "$ORIGIN/../lib"
            )
        endif()
    endforeach()
endif()
