
#include <edge_ai/runtime.hpp>
#include <edge_ai/frame_source.hpp>
#include <edge_ai/video_source.hpp>
#include <edge_ai/camera_source.hpp>

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

void test_null_source()
{
    bool exception_caught = false;

    try
    {
        edge_ai::Runtime runtime(std::unique_ptr<edge_ai::FrameSource>{});
    }
    catch (const std::invalid_argument &)
    {
        exception_caught = true;
    }

    if (!exception_caught)
    {
        throw std::runtime_error("Runtime accepted a null FrameSource");
    }

    std::cout << "[PASS] Null source rejected\n";
}

void test_invalid_camera()
{
    // 当前测试环境中已确认该设备编号不存在
    constexpr int invalid_device_index = 9999;

    auto source = std::make_unique<edge_ai::CameraSource>(invalid_device_index);

    edge_ai::Runtime runtime(std::move(source), edge_ai::QueuePolicy::LatestFrame);

    // 未打开摄像头时不应该启动工作线程
    if (runtime.start())
    {
        runtime.stop();
        runtime.join();

        throw std::runtime_error("Runtime started with an invalid camera");
    }

    // 沿用现有 Runtime 的输入源打开失败错误码
    if (runtime.exit_code() != 2)
    {
        throw std::runtime_error("Unexpected invalid camera exit code");
    }

    std::cout << "[PASS] Invalid camera handled\n";
}

void test_polymorphic_video(const std::string &video_path)
{
    // 用基类智能指针持有 VideoSource
    std::unique_ptr<edge_ai::FrameSource> source =
        std::make_unique<edge_ai::VideoSource>(video_path, 30.0);

    if (!source->is_open())
    {
        throw std::runtime_error("Failed to open test video");
    }

    // 将输入源所有权转移给 Runtime
    edge_ai::Runtime runtime(std::move(source), edge_ai::QueuePolicy::LatestFrame);

    // 验证原智能指针已经失去所有权
    if (source != nullptr)
    {
        throw std::runtime_error("FrameSource ownership was not transferred");
    }

    if (!runtime.start())
    {
        throw std::runtime_error("Polymorphic Runtime failed to start");
    }

    // 视频正常播放直到 EOF
    runtime.join();

    if (runtime.exit_code() != 0)
    {
        throw std::runtime_error("Polymorphic Runtime exited with error");
    }

    std::cout << "[PASS] Polymorphic video runtime\n";
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: runtime_frame_source_test <video_path>\n";
        return 1;
    }

    try
    {
        test_null_source();

        test_invalid_camera();

        test_polymorphic_video(argv[1]);

        std::cout << "\nRuntime FrameSource: ALL PASS\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}
