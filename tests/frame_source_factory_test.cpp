
#include <edge_ai/frame_source_factory.hpp>
#include <edge_ai/runtime_config.hpp>
#include <edge_ai/frame_source.hpp>

#include <opencv2/core.hpp>

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

// Test 1：创建视频输入源
void test_video_source(const std::string& video_path)
{
    edge_ai::SourceConfig config{};

    config.type = edge_ai::SourceType::Video;
    config.video_path = video_path;
    config.target_fps = 30.0;

    std::unique_ptr<edge_ai::FrameSource> source =
        edge_ai::create_frame_source(config);

    require(source != nullptr, "VideoSource is null");
    require(source->is_open(), "VideoSource open failed");
    require(!source->is_live(), "VideoSource must not be live");

    cv::Mat frame;

    require(source->read(frame), "VideoSource read failed");
    require(!frame.empty(), "VideoSource returned empty frame");

    require(
        source->skipped_slots() == 0,
        "Unexpected initial source skips"
    );

    std::cout << "[PASS] Factory creates VideoSource\n";
}

// Test 2：创建摄像头输入源，验证设备打开失败
void test_camera_source()
{
    edge_ai::SourceConfig config{};

    config.type = edge_ai::SourceType::Camera;

    // 当前测试环境中已验证此编号不存在
    config.camera_index = 9999;

    std::unique_ptr<edge_ai::FrameSource> source =
        edge_ai::create_frame_source(config);

    require(source != nullptr, "CameraSource is null");
    require(source->is_live(), "CameraSource must be live");
    require(!source->is_open(), "Invalid camera unexpectedly opened");

    cv::Mat frame(
        2, 2, CV_8UC3,
        cv::Scalar(255, 0, 0)
    );

    require(!source->read(frame), "Invalid camera read succeeded");
    require(frame.empty(), "Failed camera read left stale image");

    require(
        source->skipped_slots() == 0,
        "Unexpected CameraSource skips"
    );

    std::cout << "[PASS] Factory creates CameraSource\n";
}

// Test 3：非法输入源枚举必须拒绝
void test_invalid_source_type()
{
    edge_ai::SourceConfig config{};

    config.type = static_cast<edge_ai::SourceType>(999);

    bool rejected = false;

    try
    {
        auto source = edge_ai::create_frame_source(config);
        (void)source;
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(
        rejected,
        "Factory accepted an invalid SourceType"
    );

    std::cout << "[PASS] Invalid SourceType rejected\n";
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr
            << "Usage: frame_source_factory_test <video_path>\n";
        return 1;
    }

    try
    {
        test_video_source(argv[1]);
        test_camera_source();
        test_invalid_source_type();

        std::cout
            << "\nFrameSource Factory: ALL PASS\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
