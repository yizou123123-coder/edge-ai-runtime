
#include <edge_ai/config_loader.hpp>
#include <edge_ai/runtime_config.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

using edge_ai::RuntimeConfig;
using edge_ai::SourceType;

// 验证合法配置不会抛异常
void expect_valid(const RuntimeConfig &config, const std::string &case_name)
{
    edge_ai::validate_startup_config(config);

    std::cout << "[PASS] " << case_name << '\n';
}

// 验证非法配置必须抛出 invalid_argument
void expect_invalid(const RuntimeConfig &config, const std::string &case_name)
{
    bool rejected = false;

    try
    {
        edge_ai::validate_startup_config(config);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }

    if (!rejected)
    {
        throw std::runtime_error("Invalid config accepted: " + case_name);
    }

    std::cout << "[PASS] " << case_name << '\n';
}

int main()
{
    try
    {
        // Test 1：Video + 合法视频路径
        {
            RuntimeConfig config{};

            config.source.type = SourceType::Video;
            config.source.video_path = "test.mp4";
            config.model.path = "models/yolov8n.onnx";

            expect_valid(config, "Video with valid paths");
        }

        // Test 2：Video + 空视频路径
        {
            RuntimeConfig config{};

            config.source.type = SourceType::Video;
            config.source.video_path = "";

            expect_invalid(config, "Video with empty video path rejected");
        }

        // Test 3：Camera + 空视频路径是允许的
        {
            RuntimeConfig config{};

            config.source.type = SourceType::Camera;
            config.source.video_path = "";
            config.source.camera_index = 0;
            config.model.path = "models/yolov8n.onnx";

            expect_valid(config, "Camera without video path");
        }

        // Test 4：Video + 空模型路径
        {
            RuntimeConfig config{};

            config.source.type = SourceType::Video;
            config.source.video_path = "test.mp4";
            config.model.path = "";

            expect_invalid(config, "Video with empty model path rejected");
        }

        // Test 5：Camera + 空模型路径
        {
            RuntimeConfig config{};

            config.source.type = SourceType::Camera;
            config.source.video_path = "";
            config.model.path = "";

            expect_invalid(config, "Camera with empty model path rejected");
        }

        std::cout << "\nConfigStartup: ALL PASS\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}
