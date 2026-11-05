
#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: yaml_smoke_test <config_path>\n";
        return 1;
    }

    try
    {
        // 1. 从文件读取 YAML
        const YAML::Node root = YAML::LoadFile(argv[1]);

        // 2. 读取 source 配置
        const std::string source_type = root["source"]["type"].as<std::string>();

        const double target_fps = root["source"]["target_fps"].as<double>();

        // 3. 读取 runtime 配置
        const std::size_t queue_capacity = root["runtime"]["queue_capacity"].as<std::size_t>();

        const std::string queue_policy = root["runtime"]["queue_policy"].as<std::string>();

        // 4. 读取 model 和 display 配置
        const std::string model_path = root["model"]["path"].as<std::string>();

        const bool display_enabled = root["display"]["enabled"].as<bool>();

        std::cout << "========== YAML Config ==========\n";
        std::cout << "Source Type    : " << source_type << '\n';
        std::cout << "Target FPS     : " << target_fps << '\n';
        std::cout << "Queue Capacity : " << queue_capacity << '\n';
        std::cout << "Queue Policy   : " << queue_policy << '\n';
        std::cout << "Model Path     : " << model_path << '\n';
        std::cout << "Display        : " << std::boolalpha << display_enabled << '\n';

        // 5. 验证本次实验配置
        if (source_type != "video" || target_fps != 30.0 || queue_capacity != 4 ||
            queue_policy != "latest_frame" || model_path != "models/yolov8n.onnx" ||
            display_enabled)
        {
            throw std::runtime_error("Unexpected YAML configuration values");
        }

        std::cout << "YAML Smoke Test: ALL PASS\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
