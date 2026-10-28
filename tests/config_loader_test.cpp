
#include <edge_ai/config_loader.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

class TempYaml
{
public:
    TempYaml()
    {
        const auto id = std::chrono::steady_clock::now()
                            .time_since_epoch().count();

        path_ = fs::temp_directory_path() /
            ("edge_ai_config_test_" +
             std::to_string(id) + ".yaml");
    }

    ~TempYaml()
    {
        std::error_code ec;
        fs::remove(path_, ec);
    }

    TempYaml(const TempYaml&) = delete;
    TempYaml& operator=(const TempYaml&) = delete;

    void write(const std::string& content)
    {
        std::ofstream out(path_, std::ios::trunc);

        if (!out)
            throw std::runtime_error("Cannot create test YAML");

        out << content;
        out.flush();

        if (!out)
            throw std::runtime_error("Cannot write test YAML");
    }

    std::string path() const
    {
        return path_.string();
    }

private:
    fs::path path_;
};

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

bool approx(double a, double b)
{
    return std::fabs(a - b) < 1e-9;
}

// Test 1~3: 合法枚举
void test_valid_enum(
    TempYaml& file,
    const std::string& source_name,
    const std::string& policy_name,
    edge_ai::SourceType expected_source,
    edge_ai::QueuePolicy expected_policy)
{
    file.write(
        "source:\n"
        "  type: " + source_name + "\n"
        "runtime:\n"
        "  queue_policy: " + policy_name + "\n"
    );

    const auto config = edge_ai::load_config(file.path());

    require(config.source.type == expected_source,
            "SourceType mismatch");

    require(config.runtime.queue_policy == expected_policy,
            "QueuePolicy mismatch");

    std::cout << "[PASS] " << source_name
              << " + " << policy_name << '\n';
}

// Test 4~5: 非法枚举
void test_invalid_enum(
    TempYaml& file,
    const std::string& content)
{
    file.write(content);

    bool rejected = false;

    try
    {
        (void)edge_ai::load_config(file.path());
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(rejected, "Invalid enum was accepted");
    std::cout << "[PASS] Invalid enum rejected\n";
}

// Test 6: Source / Runtime 字段覆盖
void test_field_overrides(TempYaml& file)
{
    file.write(R"(
source:
  type: camera
  video_path: custom/test.mp4
  camera_index: 2
  target_fps: 25.0

runtime:
  queue_policy: drop_oldest
  queue_capacity: 8
  deadline_ms: 75.5
)");

    const auto c = edge_ai::load_config(file.path());

    require(c.source.type == edge_ai::SourceType::Camera,
            "Source type override failed");
    require(c.source.video_path == "custom/test.mp4",
            "Video path override failed");
    require(c.source.camera_index == 2,
            "Camera index override failed");
    require(approx(c.source.target_fps, 25.0),
            "Target FPS override failed");
    require(c.runtime.queue_policy ==
                edge_ai::QueuePolicy::DropOldest,
            "Queue policy override failed");
    require(c.runtime.queue_capacity == 8,
            "Queue capacity override failed");
    require(approx(c.runtime.deadline_ms, 75.5),
            "Deadline override failed");

    std::cout << "[PASS] Source/Runtime field overrides\n";
}

// Test 7: Source / Runtime 默认值
void test_default_values(TempYaml& file)
{
    file.write(R"(
source:
  type: video

runtime:
  queue_policy: blocking
)");

    const auto c = edge_ai::load_config(file.path());

    require(c.source.type == edge_ai::SourceType::Video,
            "Default source type mismatch");
    require(c.source.video_path.empty(),
            "Default video path mismatch");
    require(c.source.camera_index == 0,
            "Default camera index mismatch");
    require(approx(c.source.target_fps, 30.0),
            "Default FPS mismatch");
    require(c.runtime.queue_policy ==
                edge_ai::QueuePolicy::Blocking,
            "Queue policy mismatch");
    require(c.runtime.queue_capacity == 4,
            "Default capacity mismatch");
    require(approx(c.runtime.deadline_ms, 100.0),
            "Default deadline mismatch");

    std::cout << "[PASS] Source/Runtime defaults\n";
}

// Test 8: 数字类型错误
void test_invalid_numeric_type(TempYaml& file)
{
    file.write(R"(
source:
  type: video
  target_fps: invalid_fps

runtime:
  queue_policy: latest_frame
)");

    bool rejected = false;

    try
    {
        (void)edge_ai::load_config(file.path());
    }
    catch (const YAML::BadConversion&)
    {
        rejected = true;
    }

    require(rejected, "Invalid numeric type was accepted");
    std::cout << "[PASS] Invalid numeric type rejected\n";
}

// Test 9: Model / Display 字段覆盖
void test_model_display_overrides(TempYaml& file)
{
    file.write(R"(
source:
  type: video

runtime:
  queue_policy: latest_frame

model:
  path: models/custom.onnx
  confidence_threshold: 0.65
  nms_iou_threshold: 0.30

display:
  enabled: true
)");

    const auto c = edge_ai::load_config(file.path());

    require(c.model.path == "models/custom.onnx",
            "Model path override failed");
    require(approx(c.model.confidence_threshold, 0.65),
            "Confidence override failed");
    require(approx(c.model.nms_iou_threshold, 0.30),
            "NMS IoU override failed");
    require(c.display.enabled,
            "Display enabled override failed");

    std::cout << "[PASS] Model/Display field overrides\n";
}

// Test 10: 完全省略 Model / Display
void test_model_display_defaults(TempYaml& file)
{
    file.write(R"(
source:
  type: video

runtime:
  queue_policy: blocking
)");

    const auto c = edge_ai::load_config(file.path());

    require(c.model.path == "models/yolov8n.onnx",
            "Default model path mismatch");
    require(approx(c.model.confidence_threshold, 0.25),
            "Default confidence mismatch");
    require(approx(c.model.nms_iou_threshold, 0.45),
            "Default NMS mismatch");
    require(!c.display.enabled,
            "Default display must be disabled");

    std::cout << "[PASS] Model/Display defaults\n";
}

// Test 11: Display 分组存在，但 enabled 缺失
void test_empty_display_group(TempYaml& file)
{
    file.write(R"(
source:
  type: video

runtime:
  queue_policy: latest_frame

display: {}
)");

    const auto c = edge_ai::load_config(file.path());

    require(!c.display.enabled,
            "Missing display.enabled should keep default");

    std::cout << "[PASS] Empty display group defaults\n";
}

// Test 12: Display 布尔类型错误
void test_invalid_display_bool(TempYaml& file)
{
    file.write(R"(
source:
  type: video

runtime:
  queue_policy: latest_frame

display:
  enabled: not_a_bool
)");

    bool rejected = false;

    try
    {
        (void)edge_ai::load_config(file.path());
    }
    catch (const YAML::BadConversion&)
    {
        rejected = true;
    }

    require(rejected, "Invalid display bool was accepted");

    std::cout << "[PASS] Invalid display bool rejected\n";
}

int main()
{
    try
    {
        TempYaml file;

        test_valid_enum(
            file, "video", "blocking",
            edge_ai::SourceType::Video,
            edge_ai::QueuePolicy::Blocking
        );

        test_valid_enum(
            file, "camera", "drop_oldest",
            edge_ai::SourceType::Camera,
            edge_ai::QueuePolicy::DropOldest
        );

        test_valid_enum(
            file, "video", "latest_frame",
            edge_ai::SourceType::Video,
            edge_ai::QueuePolicy::LatestFrame
        );

        test_invalid_enum(
            file,
            "source:\n"
            "  type: microphone\n"
            "runtime:\n"
            "  queue_policy: latest_frame\n"
        );

        test_invalid_enum(
            file,
            "source:\n"
            "  type: video\n"
            "runtime:\n"
            "  queue_policy: super_fast\n"
        );

        test_field_overrides(file);
        test_default_values(file);
        test_invalid_numeric_type(file);

        test_model_display_overrides(file);
        test_model_display_defaults(file);
        test_empty_display_group(file);
        test_invalid_display_bool(file);

        std::cout << "\nConfigLoader: ALL PASS\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
