#include <edge_ai/config_loader.hpp>
#include <edge_ai/runtime_config.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

using Config = edge_ai::RuntimeConfig;

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

struct InvalidCase
{
    std::string name;
    std::function<void(Config&)> modify;
};

// 验证 validate_config 拒绝非法配置
void test_invalid_ranges()
{
    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    const double inf =
        std::numeric_limits<double>::infinity();

    const std::vector<InvalidCase> cases = {
        {"queue_capacity = 0",
         [](Config& c) { c.runtime.queue_capacity = 0; }},

        {"target_fps = -30",
         [](Config& c) { c.source.target_fps = -30.0; }},

        {"target_fps = NaN",
         [nan](Config& c) { c.source.target_fps = nan; }},

        {"target_fps = Infinity",
         [inf](Config& c) { c.source.target_fps = inf; }},

        {"deadline_ms = 0",
         [](Config& c) { c.runtime.deadline_ms = 0.0; }},

        {"deadline_ms = -100",
         [](Config& c) { c.runtime.deadline_ms = -100.0; }},

        {"deadline_ms = NaN",
         [nan](Config& c) { c.runtime.deadline_ms = nan; }},

        {"deadline_ms = Infinity",
         [inf](Config& c) { c.runtime.deadline_ms = inf; }},

        {"confidence = -0.1",
         [](Config& c) {
             c.model.confidence_threshold = -0.1;
         }},

        {"confidence = 1.5",
         [](Config& c) {
             c.model.confidence_threshold = 1.5;
         }},

        {"confidence = NaN",
         [nan](Config& c) {
             c.model.confidence_threshold = nan;
         }},

        {"nms_iou = -0.2",
         [](Config& c) {
             c.model.nms_iou_threshold = -0.2;
         }},

        {"nms_iou = 1.1",
         [](Config& c) {
             c.model.nms_iou_threshold = 1.1;
         }},

        {"nms_iou = Infinity",
         [inf](Config& c) {
             c.model.nms_iou_threshold = inf;
         }},

        {"camera_index = -1",
         [](Config& c) { c.source.camera_index = -1; }}
    };

    for (const auto& test : cases)
    {
        Config config{};
        test.modify(config);

        bool rejected = false;

        try
        {
            edge_ai::validate_config(config);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }

        require(rejected, "Accepted invalid: " + test.name);

        std::cout
            << "[PASS] Rejected " << test.name << '\n';
    }
}

// 验证合法边界值能够通过
void test_valid_boundaries()
{
    Config config{};

    // 0 表示不启用视频节拍
    config.source.target_fps = 0.0;

    // 队列容量最小为 1
    config.runtime.queue_capacity = 1;

    // 正数时间预算
    config.runtime.deadline_ms = 0.1;

    // 两个阈值允许位于 [0, 1]
    config.model.confidence_threshold = 0.0;
    config.model.nms_iou_threshold = 1.0;

    config.source.camera_index = 0;

    edge_ai::validate_config(config);

    // 测试另一个边界
    config.model.confidence_threshold = 1.0;
    config.model.nms_iou_threshold = 0.0;

    edge_ai::validate_config(config);

    std::cout << "[PASS] Valid boundary values\n";
}

// 验证 load_config 真正调用了 validate_config
void test_loader_validation()
{
    const auto id =
        std::chrono::steady_clock::now()
            .time_since_epoch().count();

    const fs::path path =
        fs::temp_directory_path() /
        ("edge_ai_invalid_config_" +
         std::to_string(id) + ".yaml");

    // RAII：测试结束时删除临时文件
    struct Cleanup
    {
        fs::path path;

        ~Cleanup()
        {
            std::error_code ec;
            fs::remove(path, ec);
        }
    } cleanup{path};

    {
        std::ofstream out(path);

        require(
            static_cast<bool>(out),
            "Failed to create YAML test file"
        );

        out << R"(
source:
  type: video

runtime:
  queue_policy: latest_frame
  queue_capacity: 0
)";

        out.flush();

        require(
            static_cast<bool>(out),
            "Failed to write YAML test file"
        );
    }

    bool rejected = false;

    try
    {
        (void)edge_ai::load_config(path.string());
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(
        rejected,
        "load_config did not validate queue_capacity"
    );

    std::cout << "[PASS] Loader invokes validation\n";
}

int main()
{
    try
    {
        // 1. 非法数值范围
        test_invalid_ranges();

        // 2. 合法边界值
        test_valid_boundaries();

        // 3. YAML 加载与校验集成
        test_loader_validation();

        std::cout
            << "\nConfigValidation: ALL PASS\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}
