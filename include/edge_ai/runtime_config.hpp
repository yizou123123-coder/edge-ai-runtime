#pragma once

#include <edge_ai/blocking_queue.hpp>

#include <cstddef>
#include <string>

namespace edge_ai
{

// 输入源类型
enum class SourceType
{
    Video,
    Camera
};


// 1. 输入源配置
struct SourceConfig
{
    SourceType type{SourceType::Video};
    std::string video_path{};
    int camera_index{0};
    double target_fps{30.0};
};


// 2. Runtime 运行参数
struct RuntimeOptions
{
    std::size_t queue_capacity{4};
    QueuePolicy queue_policy{QueuePolicy::LatestFrame};
    double deadline_ms{100.0};
};


// 3. 模型配置
struct ModelConfig
{
    std::string path{"models/yolov8n.onnx"};
    double confidence_threshold{0.25};
    double nms_iou_threshold{0.45};
};


// 4. 显示配置
struct DisplayConfig
{
    bool enabled{false};
};

// 5. ORT内部线程数量
struct InferenceConfig
{
    int intra_op_threads{0};
};


// 5. 总配置
struct RuntimeConfig
{
    SourceConfig source{};
    RuntimeOptions runtime{};
    ModelConfig model{};
    DisplayConfig display{};
    InferenceConfig intra_op{};
};

} // namespace edge_ai
