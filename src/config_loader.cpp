#include <edge_ai/config_loader.hpp>

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>
#include <cmath>

namespace edge_ai
{

void validate_config(const RuntimeConfig& config)
{
    // 1. Queue Capacity 不能为 0
    if (config.runtime.queue_capacity == 0)
    {
        throw std::invalid_argument("queue_capacity must be greater than 0");
    }

    // 2. FPS 必须是有限的非负数  0表示禁用视频回放节拍
    if (!std::isfinite(config.source.target_fps) || config.source.target_fps < 0.0)
    {
        throw std::invalid_argument("target_fps must be finite and non-negative");
    }

    if (config.runtime.deadline_ms <= 0 || !std::isfinite(config.runtime.deadline_ms))
    {
        throw std::invalid_argument("deadline_ms error");
    }
    
    const double confidence = config.model.confidence_threshold;
    if (!std::isfinite(confidence) ||confidence < 0.0 || confidence > 1.0)
    {
         throw std::invalid_argument("confidence_threshold must be finite and within [0, 1]");
    }

    if (config.model.nms_iou_threshold < 0.0 || config.model.nms_iou_threshold > 1.0)
    {
        throw std::invalid_argument("nms_iou_threshold error");
    }


    // TODO 4：
    // 验证 camera_index
    // 必须大于等于 0
    if (config.source.camera_index < 0)
    {
        throw std::invalid_argument("camera_index error");
    }
    
}


void validate_startup_config(const RuntimeConfig& config)
{
    if (config.source.type == SourceType::Video && config.source.video_path.empty())
    {
        throw std::invalid_argument("Video path error");
    }
    
    if (config.model.path.empty())
    {
        throw std::invalid_argument("model path error");
    }
}


RuntimeConfig load_config(const std::string& path)
{
    // 1. 读取并解析 YAML 文件
    const YAML::Node root = YAML::LoadFile(path);

    // 2. 创建配置对象
    RuntimeConfig config{};

    // 3. 读取输入源类型
    const std::string source_type = root["source"]["type"].as<std::string>();

    if (source_type == "video")
    {
        config.source.type = SourceType::Video;
    }else if(source_type == "camera")
    {
        config.source.type = SourceType::Camera;
    }else
    {
        throw std::invalid_argument("Unknown source type: " + source_type);
    }


    // 4. 读取队列策略
    const std::string policy_name = root["runtime"]["queue_policy"].as<std::string>();

    if (policy_name == "blocking")
    {
        config.runtime.queue_policy = QueuePolicy::Blocking;
    }else if (policy_name == "drop_oldest")
    {
        config.runtime.queue_policy = QueuePolicy::DropOldest;
    }else if (policy_name == "latest_frame")
    {
        config.runtime.queue_policy = QueuePolicy::LatestFrame;
    }else
    {
        throw std::invalid_argument("Unknown queue policy: " + policy_name);
    }

    // 5. SourceConfig 其他字段
    const YAML::Node source_node = root["source"];

    if (source_node["target_fps"])
    {
        config.source.target_fps = source_node["target_fps"].as<double>();
    }

    if (source_node["video_path"])
    {
        config.source.video_path = source_node["video_path"].as<std::string>();
    }
    
    if (source_node["camera_index"])
    {
        config.source.camera_index = source_node["camera_index"].as<int>();
    }
    

    //6. RuntimeOptions 其他字段
    const YAML::Node runtime_node = root["runtime"];

    if (runtime_node["queue_capacity"])
    {
        config.runtime.queue_capacity = runtime_node["queue_capacity"].as<std::size_t>();
    }

    if (runtime_node["deadline_ms"])
    {
        config.runtime.deadline_ms = runtime_node["deadline_ms"].as<double>();
    }

    
    // 7. ModelConfig 字段映射
    // model 整个分组允许省略，使用 RuntimeConfig 默认值
    const YAML::Node model_node = root["model"];

    if (model_node)
    {
        if (model_node["path"])
        {
            config.model.path = model_node["path"].as<std::string>();
        }

        if (model_node["confidence_threshold"])
        {
            config.model.confidence_threshold = model_node["confidence_threshold"].as<double>();
        }
        
        if (model_node["nms_iou_threshold"])
        {
            config.model.nms_iou_threshold = model_node["nms_iou_threshold"].as<double>();
        }
    }

    // 9. DisplayConfig 字段映射
    const YAML::Node display_node = root["display"];

    if (display_node)
    {
        if (display_node["enabled"])
        {
            config.display.enabled = display_node["enabled"].as<bool>();
        }
    }

    validate_config(config);
    // 10. 返回配置
    return config;


}

}// namespace edge_ai
