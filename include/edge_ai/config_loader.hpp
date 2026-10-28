#pragma once

#include <edge_ai/runtime_config.hpp>

#include <string>

namespace edge_ai
{

// 从 YAML 文件加载/返回 RuntimeConfig这个结构体
RuntimeConfig load_config(const std::string& path);

// 校验 C++ 配置对象：数值是否合法
void validate_config(const RuntimeConfig& config);

// 校验 C++ 配置对象：是否具备启动必需参数
void validate_startup_config(const RuntimeConfig& config);
} // namespace edge_ai
