#pragma once

#include <edge_ai/frame_source.hpp>
#include <edge_ai/runtime_config.hpp>

#include <memory>

namespace edge_ai
{
// 根据类型创建具体对象
std::unique_ptr<FrameSource> create_frame_source(const SourceConfig &config);

} // namespace edge_ai