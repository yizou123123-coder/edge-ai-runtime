#include <edge_ai/frame_source_factory.hpp>

#include <edge_ai/video_source.hpp>
#include <edge_ai/camera_source.hpp>

#include <memory>
#include <stdexcept>

namespace edge_ai
{

std::unique_ptr<FrameSource> create_frame_source(const SourceConfig &config)
{
    switch (config.type)
    {
    case SourceType::Video:
    {
        return std::make_unique<VideoSource>(config.video_path, config.target_fps);
    }

    case SourceType::Camera:
    {
        return std::make_unique<CameraSource>(config.camera_index);
    }
    }

    throw std::invalid_argument("SourceType error");
}

} // namespace edge_ai
