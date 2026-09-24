#pragma once

#include <edge_ai/video_source.hpp>
#include <string>

namespace edge_ai
{

class Runtime
{
public:
    explicit Runtime(const std::string& video_path);

    int run();

private:
    VideoSource source_;
};

}  // namespace edge_ai