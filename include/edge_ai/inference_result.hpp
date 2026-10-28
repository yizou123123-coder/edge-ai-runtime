#pragma once

#include <edge_ai/frame.hpp>
#include <edge_ai/postprocess.hpp>

#include <vector>

namespace edge_ai
{

struct InferenceResult
{
    Frame frame;

    std::vector<Detection> detections;
};

} // namespace edge_ai
