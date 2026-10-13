#pragma once

#include <edge_ai/preprocess.hpp>
#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace edge_ai
{

struct Detection
{
    int class_id{-1};
    float confidence{0.0F};
    cv::Rect2f box;
};

std::vector<Detection> postprocess_yolo(
    const float* output_data,
    std::size_t num_candidates,
    const PreprocessResult& preprocess,
    float confidence_threshold,
    float nms_iou_threshold
);

} // namespace edge_ai