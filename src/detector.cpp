#include <edge_ai/detector.hpp>
#include <edge_ai/ort_cpu_engine.hpp>
#include <edge_ai/preprocess.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace edge_ai
{

OnnxDetector::OnnxDetector(const std::string &model_path, double confidence_threshold,
                           double nms_iou_threshold, int intra_op_threads)
    : engine_(std::make_unique<OrtCpuEngine>(model_path, intra_op_threads)),
      confidence_threshold_(confidence_threshold), nms_iou_threshold_(nms_iou_threshold)
{
    if (!std::isfinite(confidence_threshold_) || confidence_threshold_ < 0 ||
        confidence_threshold_ > 1)
    {
        throw std::invalid_argument("confidence_threshold_ error");
    }
    if (!std::isfinite(nms_iou_threshold_) || nms_iou_threshold_ < 0 || nms_iou_threshold_ > 1)
    {
        throw std::invalid_argument("nms_iou_threshold_ error");
    }
}

std::vector<Detection> OnnxDetector::detect(const cv::Mat &frame, DetectionTiming *timing)
{
    using Clock = std::chrono::steady_clock;
    if (frame.empty())
    {
        throw std::invalid_argument("OnnxDetector received an empty frame");
    }

    const auto pre_start = Clock::now();
    auto preprocess_result = edge_ai::preprocess_yolo(frame, 640, 640);
    const auto pre_end = Clock::now();

    const auto infer_start = Clock::now();
    std::vector<float> output = engine_->infer(preprocess_result.tensor_data);
    const auto infer_end = Clock::now();

    const std::size_t num_candidates = 8400;
    const auto post_start = Clock::now();
    const auto detections = edge_ai::postprocess_yolo(
        output.data(), num_candidates, preprocess_result, static_cast<float>(confidence_threshold_),
        static_cast<float>(nms_iou_threshold_));
    const auto post_end = Clock::now();

    const double pre_ms = std::chrono::duration<double, std::milli>(pre_end - pre_start).count();
    const double infer_ms =
        std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    const double post_ms = std::chrono::duration<double, std::milli>(post_end - post_start).count();

    if (timing != nullptr)
    {
        timing->preprocess_ms = pre_ms;
        timing->inference_ms = infer_ms;
        timing->postprocess_ms = post_ms;
    }

    return detections;
}
} // namespace edge_ai