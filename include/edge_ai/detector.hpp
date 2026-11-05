#pragma once

#include <edge_ai/postprocess.hpp>
#include <edge_ai/preprocess.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace edge_ai
{

struct DetectionTiming
{
    double preprocess_ms{0.0};
    double inference_ms{0.0};
    double postprocess_ms{0.0};
};

class OnnxDetector
{
  public:
    explicit OnnxDetector(const std::string &model_path, double confidence_threshold = 0.25,
                          double nms_iou_threshold = 0.45, int intra_op_threads = 0);

    std::vector<Detection> detect(const cv::Mat &frame, DetectionTiming *timing = nullptr);

  private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;

    std::string input_name_;
    std::string output_name_;

    double confidence_threshold_{0.25};
    double nms_iou_threshold_{0.45};
};

} // namespace edge_ai