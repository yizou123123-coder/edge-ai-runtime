#include <edge_ai/detector.hpp>
#include <iostream>
#include <stdexcept>
#include <array>
#include <cstdint>
#include <cmath>
#include <chrono>

namespace edge_ai
{


namespace // 匿名命名空间
{
// 辅助函数：设置ORT 中intra_op 线程数量，并返回options
Ort::SessionOptions make_session_options(int intra_op_threads)
{
    if (intra_op_threads < 0)
    {
        throw std::invalid_argument("intra_op_threads must be >= 0");
    }

    Ort::SessionOptions options;

    if (intra_op_threads > 0)
    {
        options.SetIntraOpNumThreads(intra_op_threads);
    }

    return options;
}
} // namespace

OnnxDetector::OnnxDetector(const std::string& model_path, double confidence_threshold, double nms_iou_threshold, int intra_op_threads)
    // 构造函数初始化列表：初始化运行时环境、配置选项并传入模型路径、环境与配置，触发模型解析与加载
    : env_(ORT_LOGGING_LEVEL_WARNING, "edge_detector"),
      session_options_(make_session_options(intra_op_threads)),
      session_(env_, model_path.c_str(), session_options_),
      confidence_threshold_(confidence_threshold),
      nms_iou_threshold_(nms_iou_threshold)
{
    if (!std::isfinite(confidence_threshold_) || confidence_threshold_ < 0 || confidence_threshold_ > 1)
    {
        throw std::invalid_argument("confidence_threshold_ error");
    }
    if (!std::isfinite(nms_iou_threshold_) || nms_iou_threshold_ < 0 || nms_iou_threshold_ > 1)
    {
        throw std::invalid_argument("nms_iou_threshold_ error");
    }

    Ort::AllocatorWithDefaultOptions allocator;

    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
}


std::vector<Detection> OnnxDetector::detect(const cv::Mat& frame, DetectionTiming* timing)
{
    using Clock = std::chrono::steady_clock;
    if (frame.empty())
    {
        throw std::invalid_argument("OnnxDetector received an empty frame");
    }
    const auto pre_start = Clock::now();
    auto preprocess_result = edge_ai::preprocess_yolo(frame, 640, 640);
    const auto pre_end = Clock::now();

    // 四维矩阵 规定了输入的格式
    const std::array<std::int64_t, 4> input_shape{1, 3, 640, 640};
    // 向 ORT 描述 Tensor 所使用的内存位于哪里，以及相关内存属性：默认cpu内存
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // ONNX Runtime 能识别和使用的输入对象：
    // tensor_data：真正的 float 数据  input_shape：按 [1,3,640,640] 维度解释数据  MemoryInfo：描述输入数据所在的内存
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, preprocess_result.tensor_data.data(), preprocess_result.tensor_data.size(),
                                                        input_shape.data(), input_shape.size());

    // 创建 input_names 和 output_names
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};                                                    

    const auto infer_start = Clock::now();
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    const auto infer_end = Clock::now();

    if(outputs.size() != 1)
    {
        throw std::runtime_error("Unexpected output tensor count");
    }
    if (!outputs[0].IsTensor())
    {
        throw std::runtime_error("output Tensor");
    }
    // 获取 outputs[0] 的类型和形状信息
    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo(); 
    // 验证输出的形状
    const auto output_shape = output_info.GetShape();
    const std::vector<std::int64_t> expected_shape{1, 84, 8400};
    if (output_shape != expected_shape) //
    {
        throw std::runtime_error("Output shape mismatch");
    }
    const std::size_t element_count = output_info.GetElementCount(); //
    constexpr std::size_t expected_elements = 705600; 
    if (element_count != expected_elements)
    {
        std::cerr << "Output element count mismatch: " << element_count << '\n';
        throw std::runtime_error("Output element count mismatch");
    }
   
    const float* output_data = outputs[0].GetTensorData<float>();
    if (output_data == nullptr)
    {
        std::cerr << "Failed to get output tensor data pointer!\n";
        throw std::runtime_error("Failed to get output tensor data pointer!");
    }

    const std::size_t num_candidates = static_cast<std::size_t>(output_shape[2]);

    const auto post_start = Clock::now();
    const auto detections = edge_ai::postprocess_yolo(
        output_data,
        num_candidates,
        preprocess_result,
        static_cast<float>(confidence_threshold_),
        static_cast<float>(nms_iou_threshold_)
    );
    const auto post_end = Clock::now();

    const double pre_ms = std::chrono::duration<double, std::milli>(pre_end - pre_start).count();
    const double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
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