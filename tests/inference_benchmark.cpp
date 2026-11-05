
#include <edge_ai/preprocess.hpp>
#include <edge_ai/postprocess.hpp>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

constexpr int kWarmup = 10;
constexpr int kIterations = 100;

constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;

constexpr float kConfidenceThreshold = 0.25F;
constexpr float kNmsThreshold = 0.45F;

// 一次完整处理产生的性能数据
struct TimingSample
{
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;

    std::size_t detection_count = 0;
};

// 统计结果
struct TimingStats
{
    double mean_ms = 0.0;
    double p95_ms = 0.0;
};

// 计算 Mean 和 P95
TimingStats calculate_stats(std::vector<double> times_ms)
{
    if (times_ms.empty())
    {
        throw std::invalid_argument("Timing samples are empty");
    }

    for (double time : times_ms)
    {
        if (!std::isfinite(time) || time < 0.0)
        {
            throw std::runtime_error("Invalid timing sample");
        }
    }

    // 1. 计算平均耗时
    const double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);

    const double mean = sum / static_cast<double>(times_ms.size());

    // 2. 排序
    std::sort(times_ms.begin(), times_ms.end());

    // 3. 最近秩方法计算 P95
    // index = ceil(0.95 * N) - 1
    const std::size_t p95_index =
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(times_ms.size()))) - 1;

    return TimingStats{mean, times_ms[p95_index]};
}

// 执行一次完整的单帧推理流程
TimingSample run_one_frame(Ort::Session &session, const cv::Mat &frame,
                           const std::string &input_name, const std::string &output_name,
                           const Ort::MemoryInfo &memory_info)
{
    // ========================================
    // Stage 1: Preprocess
    // ========================================

    const auto t0 = Clock::now();

    auto preprocess_result = edge_ai::preprocess_yolo(frame, kInputWidth, kInputHeight);

    const auto t1 = Clock::now();

    if (preprocess_result.tensor_data.size() !=
        static_cast<std::size_t>(3 * kInputWidth * kInputHeight))
    {
        throw std::runtime_error("Invalid preprocess tensor size");
    }

    // ========================================
    // Stage 2: Inference
    // ========================================

    constexpr std::array<std::int64_t, 4> input_shape{1, 3, kInputHeight, kInputWidth};

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, preprocess_result.tensor_data.data(), preprocess_result.tensor_data.size(),
        input_shape.data(), input_shape.size());

    const char *input_names[] = {input_name.c_str()};

    const char *output_names[] = {output_name.c_str()};

    auto outputs =
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    const auto t2 = Clock::now();

    // ========================================
    // Stage 3: Postprocess
    // ========================================

    // 验证 ORT 输出
    if (outputs.size() != 1 || !outputs[0].IsTensor())
    {
        throw std::runtime_error("Invalid ONNX output");
    }

    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();

    const auto output_shape = output_info.GetShape();

    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] != 84 ||
        output_shape[2] != 8400)
    {
        throw std::runtime_error("Unexpected output tensor shape");
    }

    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        throw std::runtime_error("Unexpected output tensor dtype");
    }

    constexpr std::size_t expected_elements = 84 * 8400;

    if (output_info.GetElementCount() != expected_elements)
    {
        throw std::runtime_error("Unexpected output element count");
    }

    const float *output_data = outputs[0].GetTensorData<float>();

    if (output_data == nullptr)
    {
        throw std::runtime_error("Null output tensor pointer");
    }

    const std::size_t num_candidates = static_cast<std::size_t>(output_shape[2]);

    const auto detections = edge_ai::postprocess_yolo(
        output_data, num_candidates, preprocess_result, kConfidenceThreshold, kNmsThreshold);

    const auto t3 = Clock::now();

    // ========================================
    // 计算各阶段耗时
    // ========================================

    TimingSample sample;

    sample.preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    sample.inference_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    sample.postprocess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    sample.detection_count = detections.size();

    return sample;
}

// 打印某个阶段的性能统计
void print_stats(const std::string &name, const TimingStats &stats)
{
    std::cout << std::left << std::setw(16) << name << std::right << std::setw(14) << stats.mean_ms
              << std::setw(14) << stats.p95_ms << '\n';
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: inference_benchmark <video_path>\n";

        return 1;
    }

    try
    {
        // ========================================
        // 1. 读取固定视频的第一帧
        // ========================================

        const std::string video_path = argv[1];
        const std::string model_path = "models/yolov8n.onnx";

        cv::VideoCapture capture(video_path);

        if (!capture.isOpened())
        {
            throw std::runtime_error("Failed to open video: " + video_path);
        }

        cv::Mat frame;

        if (!capture.read(frame) || frame.empty())
        {
            throw std::runtime_error("Failed to read first video frame");
        }

        capture.release();

        // ========================================
        // 2. 初始化 ONNX Runtime（只执行一次）
        // ========================================

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge_inference_benchmark");

        Ort::SessionOptions session_options;

        // 使用默认 CPU Execution Provider
        // 不显式修改 ORT 内部线程数
        Ort::Session session(env, model_path.c_str(), session_options);

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::AllocatorWithDefaultOptions allocator;

        // 将 ORT 返回的节点名称复制到 std::string
        std::string input_name = session.GetInputNameAllocated(0, allocator).get();

        std::string output_name = session.GetOutputNameAllocated(0, allocator).get();

        // ========================================
        // 3. 预热
        // ========================================

        std::size_t expected_detection_count = 0;

        for (int i = 0; i < kWarmup; ++i)
        {
            const auto sample = run_one_frame(session, frame, input_name, output_name, memory_info);

            if (i == 0)
            {
                expected_detection_count = sample.detection_count;
            }
            else if (sample.detection_count != expected_detection_count)
            {
                throw std::runtime_error("Inconsistent detection count during warmup");
            }
        }

        // ========================================
        // 4. 正式测量
        // ========================================

        std::vector<TimingSample> samples;
        samples.reserve(kIterations);

        for (int i = 0; i < kIterations; ++i)
        {
            const auto sample = run_one_frame(session, frame, input_name, output_name, memory_info);

            if (sample.detection_count != expected_detection_count)
            {
                throw std::runtime_error("Inconsistent detection count during benchmark");
            }

            samples.push_back(sample);
        }

        // ========================================
        // 5. 提取三个阶段的耗时数据
        // ========================================

        std::vector<double> preprocess_times;
        std::vector<double> inference_times;
        std::vector<double> postprocess_times;

        preprocess_times.reserve(samples.size());
        inference_times.reserve(samples.size());
        postprocess_times.reserve(samples.size());

        for (const auto &sample : samples)
        {
            preprocess_times.push_back(sample.preprocess_ms);
            inference_times.push_back(sample.inference_ms);
            postprocess_times.push_back(sample.postprocess_ms);
        }

        // ========================================
        // 6. 计算统计数据
        // ========================================

        const auto preprocess_stats = calculate_stats(preprocess_times);

        const auto inference_stats = calculate_stats(inference_times);

        const auto postprocess_stats = calculate_stats(postprocess_times);

        // ========================================
        // 7. 打印性能报告
        // ========================================

        std::cout << "\n========== P3-07 Release Baseline ==========\n";

        std::cout << "Model       : " << model_path << '\n';
        std::cout << "Video       : " << video_path << '\n';

        std::cout << "Frame size  : " << frame.cols << " x " << frame.rows << '\n';

#ifdef NDEBUG
        std::cout << "Build       : Release (NDEBUG)\n";
#else
        std::cout << "Build       : Debug\n";
#endif

        std::cout << "ORT version : " << Ort::GetVersionString() << '\n';

        std::cout << "ORT threads : default (not explicitly set)\n";
        std::cout << "Warmup      : " << kWarmup << '\n';
        std::cout << "Iterations  : " << kIterations << '\n';

        std::cout << "Detections  : " << expected_detection_count << '\n';

        std::cout << "\n";

        std::cout << std::left << std::setw(16) << "Stage" << std::right << std::setw(14)
                  << "Mean(ms)" << std::setw(14) << "P95(ms)" << '\n';

        std::cout << std::fixed << std::setprecision(3);

        print_stats("Preprocess", preprocess_stats);
        print_stats("Inference", inference_stats);
        print_stats("Postprocess", postprocess_stats);

        std::cout << "============================================\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Benchmark Error] " << e.what() << '\n';

        return 1;
    }
}
