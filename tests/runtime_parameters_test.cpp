
#include <edge_ai/frame_source.hpp>
#include <edge_ai/runtime.hpp>

#include <opencv2/core.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

// 模拟一个能够产生两帧图像的视频源
class TwoFrameSource final : public edge_ai::FrameSource
{
public:
    bool is_open() const override
    {
        return true;
    }

    bool read(cv::Mat& frame) override
    {
        if (emitted_ >= 2)
        {
            frame.release();
            return false;
        }

        frame = cv::Mat(
            640, 640,
            CV_8UC3,
            cv::Scalar(0, 0, 0)
        );

        ++emitted_;
        return true;
    }

private:
    int emitted_{0};
};

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

// 使用指定模型路径和 Deadline 运行一次
int run_runtime(
    const std::string& model_path,
    double deadline_ms)
{
    edge_ai::Runtime runtime(
        std::make_unique<TwoFrameSource>(),
        edge_ai::QueuePolicy::Blocking,
        4,
        model_path,
        deadline_ms
    );

    require(
        runtime.start(),
        "Runtime failed to start"
    );

    runtime.join();
    return runtime.exit_code();
}

// 验证正常模型能完成推理
void test_valid_model()
{
    const int code = run_runtime(
        "models/yolov8n.onnx",
        100.0
    );

    require(code == 0, "Valid model failed");

    std::cout << "[PASS] Valid model path\n";
}

// 验证指定的不存在模型确实被尝试加载
void test_missing_model()
{
    const int code = run_runtime(
        "models/__intentionally_missing__.onnx",
        100.0
    );

    require(
        code == 6,
        "Missing model did not produce error code 6"
    );

    std::cout << "[PASS] Missing model rejected\n";
}

// 验证空模型路径被构造函数拒绝
void test_empty_model()
{
    bool rejected = false;

    try
    {
        edge_ai::Runtime runtime(
            std::make_unique<TwoFrameSource>(),
            edge_ai::QueuePolicy::Blocking,
            4,
            "",
            100.0
        );
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(rejected, "Empty model path accepted");
    std::cout << "[PASS] Empty model path rejected\n";
}

// 验证非法 Deadline
void test_invalid_deadline(double value)
{
    bool rejected = false;

    try
    {
        edge_ai::Runtime runtime(
            std::make_unique<TwoFrameSource>(),
            edge_ai::QueuePolicy::Blocking,
            4,
            "models/yolov8n.onnx",
            value
        );
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    require(rejected, "Invalid deadline accepted");
}

// 验证合法 Deadline 可运行
void test_valid_deadlines()
{
    require(
        run_runtime("models/yolov8n.onnx", 0.001) == 0,
        "Small positive deadline failed"
    );

    require(
        run_runtime("models/yolov8n.onnx", 1000000.0) == 0,
        "Large positive deadline failed"
    );

    std::cout << "[PASS] Valid deadline parameters\n";
}

int main()
{
    try
    {
        test_valid_model();
        test_missing_model();
        test_empty_model();

        test_invalid_deadline(0.0);
        test_invalid_deadline(-1.0);

        test_invalid_deadline(
            std::numeric_limits<double>::quiet_NaN()
        );

        std::cout
            << "[PASS] Invalid deadlines rejected\n";

        test_valid_deadlines();

        std::cout
            << "\nRuntime Parameters: ALL PASS\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
