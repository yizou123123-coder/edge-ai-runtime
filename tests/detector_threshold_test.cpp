
#include <edge_ai/detector.hpp>
#include <edge_ai/preprocess.hpp>
#include <edge_ai/postprocess.hpp>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void require(bool ok, const std::string& message)
{
    if (!ok)
    {
        throw std::runtime_error(message);
    }
}

// 测试 1：直接验证 Confidence 和 NMS 算法
void test_postprocess_thresholds()
{
    constexpr std::size_t N = 8400;
    constexpr std::size_t C = 84;

    // YOLOv8n: [1, 84, 8400]
    std::vector<float> output(C * N, 0.0F);

    auto set_candidate = [&](std::size_t i,
                             float cx,
                             float cy,
                             float w,
                             float h,
                             float score)
    {
        output[0 * N + i] = cx;
        output[1 * N + i] = cy;
        output[2 * N + i] = w;
        output[3 * N + i] = h;

        // 类别 0
        output[4 * N + i] = score;
    };

    // 两个高度重叠的同类检测框
    set_candidate(0, 320, 320, 200, 200, 0.90F);
    set_candidate(1, 322, 320, 200, 200, 0.80F);

    cv::Mat image(
        640, 640, CV_8UC3,
        cv::Scalar(0, 0, 0)
    );

    const auto prep =
        edge_ai::preprocess_yolo(image, 640, 640);

    // 置信度 0.25，严格 NMS
    const auto strict_nms =
        edge_ai::postprocess_yolo(
            output.data(), N, prep, 0.25F, 0.30F
        );

    // 置信度 0.25，宽松 NMS
    const auto loose_nms =
        edge_ai::postprocess_yolo(
            output.data(), N, prep, 0.25F, 1.0F
        );

    // 高置信度阈值，全部过滤
    const auto high_confidence =
        edge_ai::postprocess_yolo(
            output.data(), N, prep, 0.95F, 1.0F
        );

    require(
        strict_nms.size() == 1,
        "Strict NMS should retain 1 detection"
    );

    require(
        loose_nms.size() == 2,
        "Loose NMS should retain 2 detections"
    );

    require(
        high_confidence.empty(),
        "High confidence should reject both boxes"
    );

    std::cout
        << "[PASS] Confidence filtering\n"
        << "[PASS] NMS IoU threshold\n";
}

// 测试 2：确认 Detector 实际使用传入的阈值
void test_detector_thresholds(
    const std::string& video_path)
{
    cv::VideoCapture video(video_path);

    require(video.isOpened(), "Cannot open test video");

    cv::Mat frame;
    require(
        video.read(frame) && !frame.empty(),
        "Cannot read first video frame"
    );

    const std::string model_path =
        "models/yolov8n.onnx";

    edge_ai::OnnxDetector normal_detector(
        model_path, 0.25, 0.45
    );

    edge_ai::OnnxDetector high_detector(
        model_path, 1.0, 0.45
    );

    const auto normal =
        normal_detector.detect(frame);

    const auto high =
        high_detector.detect(frame);

    // 针对当前已验证过的 test.mp4 第一帧
    require(
        !normal.empty(),
        "Expected a detection at confidence 0.25"
    );

    require(
        high.empty(),
        "Confidence 1.0 should reject detections"
    );

    std::cout
        << "[PASS] Detector uses confidence parameter\n";
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr
            << "Usage: detector_threshold_test <video_path>\n";
        return 1;
    }

    try
    {
        test_postprocess_thresholds();
        test_detector_thresholds(argv[1]);

        std::cout
            << "\nDetector Threshold: ALL PASS\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
