#include <edge_ai/preprocess.hpp>
#include <edge_ai/postprocess.hpp>

#include <opencv2/core.hpp>

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr int kTargetWidth = 640;
constexpr int kTargetHeight = 640;

constexpr int kClassId = 2;
constexpr float kConfidence = 0.9F;
constexpr float kTolerance = 1e-3F;

// 每个测试用例的数据
struct TestCase
{
    std::string name;

    // 原始图片尺寸
    int original_width;
    int original_height;

    // 期望的预处理元数据
    float expected_scale;
    int expected_pad_x;
    int expected_pad_y;

    // 人工构造的 YOLO 输出框
    // 格式：cx, cy, width, height
    float model_cx;
    float model_cy;
    float model_width;
    float model_height;

    // 期望恢复到原图上的检测框
    // 格式：x, y, width, height
    cv::Rect2f expected_box;
};

// 验证两个 float 是否足够接近
void expect_near(const std::string &description, float actual, float expected)
{
    if (!std::isfinite(actual) || std::fabs(actual - expected) > kTolerance)
    {
        std::ostringstream oss;

        oss << description << " mismatch: expected=" << expected << ", actual=" << actual;

        throw std::runtime_error(oss.str());
    }
}

// 执行一个测试用例
void run_test_case(const TestCase &tc)
{
    // 1. 构造原始图像
    // 不需要真实图片，因为这里只测试几何坐标变换
    cv::Mat image(tc.original_height, tc.original_width, CV_8UC3, cv::Scalar(0, 0, 0));

    // 2. 执行真实 C++ 预处理
    // 得到 scale、pad_x、pad_y
    const auto prep = edge_ai::preprocess_yolo(image, kTargetWidth, kTargetHeight);

    // 3. 验证预处理的坐标变换参数
    expect_near(tc.name + ": scale", prep.scale, tc.expected_scale);

    if (prep.pad_x != tc.expected_pad_x || prep.pad_y != tc.expected_pad_y)
    {
        std::ostringstream oss;

        oss << tc.name << ": padding mismatch" << ", expected=(" << tc.expected_pad_x << ", "
            << tc.expected_pad_y << ")" << ", actual=(" << prep.pad_x << ", " << prep.pad_y << ")";

        throw std::runtime_error(oss.str());
    }

    // 4. 人工构造 YOLO 输出
    //
    // 输出形状：[84, 1]
    // 4 个框参数 + 80 个类别分数
    // 只有 1 个候选框
    std::vector<float> output(84, 0.0F);

    output[0] = tc.model_cx;
    output[1] = tc.model_cy;
    output[2] = tc.model_width;
    output[3] = tc.model_height;

    output[4 + kClassId] = kConfidence;

    // 5. 调用真实的后处理函数
    const auto detections = edge_ai::postprocess_yolo(output.data(), 1, prep, 0.25F, 0.45F);

    // 6. 验证检测数量
    if (detections.size() != 1)
    {
        throw std::runtime_error(tc.name + ": detection count mismatch");
    }

    const auto &det = detections[0];

    // 7. 验证类别和置信度
    if (det.class_id != kClassId)
    {
        throw std::runtime_error(tc.name + ": class ID mismatch");
    }

    expect_near(tc.name + ": confidence", det.confidence, kConfidence);

    // 8. 验证坐标还原
    expect_near(tc.name + ": x", det.box.x, tc.expected_box.x);

    expect_near(tc.name + ": y", det.box.y, tc.expected_box.y);

    expect_near(tc.name + ": width", det.box.width, tc.expected_box.width);

    expect_near(tc.name + ": height", det.box.height, tc.expected_box.height);

    std::cout << "[PASS] " << tc.name << '\n';
}

} // namespace

int main()
{
    try
    {
        const std::vector<TestCase> tests = {

            // ==================================
            // Test 1：横向图像
            // 1280×720 -> 640×360
            // 上下各填充 140
            // ==================================
            {"Landscape 1280x720", 1280, 720, 0.5F, 0, 140,

             // 模型输出 cx,cy,w,h
             200.0F, 240.0F, 200.0F, 100.0F,

             // 原图期望 x,y,w,h
             cv::Rect2f(200.0F, 100.0F, 400.0F, 200.0F)},

            // ==================================
            // Test 2：竖向图像
            // 720×1280 -> 360×640
            // 左右各填充 140
            // ==================================
            {"Portrait 720x1280", 720, 1280, 0.5F, 140, 0,

             240.0F, 200.0F, 100.0F, 200.0F,

             cv::Rect2f(100.0F, 200.0F, 200.0F, 400.0F)},

            // ==================================
            // Test 3：小尺寸输入
            // 320×240 -> 640×480
            // 上下各填充 80
            // Scale = 2
            // ==================================
            {"Small image 320x240", 320, 240, 2.0F, 0, 80,

             200.0F, 240.0F, 200.0F, 160.0F,

             cv::Rect2f(50.0F, 40.0F, 100.0F, 80.0F)},

            // ==================================
            // Test 4：奇数 Padding（左右）
            //
            // 719×1280 -> 359×640
            // 总水平 Padding = 281
            // left=140, right=141
            // ==================================
            {"Odd horizontal padding", 719, 1280, 0.5F, 140, 0,

             240.0F, 200.0F, 100.0F, 200.0F,

             cv::Rect2f(100.0F, 200.0F, 200.0F, 400.0F)},

            // ==================================
            // Test 5：奇数 Padding（上下）
            //
            // 1280×719 -> 640×359
            // 总垂直 Padding = 281
            // top=140, bottom=141
            // ==================================
            {"Odd vertical padding", 1280, 719, 0.5F, 0, 140,

             200.0F, 240.0F, 200.0F, 100.0F,

             cv::Rect2f(200.0F, 100.0F, 400.0F, 200.0F)},

            // ==================================
            // Test 6：左上角越界
            //
            // 原始预测框：
            // x1=-50, y1=-20
            // x2=200, y2=100
            //
            // 裁剪后：
            // (0,0,200,100)
            // ==================================
            {"Clip top-left", 1280, 720, 0.5F, 0, 140,

             37.5F, 160.0F, 125.0F, 60.0F,

             cv::Rect2f(0.0F, 0.0F, 200.0F, 100.0F)},

            // ==================================
            // Test 7：右下角越界
            //
            // 原图：720×1280
            //
            // 原始预测框：
            // x1=600, y1=1100
            // x2=800, y2=1400
            //
            // 裁剪后：
            // (600,1100,120,180)
            // ==================================
            {"Clip bottom-right", 720, 1280, 0.5F, 140, 0,

             490.0F, 625.0F, 100.0F, 150.0F,

             cv::Rect2f(600.0F, 1100.0F, 120.0F, 180.0F)},

            // ==================================
            // Test 8：正方形输入，无 Padding
            // 640×640 -> 640×640
            // ==================================
            {"Square image 640x640", 640, 640, 1.0F, 0, 0,

             200.0F, 250.0F, 200.0F, 100.0F,

             cv::Rect2f(100.0F, 200.0F, 200.0F, 100.0F)}};

        for (const auto &tc : tests)
        {
            run_test_case(tc);
        }

        std::cout << "\nP3-06 Coordinate Restore: ALL PASS\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}