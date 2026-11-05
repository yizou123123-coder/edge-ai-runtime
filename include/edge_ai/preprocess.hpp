#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace edge_ai
{

struct PreprocessResult
{
    // 输入张量
    std::vector<float> tensor_data;

    // 以下都是映射回原图的数据
    // 缩放系数
    float scale{1.0F};

    // 四周补黑边时，水平和垂直方向各垫了多少像素
    int pad_x{0};
    int pad_y{0};

    // 视频原图的原始宽高
    int original_width{0};
    int original_height{0};
};

// 预处理函数，返回结构体
PreprocessResult preprocess_yolo(const cv::Mat &image, int target_width, int target_height);

} // namespace edge_ai