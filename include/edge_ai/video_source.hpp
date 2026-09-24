#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>

namespace edge_ai
{

class VideoSource
{
public:
    // explicit：修饰构造函数，只允许显示转换（单参数构造函数常用）
    explicit VideoSource(const std::string& video_path);

    bool is_open() const;
    bool read(cv::Mat& frame);

    double fps() const;
    const std::string& path() const;

private:
    std::string video_path_;
    cv::VideoCapture capture_;
};

}  // namespace edge_ai