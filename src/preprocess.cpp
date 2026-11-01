#include <edge_ai/preprocess.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>


namespace edge_ai
{

PreprocessResult preprocess_yolo(
    const cv::Mat& image,
    int target_width,
    int target_height
)
{
    PreprocessResult result;

    if (image.empty())
    {
        return result;
    }

    const int original_width = image.cols;
    const int original_height = image.rows;
    result.original_width = original_width;
    result.original_height = original_height;

    const float scale_w = static_cast<float>(target_width) / original_width;
    const float scale_h = static_cast<float>(target_height) / original_height;
    const float scale = std::min(scale_w, scale_h);
    result.scale = scale;

    const int resized_width = static_cast<int>(scale * original_width);
    const int resized_height = static_cast<int>(scale * original_height);

    cv::Mat resized_image;
    cv::resize(
            image,
            resized_image,
            cv::Size(resized_width, resized_height),
            0.0,
            0.0,
            cv::INTER_LINEAR
        );
 
    const int pad_x = (target_width - resized_width) / 2;
    const int pad_y = (target_height - resized_height) / 2;
    result.pad_x = pad_x;
    result.pad_y = pad_y;

    const int top = pad_y;
    const int bottom = target_height - resized_height - top;
    const int left = pad_x;
    const int right = target_width - resized_width - left; 

    // 补黑边至目标长宽
    cv::Mat padded_image;
    cv::copyMakeBorder(
        resized_image,
        padded_image,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );


    cv::Mat float_image;
    padded_image.convertTo(float_image, CV_32FC3, 1.0F / 255.0F);

    // 内存排布转换 HWC->CHW
    const std::size_t channel_size = static_cast<std::size_t>(target_width) * static_cast<std::size_t>(target_height);
    result.tensor_data.resize(3 * channel_size);
    // 创建 3 个空的 Mat 头信息
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c)
    {
        channels[c] = cv::Mat(
            target_height,
            target_width,
            CV_32FC1,
            result.tensor_data.data() + (2 - c) * channel_size
        );
    }
    // 通道分离
    cv::split(float_image, channels);
    
    return result;
}

}