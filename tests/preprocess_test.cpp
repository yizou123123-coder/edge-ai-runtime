#include <edge_ai/preprocess.hpp>

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

int main()
{
    const std::string video_path = "/home/yizou/datasets/edge-ai-runtime/test.mp4";

    cv::VideoCapture cap(video_path);

    if(!cap.isOpened())
    {
        std::cerr << "Failed to open video: " << video_path << '\n';
        return 1;
    }

    cv::Mat frame;

    if (!cap.read(frame) || frame.empty())
    {
        std::cerr << "Failed to read first frame from video\n";
        return 1;
    }
    
    const auto result = edge_ai::preprocess_yolo(frame, 640, 640);

    std::cout << "original_width: " << result.original_width << '\n';
    std::cout << "original_height: " << result.original_height << '\n';

    const std::size_t expected_size = 3ULL * 640ULL * 640ULL;

    if (result.tensor_data.size() != expected_size)
    {
        std::cerr << "Unexpected tensor size\n";
        return 1;
    }

    std::cout << result.tensor_data.size() << '\n';
    std::cout << "scale: " << result.scale << '\n';
    std::cout << "pad_x: " << result.pad_x << '\n';
    std::cout << "pad_y: " << result.pad_y << '\n';

    if (result.scale <= 0.0F || result.pad_x < 0 || result.pad_y < 0)
    {
        std::cerr << "Invalid scale or pad values!\n";
        return 1;
    }
    
    const auto [min_it, max_it] = std::minmax_element(result.tensor_data.begin(), result.tensor_data.end());

    const float min_val = *min_it;
    const float max_val = *max_it;
    std::cout << "min_val: " << min_val << ", max_val: " << max_val << '\n';
    constexpr float eps = 1e-5F;
    if (min_val < -eps || max_val > 1.0F + eps)
    {
        std::cerr << "Tensor values are out of [0.0, 1.0] range!\n";
        return 1;
    }


    std::cout << "Preprocess test: PASS\n";

    return 0;
}