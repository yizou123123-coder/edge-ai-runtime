#include <edge_ai/preprocess.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>

int main()
{
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(10, 20, 30));

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

    const auto [min_it, max_it] =
        std::minmax_element(result.tensor_data.begin(), result.tensor_data.end());

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