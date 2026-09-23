#include <opencv2/core.hpp>
#include <iostream>

int main()
{
    cv::Mat frame(1, 1, CV_8UC3);

    std::cout << "OpenCV version: " << CV_VERSION << '\n';
    std::cout << "Frame empty: " << std::boolalpha << frame.empty() << '\n';

    return 0;
}
