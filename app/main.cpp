#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cstddef>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc != 3 || std::string(argv[1]) != "--video")
    {
        std::cerr << "Usage: " << argv[0] << " --video <video_path>\n";
        return 1;
    }

    const std::string video_path = argv[2];

    // 初始化video_path下的视频资源到capture下
    cv::VideoCapture capture(video_path);

    // 确认资源是否打开成功
    if (!capture.isOpened())
    {
        std::cerr << "[Error] Failed to open video: " << video_path << '\n';
        return 2;
    }

    // 读取视频帧率
    const double source_fps = capture.get(cv::CAP_PROP_FPS);

    // frame为视频的一帧（本质为矩阵，矩阵中的值为一个一个色块的值）
    cv::Mat frame;
    std::size_t frame_count = 0;

    int frame_width = 0;
    int frame_height = 0;

    // 记录视频总帧数和分辨率
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            std::cerr << "[Error] Received an empty frame\n";
            return 3;
        }

        if (frame_count == 0)
        {
            frame_width = frame.cols;
            frame_height = frame.rows;
        }

        ++frame_count;
    }

    if (frame_count == 0)
    {
        std::cerr << "[Error] Video contains no readable frames\n";
        return 4;
    }

    std::cout << "[Source] Video: " << video_path << '\n';
    std::cout << "[Source] Resolution: " << frame_width << 'x' << frame_height << '\n';
    std::cout << "[Source] FPS metadata: " << source_fps << '\n';
    std::cout << "[Runtime] Frames read: " << frame_count << '\n';

    return 0;
}