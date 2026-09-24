#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
    // 处理后的一帧数据
    cv::Mat processed_frame;

    int frame_width = 0;
    int frame_height = 0;

    while (capture.read(frame))
    {
        if (frame.empty())
        {
            std::cerr << "[Error] Received an empty frame\n";
            return 3;
        }

        if (frame_count == 0)
        {
            frame_width = frame.cols;    // 行
            frame_height = frame.rows;   // 列
        }

        // 在视频的读取循环中，对每一帧图像进行缩放（Resize）处理
        cv::resize(
        frame,
        processed_frame,
        cv::Size(),
        0.5,
        0.5,
        cv::INTER_LINEAR
        );

        // 将缩放处理好的视频帧实时显示在电脑屏幕上
        cv::imshow("Edge AI Runtime", processed_frame);

        ++frame_count;
        
        // 等待 1 毫秒，并检测用户有没有在键盘上按键
        const int key = cv::waitKey(1);

        if (key == 'q' || key == 27)
        {
            std::cout << "[Runtime] User requested exit\n";
            break;
        }
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