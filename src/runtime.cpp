#include <edge_ai/runtime.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <iostream>
#include <chrono>

namespace edge_ai
{

Runtime::Runtime(const std::string& video_path)
    : source_(video_path)
{
}

int Runtime::run()
{
    if (!source_.is_open())
    {
        std::cerr << "[Error] Failed to open video: " << source_.path() << '\n';
        return 2;
    }

    const auto start_time = std::chrono::steady_clock::now();

    cv::Mat frame;
    cv::Mat processed_frame;

    std::size_t frame_count = 0;

    int frame_width = 0;
    int frame_height = 0;

    while (source_.read(frame))
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

        cv::resize(
            frame,
            processed_frame,
            cv::Size(),
            0.5,
            0.5,
            cv::INTER_LINEAR
        );

        cv::imshow("Edge AI Runtime", processed_frame);

        ++frame_count;

        const int key = cv::waitKey(1);

        if (key == 'q' || key == 27)
        {
            std::cout << "[Runtime] User requested exit\n";
            break;
        }
    }

    const auto end_time = std::chrono::steady_clock::now();

    if (frame_count == 0)
    {
        std::cerr << "[Error] Video contains no readable frames\n";
        return 4;
    }
    
    const double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    const double processing_fps = static_cast<double>(frame_count) / elapsed_seconds;
    const double processing_avetime = elapsed_seconds / static_cast<double>(frame_count);


    std::cout << "[Source] Video: " << source_.path() << '\n';
    std::cout << "[Source] Resolution: " << frame_width << 'x' << frame_height << '\n';
    std::cout << "[Source] FPS metadata: " << source_.fps() << '\n';

    std::cout << "[Runtime] Frames processed: " << frame_count << '\n';
    std::cout << "[Runtime] Elapsed time: " << elapsed_seconds << " s\n";
    std::cout << "[Runtime] Processing FPS: " << processing_fps << '\n';
    std::cout << "[Runtime] Processing Average time: " << processing_avetime << '\n';


    return 0;
}

}  // namespace edge_ai