#include <edge_ai/runtime.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <edge_ai/logger.hpp>

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
        edge_ai::Logger::error(std::string("Failed to open video: ") + source_.path());
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
            edge_ai::Logger::error(std::string("Received an empty frame"));
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
            edge_ai::Logger::info(std::string("[Runtime] User requested exit"));
            break;
        }
    }

    const auto end_time = std::chrono::steady_clock::now();

    if (frame_count == 0)
    {
        edge_ai::Logger::error(std::string("Video contains no readable frames"));
        return 4;
    }
    
    const double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    const double processing_fps = static_cast<double>(frame_count) / elapsed_seconds;
    const double processing_avetime = (elapsed_seconds * 1000.0) / static_cast<double>(frame_count);

    edge_ai::Logger::info(std::string("[Source] Video: ") + source_.path());
    edge_ai::Logger::info(std::string("[Source] Resolution: ") + std::to_string(frame_width) + 'x' + std::to_string(frame_height));
    edge_ai::Logger::info(std::string("[Source] FPS metadata: ") + std::to_string(source_.fps()));

    edge_ai::Logger::info(std::string("[Runtime] Frames processed: ") + std::to_string(frame_count));
    edge_ai::Logger::info(std::string("[Runtime] Elapsed time: ") + std::to_string(elapsed_seconds) + 's');
    edge_ai::Logger::info(std::string("[Runtime] Processing FPS: ") + std::to_string(processing_fps));
    edge_ai::Logger::info(std::string("[Runtime] Processing Average time: ") + std::to_string(processing_avetime) + "ms");

    return 0;
}

}  // namespace edge_ai