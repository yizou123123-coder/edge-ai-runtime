#include <edge_ai/runtime.hpp>
#include <edge_ai/logger.hpp>
#include <edge_ai/detector.hpp>

#include <cstddef>
#include <iostream>
#include <chrono>
#include <utility>
#include <exception>

namespace edge_ai
{

Runtime::Runtime(const std::string& video_path)
    : source_(video_path),
      frame_queue_(4)
{
}


Runtime::~Runtime()
{
    stop();
    join();
}

bool Runtime::start()
{
    if(started_)
    {
        Logger::warn("[Runtime] Runtime can only be started once");
        exit_code_.store(5);
        return false;
    }

    if (!source_.is_open())
    {
        Logger::error("Failed to open video: " + source_.path());
        exit_code_.store(2);
        return false;
    }

    // 防止重复启动线程
    if (processing_thread.joinable() || capture_thread.joinable())
    {
        Logger::warn("[Runtime] thread is already running");
        exit_code_.store(5);
        return false;
    }

    started_ = true;

    // 原子变量stop_requested_ 设为 false
    stop_requested_.store(false);

    exit_code_.store(0);

    // 启动2个新的子线程，这里&为取地址
    processing_thread = std::thread(&Runtime::processing_loop, this);
    Logger::info("[Runtime] processing thread started");

    capture_thread = std::thread(&Runtime::capture_loop, this);
    Logger::info("[Runtime] capture thread started");

    return true;
}


void Runtime::join()
{
    if (capture_thread.joinable())
    {
        capture_thread.join();
    }

    if (processing_thread.joinable())
    {
        processing_thread.join();
    }
}


void Runtime::stop()
{
    stop_requested_.store(true);
    frame_queue_.close();
}


int Runtime::exit_code() const
{
    return exit_code_.load();
}

void Runtime::processing_loop()
{
    cv::Mat frame;
    std::size_t processed_frames = 0;
    const auto start_time = std::chrono::steady_clock::now();

    try
    {
        OnnxDetector detector("models/yolov8n.onnx");
        while (frame_queue_.pop(frame))
        {
            auto detections = detector.detect(frame);
            ++processed_frames;
        }
    }
    catch(const std::exception& e)
    {
        Logger::error(std::string("[Processing] ") + e.what());
        int expected = 0;
        exit_code_.compare_exchange_strong(expected, 6);
        stop();
    }
    

    const auto end_time = std::chrono::steady_clock::now();

    const double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    const double processing_fps = static_cast<double>(processed_frames) / elapsed_seconds;

    Logger::info(std::string("[Processing] Elapsed time: ") + std::to_string(elapsed_seconds) + 's');
    Logger::info(std::string("[Processing] Processing FPS: ") + std::to_string(processing_fps));
    Logger::info("[Processing] Frames processed: "+ std::to_string(processed_frames)); 
}


void Runtime::capture_loop()
{
    cv::Mat frame;
    std::size_t captured_frames = 0;
    std::size_t peak_queue_size = 0;

    const auto start_time = std::chrono::steady_clock::now();

    while(!stop_requested_.load() && source_.read(frame))
    {
        if (frame.empty())
        {
            Logger::error("Received an empty frame");
            exit_code_.store(3);
            stop(); 
            return;
        }

        if(!frame_queue_.push(std::move(frame)))
        {
            break;
        }

        ++captured_frames;
    }
    
    const auto end_time = std::chrono::steady_clock::now();

    if (captured_frames == 0)
    {
        if(stop_requested_.load())
        {
            Logger::info("[Capture] Stop requested");           
        }else
        {
            Logger::error("Video contains no readable frames");
            exit_code_.store(4);
        }
    }

    frame_queue_.close();

    peak_queue_size = frame_queue_.peak_size();

    const double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    const double capture_fps = static_cast<double>(captured_frames) / elapsed_seconds;

    Logger::info(std::string("[Capture] Elapsed time: ") + std::to_string(elapsed_seconds) + 's');
    Logger::info("[Capture] Frames captured: " + std::to_string(captured_frames)); 
    Logger::info(std::string("[Capture] Capture FPS: ") + std::to_string(capture_fps));
    Logger::info(std::string("[Capture] Peak_Queue_Size: ") + std::to_string(peak_queue_size));
}

}  // namespace edge_ai