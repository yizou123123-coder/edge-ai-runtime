#pragma once

#include <edge_ai/video_source.hpp>
#include <string>
#include <atomic>
#include <thread>
#include <edge_ai/blocking_queue.hpp>

namespace edge_ai
{

class Runtime
{
public:
    explicit Runtime(const std::string& video_path);

    ~Runtime();

    bool start();
    void stop();
    void join();

    int exit_code() const;

private:
    void capture_loop();
    void processing_loop();

    VideoSource source_;

    BlockingQueue<cv::Mat> frame_queue_;

    std::thread capture_thread;
    std::thread processing_thread;

    // 下达停止通知
    std::atomic<bool> stop_requested_{false}; 
    // 判断退出状态
    std::atomic<int> exit_code_{0};

    // 是否开始的flag
    bool started_{false};
};

}  // namespace edge_ai