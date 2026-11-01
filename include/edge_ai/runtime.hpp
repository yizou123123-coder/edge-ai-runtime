#pragma once

#include <edge_ai/video_source.hpp>
#include <string>
#include <atomic>
#include <thread>
#include <edge_ai/blocking_queue.hpp>
#include <edge_ai/frame.hpp>
#include <edge_ai/frame_source.hpp>
#include <memory>
#include <cstddef>
#include <edge_ai/inference_result.hpp>

namespace edge_ai
{

class Runtime
{
public:
    // 原有构造函数：保留，兼容现有 main 和测试
    explicit Runtime(const std::string& video_path, double target_fps = 0.0, QueuePolicy policy = QueuePolicy::Blocking);
    
    // 新构造函数：接收任意 FrameSource 实现
    explicit Runtime(std::unique_ptr<FrameSource> source_, 
                     QueuePolicy policy = QueuePolicy::Blocking,
                     std::size_t queue_capacity = 4,
                     std::string model_path = "models/yolov8n.onnx",
                     double deadline_ms = 100.0,
                     double confidence_threshold = 0.25,
                     double nms_iou_threshold = 0.45,
                     bool display_enabled = false,
                     int intra_op_threads = 0);

    ~Runtime();

    bool start();
    void stop();
    void join();

    int exit_code() const;

    bool try_pop_display(InferenceResult& result);

    bool display_finished() const;

private:
    void capture_loop();
    void processing_loop();

    // 创建智能指针source_ 指向的类型为FrameSource
    std::unique_ptr<FrameSource> source_;

    BlockingQueue<Frame> frame_queue_;

    std::thread capture_thread;
    std::thread processing_thread;

    // 下达停止通知
    std::atomic<bool> stop_requested_{false}; 
    // 判断退出状态
    std::atomic<int> exit_code_{0};

    // 是否开始的flag
    bool started_{false};
    // onnx 
    std::string model_path_;

    double deadline_ms_{100.0};
    double confidence_threshold_{0.25};
    double nms_iou_threshold_{0.45};

    bool display_enabled_{false};

    int intra_op_threads_{0};

    // 仅供显示使用，不影响原 FrameQueue
    BlockingQueue<InferenceResult> display_queue_{1, QueuePolicy::LatestFrame};

    // Processing Thread 是否已经结束
    std::atomic<bool> display_finished_{false};
};

}  // namespace edge_ai