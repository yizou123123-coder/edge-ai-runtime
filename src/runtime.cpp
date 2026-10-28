#include <edge_ai/runtime.hpp>
#include <edge_ai/logger.hpp>
#include <edge_ai/detector.hpp>

#include <cstddef>
#include <iostream>
#include <chrono>
#include <utility>
#include <exception>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace edge_ai
{
// 通用
Runtime::Runtime(std::unique_ptr<FrameSource> source, QueuePolicy policy, std::size_t queue_capacity, std::string model_path, double deadline_ms,
                 double confidence_threshold, double nms_iou_threshold, bool display_enabled)
    : source_(std::move(source)),
      frame_queue_(queue_capacity, policy),
      model_path_(std::move(model_path)),
      deadline_ms_(deadline_ms),
      confidence_threshold_(confidence_threshold),
      nms_iou_threshold_(nms_iou_threshold),
      display_enabled_(display_enabled)
{
    if (source_ == nullptr)
    {
        throw std::invalid_argument("Runtime requires a non-null FrameSource");
    }
    if (model_path_.empty())
    {
        throw std::invalid_argument("model_path error");
    }
    if (!std::isfinite(deadline_ms_) || deadline_ms_ <= 0)
    {
        throw std::invalid_argument("deadline_ms error");
    }
    if (!std::isfinite(confidence_threshold_) || confidence_threshold_ < 0 || confidence_threshold_ > 1)
    {
        throw std::invalid_argument("confidence_threshold_ error");
    }
    if (!std::isfinite(nms_iou_threshold_) || nms_iou_threshold_ < 0 || nms_iou_threshold_ > 1)
    {
        throw std::invalid_argument("nms_iou_threshold_ error");
    }
    
}

// old
Runtime::Runtime(const std::string& video_path, double target_fps, QueuePolicy policy)
    : Runtime(std::make_unique<VideoSource>(video_path, target_fps), policy)
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

    if (!source_->is_open())
    {
        Logger::error("[Runtime] Failed to open frame source");
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
    display_queue_.close();
}


int Runtime::exit_code() const
{
    return exit_code_.load();
}

// 辅助函数：用于排序，并给出Frame Age数据
double percentile(std::vector<double> samples, double p)
{
    if (samples.empty() || p <= 0.0 || p > 1.0)
    {
        throw std::invalid_argument("Invalid percentile input");
    }

    std::sort(samples.begin(), samples.end());

    const std::size_t index =static_cast<std::size_t>(std::ceil(p * samples.size())) - 1;

    return samples[index];
}


void Runtime::processing_loop()
{
    Frame frame;
    std::size_t processed_frames = 0;
    const auto start_time = std::chrono::steady_clock::now();
    std::vector<double> frame_ages_ms; // 每一帧处理时间组成动态数组
    std::size_t deadline_misses = 0;   // 处理帧超时的数量
    try
    {
        OnnxDetector detector(model_path_, confidence_threshold_, nms_iou_threshold_);
        while (frame_queue_.pop(frame))
        {
            auto detections = detector.detect(frame.image);
            const auto output_at = std::chrono::steady_clock::now();
            const double frame_age_ms = std::chrono::duration<double, std::milli>(output_at - frame.capture_ts).count();
            frame_ages_ms.push_back(frame_age_ms);
            if (frame_age_ms > deadline_ms_)
            {
                deadline_misses++;
            }

            if (frame.id == 0)
            {
                 Logger::info("[Processing] Frame ID: " + std::to_string(frame.id) +
                              ", Detections: " + std::to_string(detections.size()) +
                              ", Frame Age: " + std::to_string(frame_age_ms) + " ms");
            }
            
            ++processed_frames;

            if (display_enabled_)
            {
                InferenceResult result;

                result.frame = std::move(frame);

                result.detections = std::move(detections);
                    
                (void)display_queue_.push(std::move(result));
            }
        }

        Logger::info("[Processing] Deadline Budget: " + std::to_string(deadline_ms_) + " ms");
    }
    catch(const std::exception& e)
    {
        Logger::error(std::string("[Processing] ") + e.what());
        int expected = 0;
        exit_code_.compare_exchange_strong(expected, 6);
        stop();

        display_queue_.close();
        display_finished_.store(true);

        return;
    }
    

    const auto end_time = std::chrono::steady_clock::now();

    const double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    const double processing_fps = static_cast<double>(processed_frames) / elapsed_seconds;

    Logger::info(std::string("[Processing] Elapsed time: ") + std::to_string(elapsed_seconds) + 's');
    Logger::info(std::string("[Processing] Processing FPS: ") + std::to_string(processing_fps));
    Logger::info("[Processing] Frames processed: "+ std::to_string(processed_frames));

    if (!frame_ages_ms.empty())
    {
        const double p50 = percentile(frame_ages_ms, 0.50);
        const double p95 = percentile(frame_ages_ms, 0.95);
        const double p99 = percentile(frame_ages_ms, 0.99);

        Logger::info("[Processing] Frame Age P50: " +std::to_string(p50) + " ms");
        Logger::info("[Processing] Frame Age P95: " +std::to_string(p95) + " ms");
        Logger::info("[Processing] Frame Age P99: " +std::to_string(p99) + " ms");

        const double deadline_miss_rate =100.0 * static_cast<double>(deadline_misses) /static_cast<double>(frame_ages_ms.size());
        Logger::info("[Processing] Deadline Misses: " + std::to_string(deadline_misses));
        Logger::info("[Processing] Deadline Miss Rate: " +std::to_string(deadline_miss_rate) + "%");
    }
    else
    {
        Logger::info("[Processing] No frames available for latency statistics");
    }

    // 计算On-time Rate（按时输出率）
    // 已经成功输出的帧数
    const std::size_t output_count = frame_ages_ms.size();

    // 被队列策略主动丢弃的帧数
    const std::size_t queue_dropped = frame_queue_.dropped_count();

    // 正常 EOF 并排空队列后：
    // Accepted = Processed + Dropped
    const std::size_t accepted_count = output_count + queue_dropped;

    // 在 deadline_ms_ 指定的预算内完成的帧数
    const std::size_t on_time_count = output_count - deadline_misses;

    if (accepted_count > 0)
    {
        const double on_time_rate = 100.0 * static_cast<double>(on_time_count) / static_cast<double>(accepted_count);

        const double queue_drop_rate = 100.0 * static_cast<double>(queue_dropped) / static_cast<double>(accepted_count);

        Logger::info("[Processing] On-time Frames: " + std::to_string(on_time_count));

        Logger::info("[Processing] On-time Rate: " + std::to_string(on_time_rate) + "%");

        Logger::info("[Processing] Queue Drop Rate: " + std::to_string(queue_drop_rate) + "%");
    }

    display_queue_.close();
    display_finished_.store(true);
}

void Runtime::capture_loop()
{
    std::size_t captured_frames = 0;
    std::size_t peak_queue_size = 0;

    cv::Mat raw_frame;
    std::uint64_t next_frame_id = 0;

    const auto start_time = std::chrono::steady_clock::now();

    while(!stop_requested_.load() && source_->read(raw_frame))
    {
        const auto captured_at = std::chrono::steady_clock::now();

        if (raw_frame.empty())
        {
            Logger::error("Received an empty frame");
            exit_code_.store(3);
            stop(); 
            return;
        }

        Frame frame;
        frame.id = next_frame_id;
        next_frame_id++;
        frame.capture_ts = captured_at;
        frame.image = std::move(raw_frame);

        if(!frame_queue_.push(std::move(frame)))
        {
            break; 
        }

        ++captured_frames;
    }
    
    // 判断是否为实时输入源读取失败（camera）
    const bool live_source_failed = !stop_requested_.load() && source_->is_live();
    if (live_source_failed)
    {
        Logger::error("[Capture] Live frame source read failed");
        int expected = 0;
        exit_code_.compare_exchange_strong(expected, 7);
    }

    const auto end_time = std::chrono::steady_clock::now();

    if (captured_frames == 0)
    {
        if(stop_requested_.load())
        {
            Logger::info("[Capture] Stop requested");           
        }
        // video
        else if(!live_source_failed)
        {
            Logger::error("[Capture] Video contains no readable frames");
            int expected = 0;
            exit_code_.compare_exchange_strong(expected, 4);
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

    Logger::info("[Capture] Queue Dropped: " +std::to_string(frame_queue_.dropped_count()));

    Logger::info("[Capture] Source Skipped Slots: " + std::to_string(source_->skipped_slots()));
}

bool Runtime::try_pop_display(InferenceResult& result)
{
    return display_queue_.try_pop(result);
}

bool Runtime::display_finished() const
{
    return display_finished_.load();
}
}  // namespace edge_ai