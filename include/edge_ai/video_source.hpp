#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <chrono>
#include <cstdint>
#include <edge_ai/frame_source.hpp>

namespace edge_ai
{

class VideoSource : public FrameSource
{
  public:
    // explicit：修饰构造函数，只允许显示转换（单参数构造函数常用）
    explicit VideoSource(const std::string &video_path, double target_fps = 0.0);

    bool is_open() const override;
    bool read(cv::Mat &frame) override;

    double fps() const;
    const std::string &path() const;

    std::uint64_t skipped_slots() const override;

  private:
    std::string video_path_;
    cv::VideoCapture capture_;

    using Clock = std::chrono::steady_clock;
    double target_fps_{0.0}; // 希望模拟的发帧速率

    Clock::duration frame_interval_{};   // 帧间隔时长
    Clock::time_point playback_start_{}; // 第一帧实际成功提供时的时间点
    std::uint64_t last_emitted_slot_{0}; // 上一次成功提供的视频帧对应哪个时间槽

    std::uint64_t emitted_frames_{0}; // 已发帧计数器

    std::uint64_t skipped_slots_{0}; // 跳帧数量
};

} // namespace edge_ai