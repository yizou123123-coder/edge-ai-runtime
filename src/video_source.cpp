#include <edge_ai/video_source.hpp>
#include <edge_ai/logger.hpp>
#include <thread>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace edge_ai
{
VideoSource::VideoSource(const std::string& video_path, double target_fps)
    :video_path_(video_path),
     capture_(video_path),
     target_fps_(target_fps)
{
    if (!std::isfinite(target_fps_) || target_fps_ < 0.0)
    {
        throw std::invalid_argument("Invalid target FPS");
    }
}

bool VideoSource::is_open() const
{
    return capture_.isOpened();
}

bool VideoSource::read(cv::Mat& frame)
{
    if (target_fps_ == 0)
    {
        return capture_.read(frame);
    }
    
    // 第一次读取时初始化
    if (emitted_frames_ == 0)
    {
        if (!capture_.read(frame))
        {
            Logger::error("capture read");
            return false;
        }
        // 帧间隔时长
        frame_interval_ = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / target_fps_));

        // 初始化第一帧实际成功提供时的时间点、槽
        playback_start_ = Clock::now();
        last_emitted_slot_ = 0;

        emitted_frames_++;
        return true;
    }

    const auto now = Clock::now();
    const auto elapsed = now - playback_start_;
    // 计算离上次有多少时间槽
    const auto elapsed_slot = static_cast<decltype(last_emitted_slot_)>(elapsed / frame_interval_);

    const auto next_elapsed_slot = std::max(last_emitted_slot_ + 1, elapsed_slot);

    // 视频帧编号
    const auto last_source_index = 2 * last_emitted_slot_;
    const auto target_source_index = 2 * next_elapsed_slot;

    // 跳多少帧
    auto frames_to_skip = target_source_index - last_source_index - 1;
    auto skipped_this_read = next_elapsed_slot - last_emitted_slot_ - 1;
    for (decltype(frames_to_skip) i = 0; i < frames_to_skip; ++i)
    {
        if (!capture_.grab())
        {
            return false;
        }
    }

    const auto target_time = playback_start_ + next_elapsed_slot * frame_interval_;
    if (Clock::now() < target_time)
    {
        std::this_thread::sleep_until(target_time);
    }

    if (!capture_.read(frame))
    {
        return false;
    }
    skipped_slots_ = skipped_this_read + skipped_slots_;
    last_emitted_slot_ = next_elapsed_slot;
    emitted_frames_++;
    return true;
}

double VideoSource::fps() const{
    return capture_.get(cv::CAP_PROP_FPS);
}

const std::string& VideoSource::path() const{
    return video_path_;
}

std::uint64_t VideoSource::skipped_slots() const
{

    return skipped_slots_;
}
} // namespace edge_ai