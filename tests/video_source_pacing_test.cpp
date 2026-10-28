
#include <edge_ai/video_source.hpp>

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using Clock = std::chrono::steady_clock;

struct ReadStats
{
    std::size_t frames = 0;
    double elapsed_seconds = 0.0;
    std::uint64_t skipped_slots = 0;
};


// 测试普通读取和 30 FPS 节拍读取
ReadStats measure_video(
    const std::string& path,
    double target_fps)
{
    edge_ai::VideoSource source(path, target_fps);

    if (!source.is_open())
    {
        throw std::runtime_error("Failed to open video");
    }

    cv::Mat frame;
    std::size_t count = 0;

    Clock::time_point first_time{};
    Clock::time_point last_time{};

    while (source.read(frame))
    {
        if (frame.empty())
        {
            throw std::runtime_error("Empty video frame");
        }

        const auto now = Clock::now();

        if (count == 0)
        {
            first_time = now;
        }

        last_time = now;
        ++count;
    }

    ReadStats stats;
    stats.frames = count;

    if (count > 1)
    {
        stats.elapsed_seconds =
            std::chrono::duration<double>(
                last_time - first_time
            ).count();
    }

    stats.skipped_slots = source.skipped_slots();

    return stats;
}


// 模拟慢速消费者，每处理一帧休眠 50 ms
ReadStats measure_slow_consumer(
    const std::string& path)
{
    edge_ai::VideoSource source(path, 30.0);

    if (!source.is_open())
    {
        throw std::runtime_error("Failed to open video");
    }

    cv::Mat frame;
    std::size_t count = 0;

    Clock::time_point first_time{};
    Clock::time_point last_time{};

    while (source.read(frame))
    {
        if (frame.empty())
        {
            throw std::runtime_error("Empty video frame");
        }

        const auto now = Clock::now();

        if (count == 0)
        {
            first_time = now;
        }

        last_time = now;
        ++count;

        std::this_thread::sleep_for(
            std::chrono::milliseconds(50)
        );
    }

    ReadStats stats;
    stats.frames = count;

    if (count > 1)
    {
        stats.elapsed_seconds =
            std::chrono::duration<double>(
                last_time - first_time
            ).count();
    }

    stats.skipped_slots = source.skipped_slots();

    return stats;
}


int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr
            << "Usage: video_source_pacing_test <video_path>\n";
        return 1;
    }

    try
    {
        const std::string path = argv[1];

        // 1. 普通读取：不开启 Pacing
        const auto normal = measure_video(path, 0.0);

        // 2. 正常 30 FPS 回放
        const auto paced = measure_video(path, 30.0);

        // 3. 模拟每帧处理 50 ms 的慢速消费者
        const auto slow = measure_slow_consumer(path);

        if (paced.frames < 2 || paced.elapsed_seconds <= 0.0)
        {
            throw std::runtime_error(
                "Insufficient pacing samples"
            );
        }

        const double actual_fps =
            static_cast<double>(paced.frames - 1) /
            paced.elapsed_seconds;

        // ==============================
        // 打印测试数据
        // ==============================

        std::cout
            << "\n========== VideoSource Pacing ==========\n";

        std::cout
            << "Normal frames         : "
            << normal.frames << '\n';

        std::cout
            << "Normal source skips   : "
            << normal.skipped_slots << '\n';

        std::cout
            << "Paced frames          : "
            << paced.frames << '\n';

        std::cout
            << "Paced elapsed         : "
            << paced.elapsed_seconds << " s\n";

        std::cout
            << "Actual output FPS     : "
            << actual_fps << '\n';

        std::cout
            << "Paced source skips    : "
            << paced.skipped_slots << '\n';

        std::cout
            << "Slow consumer frames  : "
            << slow.frames << '\n';

        std::cout
            << "Slow consumer elapsed : "
            << slow.elapsed_seconds << " s\n";

        std::cout
            << "Slow source skips     : "
            << slow.skipped_slots << '\n';


        // ==============================
        // 验收断言
        // ==============================

        // 原始测试视频共有 478 帧
        if (normal.frames != 478)
        {
            throw std::runtime_error(
                "Normal mode frame count mismatch"
            );
        }

        // 不启用节拍，不应该产生时间槽跳帧
        if (normal.skipped_slots != 0)
        {
            throw std::runtime_error(
                "Unexpected source skips in normal mode"
            );
        }

        // 60 FPS -> 30 FPS，最多输出 239 帧
        if (paced.frames > 239)
        {
            throw std::runtime_error(
                "Paced frame count exceeded 239"
            );
        }

        // Source Skip 只统计额外跳过的目标时间槽
        // 不包含 60 -> 30 的正常抽帧
        if (paced.frames + paced.skipped_slots > 239)
        {
            throw std::runtime_error(
                "Paced source slot accounting mismatch"
            );
        }

        // 正常负载下的输出帧率应接近 30 FPS
        if (actual_fps < 27.0 || actual_fps > 33.0)
        {
            throw std::runtime_error(
                "Actual pacing FPS outside tolerance"
            );
        }

        // 慢速消费者应该触发额外跳帧
        if (slow.skipped_slots == 0)
        {
            throw std::runtime_error(
                "Slow consumer did not skip source slots"
            );
        }

        if (slow.frames >= paced.frames)
        {
            throw std::runtime_error(
                "Slow consumer did not reduce output frames"
            );
        }

        if (slow.frames + slow.skipped_slots > 239)
        {
            throw std::runtime_error(
                "Slow consumer slot accounting mismatch"
            );
        }

        // 当前测试视频的时间尺度约 8 秒。
        // 使用较宽的容差，允许调度波动。
        if (slow.elapsed_seconds < 7.0 ||
            slow.elapsed_seconds > 9.5)
        {
            throw std::runtime_error(
                "Slow consumer playback duration mismatch"
            );
        }

        std::cout
            << "\nVideoSource Pacing: ALL PASS\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}
