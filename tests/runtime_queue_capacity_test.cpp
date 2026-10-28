
#include <edge_ai/frame_source.hpp>
#include <edge_ai/runtime.hpp>

#include <opencv2/core.hpp>

#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

// 快速生成指定数量的测试帧
class BurstSource final : public edge_ai::FrameSource
{
public:
    explicit BurstSource(std::size_t total_frames)
        : total_frames_(total_frames)
    {
    }

    bool is_open() const override
    {
        return true;
    }

    bool read(cv::Mat& frame) override
    {
        if (next_frame_ >= total_frames_)
        {
            frame.release();
            return false;
        }

        frame = cv::Mat(
            640,
            640,
            CV_8UC3,
            cv::Scalar(0, 0, 0)
        );

        ++next_frame_;
        return true;
    }

private:
    std::size_t total_frames_{0};
    std::size_t next_frame_{0};
};


// 测试指定容量下 Runtime 能否正常工作
void test_capacity(std::size_t capacity)
{
    constexpr std::size_t kFrames = 24;

    std::cout
        << "\n========== Queue Capacity: "
        << capacity
        << " ==========\n";

    auto source =
        std::make_unique<BurstSource>(kFrames);

    edge_ai::Runtime runtime(
        std::move(source),
        edge_ai::QueuePolicy::Blocking,
        capacity
    );

    if (!runtime.start())
    {
        throw std::runtime_error(
            "Runtime failed to start"
        );
    }

    runtime.join();

    if (runtime.exit_code() != 0)
    {
        throw std::runtime_error(
            "Runtime failed with capacity " +
            std::to_string(capacity)
        );
    }

    std::cout
        << "[PASS] Runtime capacity "
        << capacity
        << " completed\n";
}


// 测试容量为 0 时必须拒绝
void test_invalid_capacity()
{
    std::cout
        << "\n========== Invalid Capacity ==========\n";

    bool rejected = false;

    try
    {
        auto source =
            std::make_unique<BurstSource>(1);

        edge_ai::Runtime runtime(
            std::move(source),
            edge_ai::QueuePolicy::Blocking,
            0
        );
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    if (!rejected)
    {
        throw std::runtime_error(
            "Queue capacity 0 was accepted"
        );
    }

    std::cout
        << "[PASS] Capacity 0 rejected\n";
}


int main()
{
    try
    {
        test_capacity(1);
        test_capacity(8);
        test_invalid_capacity();

        std::cout
            << "\nRuntime Queue Capacity: ALL PASS\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}
