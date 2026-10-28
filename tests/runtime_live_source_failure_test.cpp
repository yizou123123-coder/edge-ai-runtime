
#include <edge_ai/frame_source.hpp>
#include <edge_ai/runtime.hpp>

#include <opencv2/core.hpp>

#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

// 模拟一个已经打开的实时摄像头。
// 成功输出指定数量的帧后，模拟设备读取失败。
class MockLiveSource final : public edge_ai::FrameSource
{
public:
    explicit MockLiveSource(std::size_t valid_frames)
        : valid_frames_(valid_frames)
    {
    }

    bool is_open() const override
    {
        return true;
    }

    bool is_live() const override
    {
        return true;
    }

    bool read(cv::Mat& frame) override
    {
        // 模拟摄像头突然断开
        if (emitted_frames_ >= valid_frames_)
        {
            frame.release();
            return false;
        }

        // 创建一张有效的 BGR 测试图像
        frame = cv::Mat(
            640,
            640,
            CV_8UC3,
            cv::Scalar(0, 0, 0)
        );

        ++emitted_frames_;
        return true;
    }

private:
    std::size_t valid_frames_{0};
    std::size_t emitted_frames_{0};
};


// 测试：实时输入源在输出 N 帧后失败
void test_live_source_failure(std::size_t valid_frames)
{
    std::cout
        << "\n========== Mock Camera: "
        << valid_frames
        << " valid frames ==========\n";

    // 通过 FrameSource 多态接口传递输入源
    std::unique_ptr<edge_ai::FrameSource> source =
        std::make_unique<MockLiveSource>(valid_frames);

    // Blocking 保证这次生成的少量测试帧不被队列策略丢弃
    edge_ai::Runtime runtime(
        std::move(source),
        edge_ai::QueuePolicy::Blocking
    );

    if (!runtime.start())
    {
        throw std::runtime_error(
            "Runtime failed to start MockLiveSource"
        );
    }

    // 等待 Capture 和 Processing 线程正常结束
    runtime.join();

    // 实时输入源读取失败必须返回 7
    if (runtime.exit_code() != 7)
    {
        throw std::runtime_error(
            "Expected exit code 7, got " +
            std::to_string(runtime.exit_code())
        );
    }

    std::cout
        << "[PASS] Live source failure after "
        << valid_frames
        << " frames\n";
}


int main()
{
    try
    {
        // 场景 A：第一帧立即读取失败
        test_live_source_failure(0);

        // 场景 B：成功读取 3 帧后断开
        test_live_source_failure(3);

        std::cout
            << "\nRuntime Live Source Failure: ALL PASS\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}
