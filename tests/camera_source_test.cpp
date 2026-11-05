
#include <edge_ai/camera_source.hpp>
#include <edge_ai/frame_source.hpp>

#include <opencv2/core.hpp>

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>

int main()
{
    try
    {
        // 使用一个预期不存在的摄像头编号
        constexpr int invalid_device_index = 9999;

        // 通过基类指针持有派生类对象：
        // 验证 FrameSource 多态接口
        std::unique_ptr<edge_ai::FrameSource> source =
            std::make_unique<edge_ai::CameraSource>(invalid_device_index);

        // 1. 不存在的设备不应打开成功
        if (source->is_open())
        {
            throw std::runtime_error("Unexpected camera open success");
        }

        // 2. 模拟调用者传入的旧图像
        cv::Mat frame(2, 2, CV_8UC3, cv::Scalar(255, 0, 0));

        if (frame.empty())
        {
            throw std::runtime_error("Test frame initialization failed");
        }

        // 3. 未打开的摄像头不应读取成功
        if (source->read(frame))
        {
            throw std::runtime_error("Read unexpectedly succeeded");
        }

        // 4. 读取失败后必须清空旧图像
        if (!frame.empty())
        {
            throw std::runtime_error("Stale frame was not cleared");
        }

        // 5. CameraSource 使用基类默认的跳帧统计
        if (source->skipped_slots() != 0)
        {
            throw std::runtime_error("Unexpected camera source skips");
        }

        std::cout << "CameraSource: ALL PASS\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
