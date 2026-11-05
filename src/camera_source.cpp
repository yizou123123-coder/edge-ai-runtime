#include <edge_ai/camera_source.hpp>

namespace edge_ai
{

// 构造函数：打开指定的 Linux 摄像头
CameraSource::CameraSource(int device_index) : capture_(device_index, cv::CAP_V4L2)
{
    // 本轮不需要额外初始化其他成员
}

// 检查摄像头是否成功打开
bool CameraSource::is_open() const { return capture_.isOpened(); }

// 获取下一帧图像
bool CameraSource::read(cv::Mat &frame)
{
    if (!is_open())
    {
        frame.release();
        return false;
    }

    if (!capture_.read(frame))
    {
        frame.release();
        return false;
    }

    if (frame.empty())
    {
        return false;
    }
    return true;
}
} // namespace edge_ai
