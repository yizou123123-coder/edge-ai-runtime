#pragma once

#include <cstdint>
#include <opencv2/core.hpp>

namespace edge_ai
{

class FrameSource
{
public:
    virtual ~FrameSource() = default;

    virtual bool is_open() const = 0;

    virtual bool read(cv::Mat& frame) = 0;

    virtual std::uint64_t skipped_slots() const
    {
        return 0;
    }

    // 输入源是否属于持续实时采集设备：VideoSource 使用默认值 false；CameraSource 则覆盖它
    virtual bool is_live() const
    {
        return false;
    }
};

} // namespace edge_ai
