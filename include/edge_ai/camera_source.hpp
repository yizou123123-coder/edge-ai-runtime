
#pragma once

#include <edge_ai/frame_source.hpp>
#include <opencv2/videoio.hpp>

namespace edge_ai
{

class CameraSource : public FrameSource
{
public:
    explicit CameraSource(int device_index = 0);

    bool is_open() const override;

    bool read(cv::Mat& frame) override;

    bool is_live() const override
    {
        return true;
    }

private:
    cv::VideoCapture capture_;

};

} // namespace edge_ai
