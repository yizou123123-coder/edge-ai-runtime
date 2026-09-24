#include <edge_ai/video_source.hpp>

namespace edge_ai
{
VideoSource::VideoSource(const std::string& video_path)
    :video_path_(video_path),
     capture_(video_path)
{
}

bool VideoSource::is_open() const{
    return capture_.isOpened();
}

bool VideoSource::read(cv::Mat& frame){
    return capture_.read(frame);
}

double VideoSource::fps() const{
    return capture_.get(cv::CAP_PROP_FPS);
}

const VideoSource::std::string& path() const{
    return video_path_;
}
} // namespace edge_ai