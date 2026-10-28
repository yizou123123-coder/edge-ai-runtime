#pragma once

#include <chrono>
#include <cstdint>
#include <opencv2/core.hpp>

namespace edge_ai
{

struct Frame
{
    std::uint64_t id{0};

    // {}：初始化：有默认构造函数就调默认构造函数；没有构造函数就直接把内存清零
    std::chrono::steady_clock::time_point capture_ts{};

    cv::Mat image;
};

} // namespace edge_ai