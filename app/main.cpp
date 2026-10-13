#include <edge_ai/runtime.hpp>

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc != 3 || std::string(argv[1]) != "--video")
    {
        std::cerr << "Usage: " << argv[0] << " --video <video_path>\n";
        return 1;
    }

    const std::string video_path = argv[2];

    edge_ai::Runtime runtime(video_path);

    if (!runtime.start())
    {
        return runtime.exit_code();
    }

    // 主线程等待worker线程完成
    runtime.join();

    return runtime.exit_code();
}   