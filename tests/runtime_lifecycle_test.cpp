#include <edge_ai/runtime.hpp>

#include <iostream>

int main()
{
    const std::string video_path = "/home/yizou/datasets/edge-ai-runtime/test.mp4";
    edge_ai::Runtime runtime(video_path, 30);

    const bool first_start = runtime.start(); //first_start为true
    const bool second_start = runtime.start(); // second_start为false

    if (!first_start)
    {
        std::cerr << "One-shot test failed: First start should succeed!\n";
        return 1;
    }

    if (second_start)
    {
        std::cerr << "One-shot test failed: Second start should be rejected!\n";
        return 1;
    }

    runtime.stop();
    runtime.join();

    const int exit_code = runtime.exit_code();
    if (exit_code != 5)
    {
        std::cerr << "One-shot test failed: Expected exit_code 0, but got: " << exit_code << "\n";
        return 1;
    }
    
    std::cout << "Runtime one-shot test: PASS\n";
}