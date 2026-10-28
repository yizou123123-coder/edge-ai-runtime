
#include <edge_ai/blocking_queue.hpp>

#include <iostream>
#include <stdexcept>

using edge_ai::BlockingQueue;
using edge_ai::QueuePolicy;

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

// 连续推入新帧，旧的待处理帧应被替换
void test_latest_frame_replacement()
{
    BlockingQueue<int> queue(4, QueuePolicy::LatestFrame);

    require(queue.push(1), "Push 1 failed");
    require(queue.push(2), "Push 2 failed");
    require(queue.push(3), "Push 3 failed");
    require(queue.push(4), "Push 4 failed");

    // 应该只剩最新元素 4
    require(queue.size() == 1, "Queue size should be 1");
    require(queue.peak_size() == 1, "Peak size should be 1");
    require(queue.dropped_count() == 3, "Expected 3 dropped frames");

    int value = 0;

    require(queue.pop(value), "Pop failed");
    require(value == 4, "Expected latest frame 4");

    // 队列已空，再插入不会产生新的丢帧
    require(queue.push(5), "Push 5 failed");

    require(queue.size() == 1, "Expected one pending frame");
    require(queue.dropped_count() == 3, "Unexpected extra drop");

    queue.close();

    require(queue.pop(value) && value == 5, "Expected frame 5");
    require(!queue.pop(value), "Pop should fail after drain");

    std::cout << "[PASS] LatestFrame replacement\n";
}

// 关闭后不能删除旧帧，也不能接收新帧
void test_close_preserves_latest_frame()
{
    BlockingQueue<int> queue(4, QueuePolicy::LatestFrame);

    require(queue.push(10), "Push 10 failed");

    queue.close();

    require(!queue.push(20), "Push succeeded after close");
    require(queue.size() == 1, "Closed queue modified");
    require(queue.dropped_count() == 0, "Unexpected dropped frame");

    int value = 0;

    require(queue.pop(value) && value == 10, "Expected frame 10");
    require(!queue.pop(value), "Expected closed and empty queue");

    std::cout << "[PASS] Close preserves latest frame\n";
}

int main()
{
    try
    {
        test_latest_frame_replacement();
        test_close_preserves_latest_frame();

        std::cout << "LatestFrame tests: ALL PASS\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
