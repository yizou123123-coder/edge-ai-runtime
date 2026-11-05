
#include <edge_ai/blocking_queue.hpp>

#include <iostream>
#include <stdexcept>

using edge_ai::BlockingQueue;
using edge_ai::QueuePolicy;

void require(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void test_drop_oldest()
{
    BlockingQueue<int> queue(3, QueuePolicy::DropOldest);

    require(queue.push(1), "Push 1 failed");
    require(queue.push(2), "Push 2 failed");
    require(queue.push(3), "Push 3 failed");

    require(queue.size() == 3, "Initial size mismatch");
    require(queue.dropped_count() == 0, "Unexpected drop");

    // 容量为 3，插入 4 后应删除 1
    require(queue.push(4), "Push 4 failed");

    require(queue.size() == 3, "Queue exceeded capacity");
    require(queue.peak_size() == 3, "Peak size mismatch");
    require(queue.dropped_count() == 1, "Drop count mismatch");

    // 再插入 5，应删除 2
    require(queue.push(5), "Push 5 failed");

    require(queue.dropped_count() == 2, "Second drop not counted");

    // 当前队列应该是 [3,4,5]
    queue.close();

    int value = 0;

    require(queue.pop(value) && value == 3, "Expected 3");
    require(queue.pop(value) && value == 4, "Expected 4");
    require(queue.pop(value) && value == 5, "Expected 5");

    require(!queue.pop(value), "Pop should fail after drain");

    std::cout << "[PASS] DropOldest FIFO and drop count\n";
}

void test_close_full_queue()
{
    BlockingQueue<int> queue(2, QueuePolicy::DropOldest);

    require(queue.push(10), "Push 10 failed");
    require(queue.push(20), "Push 20 failed");

    queue.close();

    // 已关闭的满队列不能删除旧元素
    require(!queue.push(30), "Push succeeded after close");

    require(queue.size() == 2, "Closed queue was modified");
    require(queue.dropped_count() == 0, "Closed queue dropped data");

    int value = 0;

    require(queue.pop(value) && value == 10, "Expected 10");
    require(queue.pop(value) && value == 20, "Expected 20");
    require(!queue.pop(value), "Expected closed and empty");

    std::cout << "[PASS] Close preserves pending frames\n";
}

void test_zero_capacity()
{
    bool threw = false;

    try
    {
        BlockingQueue<int> queue(0, QueuePolicy::DropOldest);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }

    require(threw, "Zero capacity should be rejected");

    std::cout << "[PASS] Zero capacity rejected\n";
}

int main()
{
    try
    {
        test_drop_oldest();
        test_close_full_queue();
        test_zero_capacity();

        std::cout << "DropOldest tests: ALL PASS\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
