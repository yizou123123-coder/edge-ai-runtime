
#include <edge_ai/blocking_queue.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using edge_ai::BlockingQueue;
using edge_ai::QueuePolicy;

void run_test(QueuePolicy policy, const char* name)
{
    constexpr std::size_t kCapacity = 4;
    constexpr int kFramesPerProducer = 100;

    BlockingQueue<int> queue(kCapacity, policy);

    std::atomic<std::size_t> accepted{0};
    std::atomic<std::size_t> consumed{0};

    // 消费者：故意放慢消费速度
    std::thread consumer([&]
    {
        int value = 0;

        while (queue.pop(value))
        {
            ++consumed;

            std::this_thread::sleep_for(
                std::chrono::microseconds(500)
            );
        }
    });

    // 生产者任务
    auto produce = [&](int start_id)
    {
        for (int i = 0; i < kFramesPerProducer; ++i)
        {
            if (queue.push(start_id + i))
            {
                ++accepted;
            }
        }
    };

    std::thread producer1(produce, 0);
    std::thread producer2(produce, 1000);

    // 等待两个生产者结束
    producer1.join();
    producer2.join();

    // 关闭队列，让消费者排空后退出
    queue.close();
    consumer.join();

    const auto a = accepted.load();
    const auto c = consumed.load();
    const auto d = queue.dropped_count();

    if (a != c + d)
    {
        throw std::runtime_error(
            "Accepted != Consumed + Dropped"
        );
    }

    if (queue.size() != 0 ||
        queue.peak_size() > kCapacity)
    {
        throw std::runtime_error(
            "Queue size invariant failed"
        );
    }

    if (policy == QueuePolicy::Blocking && d != 0)
    {
        throw std::runtime_error(
            "Blocking policy unexpectedly dropped frames"
        );
    }

    if (policy == QueuePolicy::LatestFrame &&
        queue.peak_size() > 1)
    {
        throw std::runtime_error(
            "LatestFrame kept multiple pending frames"
        );
    }

    std::cout
        << "[PASS] " << name
        << " | accepted=" << a
        << " consumed=" << c
        << " dropped=" << d
        << " peak=" << queue.peak_size()
        << '\n';
}

int main()
{
    try
    {
        run_test(QueuePolicy::Blocking, "Blocking");
        run_test(QueuePolicy::DropOldest, "DropOldest");
        run_test(QueuePolicy::LatestFrame, "LatestFrame");

        std::cout << "Queue concurrency: ALL PASS\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
