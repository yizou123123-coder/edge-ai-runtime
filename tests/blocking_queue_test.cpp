#include <edge_ai/blocking_queue.hpp>

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main()
{
    edge_ai::BlockingQueue<int> queue(2);

    // 消费者用来记录实际接收到数据的动态数组
    std::vector<int> received;

    std::thread producer(
        // 引用捕获：让子线程直接操作主线程中创建的同一个 queue 实体，避免在线程间发生对象的拷贝
        [&queue]
        {
            for (int value = 1; value <= 5; ++value)
            {
                if (!queue.push(value))
                {
                    return;
                }

                std::cout << "[Producer] push " << value << '\n';
            }

            queue.close();
        });

    std::thread consumer(
        [&queue, &received]
        {
            int value = 0;

            while (queue.pop(value))
            {
                received.push_back(value);

                std::cout << "[Consumer] pop " << value << '\n';

                // 不管发生什么，我都睡固定时间 100ms
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

    // 主线程等待两个子线程运行结束
    producer.join();
    consumer.join();

    const std::vector<int> expected{1, 2, 3, 4, 5};

    if (received != expected)
    {
        std::cerr << "BlockingQueue test: FAIL\n";

        return 1;
    }

    std::cout << "BlockingQueue producer-consumer test: PASS\n";

    return 0;
}