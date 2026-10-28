#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>
#include <algorithm>
#include <stdexcept>

namespace edge_ai
{

enum class QueuePolicy
{
    Blocking,
    DropOldest,
    LatestFrame
};

// 这个 Queue 不提前规定里面一定装什么类型，T=int/float/double/Mat... T 就是一个“占位符”，根据对象创建的不同改变
template <typename T>
class BlockingQueue
{
public:
    explicit BlockingQueue(std::size_t capacity, QueuePolicy policy = QueuePolicy::Blocking)
        : capacity_(capacity),
          policy_(policy)
    {
        if (capacity_ == 0)
        {
            throw std::invalid_argument("Queue capacity must be greater than zero");
        }
    }

    bool push(T value)
    {
        // 创造 lock 这个对象，把 mutex_ 传进去初始化它，lock 利用 RAII 机制管理 mutex
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (policy_ == QueuePolicy::Blocking)
        {
            // 如果队列现在已经关闭，或者队列还有空间，我就继续；没有空间则producer等待
            // 当 Lambda 返回 false，线程就乖乖在原地睡觉；当 Lambda 返回 true 时，线程才会停止等待、被放行
            not_full_.wait(
                lock,
                // 放行条件
                [this]
                {
                    return closed_ || queue_.size() < capacity_;
                }
            );
        }

        // 确认队列是否被关闭
        if (closed_)
        {
            return false;
        }

        if (policy_ == QueuePolicy::DropOldest && queue_.size() == capacity_)
        {
            queue_.pop();
            dropped_count_++;
        }
        else if (policy_ == QueuePolicy::LatestFrame)
        {
            while(!queue_.empty())
            {
                queue_.pop();
                dropped_count_++;
            }
        }
        
        queue_.push(std::move(value));
        peak_size_ = std::max(peak_size_, queue_.size());

        lock.unlock();

        // 解锁后唤醒消费者
        not_empty_.notify_one();

        return true;
    }

    bool pop(T& value)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        // 不空则继续，空则等待
        not_empty_.wait(
            lock,
            [this]
            {
                return closed_ || !queue_.empty();
            }
        );

        if (queue_.empty())
        {
            return false;
        }
        
        value = std::move(queue_.front());
        queue_.pop();

        lock.unlock();
        not_full_.notify_one();
        return true;
    }

    // 查询丢失的帧数
    std::size_t dropped_count() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_count_;
    }

    void close()
    {
        // 先上锁 再解锁后通知 {}结束后自动释放锁
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }

        // Queue 已关闭，所有在等数据和等空间的人都起来重新检查状态
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    std::size_t size() const
    {
        // 构造时加锁，析构时自动解锁，期间无法手动解锁
        std::lock_guard<std::mutex> lock(mutex_);
        const auto n = queue_.size();
        return n;
    }

    std::size_t peak_size() const 
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_size_;
    }

    // 队列有元素：立即取出，返回 true。
    // 队列为空：立即返回 false，不等待。
    // 队列已经关闭但还有元素：仍然允许取出剩余元素。
    bool try_pop(T& value)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.empty())
        {
            return false;
        }

        value = std::move(queue_.front());

        queue_.pop();

        lock.unlock();
        not_full_.notify_one();

        return true;
    }


private:
    std::queue<T> queue_;

    std::size_t capacity_;

    // 允许mutex_即使在 const成员函数中，也能够被修改（即能够被加锁和解锁）
    mutable std::mutex mutex_;

    // 创造了两个条件
    std::condition_variable not_empty_;
    std::condition_variable not_full_;

    bool closed_{false};

    std::size_t peak_size_{0};

    QueuePolicy policy_;
    std::size_t dropped_count_{0};
};

}  // namespace edge_ai