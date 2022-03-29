//
// Created by 秋鱼头 on 2022/3/29.
//

#pragma once

namespace CGT {

// 这是一个简陋的多线程类，工作原理如下：
//
// 每个任务（task）都是由应用程序（主）线程使用 Async 类调用的。比如，在新的线程上执行着色器编译。
// 为了防止上下文切换，我们需要将运行的线程数量限制在内核数量上。
// 这是通过全局计数器来完成的，它跟踪运行线程的数量。
// 当线程正在运行一个任务时，这个计数器被递增，当它结束时被递减。
// 当一个线程进入等待模式时，它也会被减去，而当一个线程被发出信号并且有一个核心可以恢复该线程时，它就会被增加。
// 如果所有内核都很忙，应用程序的线程就会进入等待状态，以防止它产生更多的线程。

class Sync
{
    int count_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
public:
    int Inc()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        count_++;
        return count_;
    }

    int Dec()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        count_--;
        if (count_ == 0)
            condition_.notify_all();
        return count_;
    }

    int Get()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return count_;
    }

    void Reset()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        count_ = 0;
        condition_.notify_all();
    }

    void Wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (count_ != 0)
            condition_.wait(lock);
    }

};

class Async
{
    static int Active_Threads_;
    static int Max_Threads_;
    static std::mutex Mutex_;
    static std::condition_variable Condition_;
    static bool bExiting_;

    std::function<void()> job_;
    Sync* sync_;
    std::thread* thread_;

public:
    explicit Async(std::function<void()> job, Sync* pSync = nullptr);
    ~Async();
    static void Wait(Sync* pSync);
};

class AsyncPool
{
    std::vector<Async*> pool_;
public:
    ~AsyncPool();
    void Flush();
    void AddAsyncTask(std::function<void()> job, Sync* pSync = nullptr);
};

void ExecAsyncIfThereIsAPool(AsyncPool* pAsyncPool, std::function<void()> job);
} // namespace CGT

 