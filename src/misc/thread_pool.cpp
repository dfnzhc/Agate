//
// Created by 秋鱼头 on 2022/3/29.
//

#include "CGT/misc/thread_pool.h"

#include <utility>

namespace CGT {

static ThreadPool g_threadPool;

ThreadPool* GetThreadPool()
{
    return &g_threadPool;
}

#ifndef ENABLE_MULTI_THREADING
#define ENABLE_MULTI_THREADING
#endif

ThreadPool::ThreadPool()
{
    active_threads_ = 0;

#ifdef ENABLE_MULTI_THREADING
    num_threads_ = std::thread::hardware_concurrency();
    bExiting_ = false;
    for (int ii = 0; ii < num_threads_; ii++) {
        pool_.emplace_back(&ThreadPool::JobStealerLoop, GetThreadPool());
    }
#endif
}

ThreadPool::~ThreadPool()
{
#ifdef ENABLE_MULTI_THREADING
    bExiting_ = true;
    condition_.notify_all();
    for (int ii = 0; ii < num_threads_; ii++) {
        pool_[ii].join();
    }
#endif
}

void ThreadPool::JobStealerLoop()
{
#ifdef ENABLE_MULTI_THREADING
    while (true) {
        Task t;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            condition_.wait(lock, [this] { return bExiting_ || (!deque_.empty() && (active_threads_ < num_threads_)); });
            if (bExiting_)
                return;

            active_threads_++;

            t = deque_.front();
            deque_.pop_front();
        }

        t.job();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            active_threads_--;
        }
    }
#endif
}

void ThreadPool::AddJob(std::function<void()> job)
{
#ifdef ENABLE_MULTI_THREADING
    if (!bExiting_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        Task t;
        t.job = std::move(job);

        deque_.push_back(t);

        if (active_threads_ < num_threads_)
            condition_.notify_one();
    }
#else
    job();
#endif
}

} // namespace CGT