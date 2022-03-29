//
// Created by 秋鱼头 on 2022/3/29.
//

#include "CGT/misc/async.h"

#include <utility>

namespace CGT {

//
// Some static functions
//
int Async::Active_Threads_ = 0;
std::mutex Async::Mutex_;
std::condition_variable Async::Condition_;
bool Async::bExiting_ = false;
int Async::Max_Threads_ = std::thread::hardware_concurrency();

Async::Async(std::function<void()> job, Sync* pSync) :
    job_{std::move(job)},
    sync_{pSync}
{
    if (sync_)
        sync_->Inc();

    {
        std::unique_lock<std::mutex> lock(Mutex_);

        while (Active_Threads_ >= Max_Threads_) {
            Condition_.wait(lock);
        }

        Active_Threads_++;
    }

    thread_ = new std::thread([this]()
                                {
                                    job_();

                                    {
                                        std::lock_guard<std::mutex> lock(Mutex_);
                                        Active_Threads_--;
                                    }

                                    Condition_.notify_one();

                                    if (sync_)
                                        sync_->Dec();
                                });
}

Async::~Async()
{
    thread_->join();
    delete thread_;
}

void Async::Wait(Sync* pSync)
{
    if (pSync->Get() == 0)
        return;

    {
        std::lock_guard<std::mutex> lock(Mutex_);
        Active_Threads_--;
    }

    Condition_.notify_one();

    pSync->Wait();

    {
        std::unique_lock<std::mutex> lock(Mutex_);

        Condition_.wait(lock, []
        {
            return bExiting_ || (Active_Threads_ < Max_Threads_);
        });

        Active_Threads_++;
    }
}

//
// Basic async pool
//
AsyncPool::~AsyncPool()
{
    Flush();
}

void AsyncPool::Flush()
{
    for (auto & i : pool_)
        delete i;
    pool_.clear();
}

void AsyncPool::AddAsyncTask(std::function<void()> job, Sync* pSync)
{
    pool_.push_back(new Async(std::move(job), pSync));
}

// ExecAsyncIfThereIsAPool, 如果存在一个线程池，将使用异步，否则使用同步运行任务
void ExecAsyncIfThereIsAPool(AsyncPool* pAsyncPool, std::function<void()> job)
{
    // use MT if there is a pool
    if (pAsyncPool != nullptr) {
        pAsyncPool->AddAsyncTask(job);
    } else {
        job();
    }
}

} // namespace CGT