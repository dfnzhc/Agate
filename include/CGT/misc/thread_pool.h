//
// Created by 秋鱼头 on 2022/3/29.
//

#pragma once

namespace CGT {

struct Task
{
    std::function<void()> job;
    std::vector<Task *> child_tasks;
};

class ThreadPool
{
public:
    ThreadPool();
    ~ThreadPool();
    void JobStealerLoop();
    void AddJob(std::function<void()> New_Job);
private:
    bool bExiting_;
    int num_threads_;
    int active_threads_ = 0;
    std::vector<std::thread> pool_;
    std::deque<Task> deque_;
    std::condition_variable condition_;
    std::mutex queue_mutex_;
};

} // namespace CGT

 