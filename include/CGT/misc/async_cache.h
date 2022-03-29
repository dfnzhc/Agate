//
// Created by 秋鱼头 on 2022/3/29.
//

#pragma once

#include "async.h"
namespace CGT {

// 这是一个多线程的着色器缓存。工作原理如下：
//
// 每个着色器的编译都由一个应用（主）线程使用 Async 类来调用，这个类会在一个新的线程中执行着色器的编译。

// 当多个线程试图编译同一个着色器时，会发生以下情况。
// 1）最先来的线程可以编译该着色器
// 2）其余的线程（假设是n个）进入 "等待 "模式。
// 3）由于n个核心现在是空闲的，n个线程被恢复/生成，继续执行其他任务。
//
// 这样一来，所有的核心都应该有大量的工作，线程上下文切换应该是最小的。
#define CACHE_ENABLE

template<typename T>
class Cache
{
public:
    struct CacheEntry
    {
        Sync m_Sync;
        T m_data;
    };
    typedef std::map<size_t, CacheEntry> DatabaseType;

private:
    DatabaseType m_database;
    std::mutex m_mutex;

public:
    bool CacheMiss(size_t hash, T* pOut)
    {
#ifdef CACHE_ENABLE
        typename DatabaseType::iterator it;

        // 查找着色器是否在缓存中，创建一个空条目，让其他线程知道这个线程将编译该着色器。
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            it = m_database.find(hash);

            // 没有找到 shader 的缓存，我们需要编译 shader!
            if (it == m_database.end()) {
                // 在这个过程中，其他请求该着色器的线程就知道有一个编译正在进行，他们需要等待该线程完成。
                m_database[hash].m_Sync.Inc();
                return true;
            }
        }

        // 如果记录过该着色器，那么：
        {
            // 如果有一个线程正在尝试编译这个着色器，那么就等待该线程完成。
            if (it->second.m_Sync.Get() == 1) {
                Async::Wait(&it->second.m_Sync);
            }

            // 返回着色器的编译数据
            *pOut = it->second.m_data;

            return false;
        }
#endif
        return true;
    }

    void UpdateCache(size_t hash, T* pValue)
    {
#ifdef CACHE_ENABLE
        typename DatabaseType::iterator it;

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            it = m_database.find(hash);
            assert(it != m_database.end());
        }
        it->second.m_data = *pValue;

        // 着色器已经被编译了，把sync设置为0，表示它已经被编译
        // 这也将唤醒所有等待Async::Wait(&it->second.m_Sync)的线程。
        it->second.m_Sync.Dec();
#endif
    }

    template<typename Func>
    void ForEach(Func func)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto it = m_database.begin(); it != m_database.end(); ++it) {
            func(it);
        }
    }
};

} // namespace CGT
