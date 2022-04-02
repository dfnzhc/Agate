//
// Created by 秋鱼头 on 2022/4/2.
//

#pragma once

#include <CGT/common.h>
namespace CGT {

/// 这是典型的环形缓冲区，它被用于组织那些会被重复使用的资源。
/// 例如，命令列表和 "动态 "常量缓冲区，等。

class Ring
{
public:
    void Create(uint32_t TotalSize)
    {
        head_ = 0;
        allocated_size_ = 0;
        total_size_ = TotalSize;
    }

    uint32_t GetSize() { return allocated_size_; }
    uint32_t GetHead() { return head_; }
    uint32_t GetTail() { return (head_ + allocated_size_) % total_size_; }

    /// 帮助函数，让分配的块在环形数组尽量连续，避免分配块数据在头尾交叉
    uint32_t PaddingToAvoidCrossOver(uint32_t size)
    {
        int tail = GetTail();
        if ((tail + size) > total_size_)
            return (total_size_ - tail);
        else
            return 0;
    }

    bool Alloc(uint32_t size, uint32_t *pOut)
    {
        if (allocated_size_ + size <= total_size_)
        {
            if (pOut)
                *pOut = GetTail();

            allocated_size_ += size;
            return true;
        }

        throw CGTException("No available size.");
        return false;
    }

    bool Free(uint32_t size)
    {
        if (allocated_size_ >= size)
        {
            head_ = (head_ + size) % total_size_;
            allocated_size_ -= size;
            return true;
        }
        return false;
    }
private:
    uint32_t head_;
    uint32_t allocated_size_;
    uint32_t total_size_;
};

/// 这个类可以被认为是环形缓冲区中的环形缓冲区。
/// 外面的环是用来存放帧，而内部则是为该帧分配的资源。
/// 外环的大小通常是指背后缓冲区的数量。

/// 当外环已满时，对于下一次分配，它将自动释放最旧帧的条目，并使这些条目在下一帧可用。
/// 这发生在当你调用 'OnBeginFrame()' 时 

class RingWithTabs
{
public:

    void OnCreate(uint32_t numberOfBackBuffers, uint32_t memTotalSize)
    {
        back_buffer_index_ = 0;
        number_of_back_buffers_ = numberOfBackBuffers;

        /// 将每一帧追踪器的内存初始化
        mem_allocated_in_frame_ = 0;
        for (unsigned int & i : mem_per_back_buffer_)
            i = 0;

        mem_.Create(memTotalSize);
    }

    void OnDestroy()
    {
        mem_.Free(mem_.GetSize());
    }

    bool Alloc(uint32_t size, uint32_t *pOut)
    {
        uint32_t padding = mem_.PaddingToAvoidCrossOver(size);
        /// 先进行 padding，避免分配块数据在头尾交叉
        if (padding > 0)
        {
            mem_allocated_in_frame_ += padding;

            /// 没有多余内存，不能分配 padding
            if (mem_.Alloc(padding, NULL) == false)
            {
                return false;  
            }
        }

        if (mem_.Alloc(size, pOut) == true)
        {
            mem_allocated_in_frame_ += size;
            return true;
        }
        return false;
    }

    void OnBeginFrame()
    {
        mem_per_back_buffer_[back_buffer_index_] = mem_allocated_in_frame_;
        mem_allocated_in_frame_ = 0;

        back_buffer_index_ = (back_buffer_index_ + 1) % number_of_back_buffers_;

        /// 将最早缓冲区的所有条目进行释放
        uint32_t memToFree = mem_per_back_buffer_[back_buffer_index_];
        mem_.Free(memToFree);
    }
private:
    /// 内部环形缓冲区
    Ring mem_;

    /// 这是外部环形缓冲器 (I could have reused the Ring class though)
    uint32_t back_buffer_index_;
    uint32_t number_of_back_buffers_;

    uint32_t mem_allocated_in_frame_;
    uint32_t mem_per_back_buffer_[4];
};

} // namespace CGT