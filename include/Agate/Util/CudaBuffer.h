//
// Created by 秋鱼头 on 2022/4/12.
//

#pragma once

#include <optix_types.h>
#include <vector>
#include <cassert>
#include <cuda.h>
#include "Agate/Core/Error.h"
namespace Agate {

struct CudaBuffer
{
    size_t size_in_bytes{0};
    void* dev_ptr{nullptr};

    [[nodiscard]]
    inline CUdeviceptr device_ptr() const { return (CUdeviceptr) dev_ptr; }

    void Resize(size_t size)
    {
        if (dev_ptr) Free();
        Alloc(size);
    }

    void Alloc(size_t size)
    {
        assert(d_ptr == nullptr);
        this->size_in_bytes = size;
        CUDA_CHECK(cudaMalloc((void**) &dev_ptr, size_in_bytes));
    }

    void Free()
    {
        cudaFree(dev_ptr);
        dev_ptr = nullptr;
        size_in_bytes = 0;
    }

    template<typename T>
    void Upload(const T* t, size_t count)
    {
        assert(dev_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(dev_ptr, (void*) t, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void Alloc_And_Upload(const std::vector<T>& vt)
    {
        Alloc(vt.size() * sizeof(T));
        Upload((const T*) vt.data(), vt.size());
    }

    template<typename T>
    void Download(T* t, size_t count)
    {
        assert(dev_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy((void*) t, dev_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

} // namespace Agate