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
    size_t byte_size{0};
    void* dev_ptr{nullptr};

    [[nodiscard]]
    inline CUdeviceptr get() const { return (CUdeviceptr) dev_ptr; }

    void resize(size_t size)
    {
        if (dev_ptr) free();
        alloc(size);
    }

    void alloc(size_t size)
    {
        assert(d_ptr == nullptr);
        this->byte_size = size;
        CUDA_CHECK(cudaMalloc((void**) &dev_ptr, byte_size));
    }

    void free()
    {
        cudaFree(dev_ptr);
        dev_ptr = nullptr;
        byte_size = 0;
    }

    template<typename T>
    void upload(const T* t, size_t count)
    {
        assert(dev_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(dev_ptr, (void*) t, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void allocAndUpload(const std::vector<T>& vt)
    {
        alloc(vt.size() * sizeof(T));
        upload((const T*) vt.data(), vt.size());
    }

    template<typename T>
    void download(T* t, size_t count)
    {
        assert(dev_ptr != nullptr);
        assert(sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy((void*) t, dev_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

} // namespace Agate