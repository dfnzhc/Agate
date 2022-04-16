//
// Created by 秋鱼头 on 2022/4/16.
//

#pragma once

#include "Agate/Core/Error.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <sstream>
#include <vector>

namespace Agate {

template<typename PIXEL_FORMAT>
class CudaOutputBuffer
{
public:
    CudaOutputBuffer(int32_t width, int32_t height);
    ~CudaOutputBuffer();

    void SetStream(CUstream stream) { stream_ = stream; }

    void Resize(int32_t width, int32_t height);

    // Allocate or update device pointer as necessary for CUDA access
    PIXEL_FORMAT* Map();
    void Unmap();

    int32_t Width() const { return width_; }
    int32_t Height() const { return height_; }

    // Get output buffer
    GLuint GetPbo();
    PIXEL_FORMAT* GetHostPointer();
private:
    int32_t width_ = 0u;
    int32_t height_ = 0u;

    cudaGraphicsResource* cuda_gfx_resource_ = nullptr;
    GLuint pbo_ = 0u;
    PIXEL_FORMAT* device_pixels_ = nullptr;
    std::vector<PIXEL_FORMAT> host_pixels_;

    CUstream stream_ = nullptr;
};

template<typename PIXEL_FORMAT>
CudaOutputBuffer<PIXEL_FORMAT>::CudaOutputBuffer(int32_t width, int32_t height)
{
    AGATE_ASSERT(width > 1 && height > 1);

    // If using GL Interop, expect that the active device is also the display device.
    int current_device, is_display_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout, current_device));
    if (!is_display_device) {
        throw AgateException(
            "GL interop is only available on display device, please use Display device for optimal "
            "performance.  Alternatively you can disable GL interop with --no-gl-interop and run with "
            "degraded performance."
        );
    }
    Resize(width, height);
}

template<typename PIXEL_FORMAT>
CudaOutputBuffer<PIXEL_FORMAT>::~CudaOutputBuffer()
{
    try {
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CHECK(glDeleteBuffers(1, &pbo_));
    }
    catch (std::exception& e) {
        LOG_ERROR("CUDAOutputBuffer destructor caught exception: {}", e.what());
    }
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::Resize(int32_t width, int32_t height)
{
    AGATE_ASSERT(width > 1 && height > 1);

    if (width_ == width && height_ == height)
        return;

    width_ = width;
    height_ = height;

    // GL buffer gets resized below
    GL_CHECK(glGenBuffers(1, &pbo_));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo_));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * width_ * height_, nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &cuda_gfx_resource_,
        pbo_,
        cudaGraphicsMapFlagsWriteDiscard
    ));

    if (!host_pixels_.empty())
        host_pixels_.resize(width_ * height_);
}

template<typename PIXEL_FORMAT>
PIXEL_FORMAT* CudaOutputBuffer<PIXEL_FORMAT>::Map()
{
    size_t buffer_size = 0u;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_gfx_resource_, stream_));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>( &device_pixels_ ),
        &buffer_size,
        cuda_gfx_resource_
    ));

    return device_pixels_;
}

template<typename PIXEL_FORMAT>
void CudaOutputBuffer<PIXEL_FORMAT>::Unmap()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_gfx_resource_, stream_));
}

template<typename PIXEL_FORMAT>
GLuint CudaOutputBuffer<PIXEL_FORMAT>::GetPbo()
{
    if (pbo_ == 0u)
        GL_CHECK(glGenBuffers(1, &pbo_));

    return pbo_;
}

template<typename PIXEL_FORMAT>
PIXEL_FORMAT* CudaOutputBuffer<PIXEL_FORMAT>::GetHostPointer()
{
    host_pixels_.resize(width_ * height_);

    CUDA_CHECK(cudaMemcpy(
        static_cast<void*>( host_pixels_.data()),
        Map(),
        width_ * height_ * sizeof(PIXEL_FORMAT),
        cudaMemcpyDeviceToHost
    ));
    Unmap();

    return host_pixels_.data();
}

} // namespace Agate