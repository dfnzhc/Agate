//
// Created by 秋鱼头 on 2022/4/11.
//

#pragma once

#include "Window.h"
#include "OptixRenderer.h"
#include "GLDisplay.h"

#include "Agate/Util/CudaOutputBuffer.hpp"

namespace Agate {

struct AppProps
{
    int2 windowExtend{960, 720};
    std::string title{"The Agate Renderer"};
};

class Application : public AgateWindow
{
    std::unique_ptr<OptixRenderer> optix_renderer_;
    
    std::vector<uint32_t> pixels_;
    
    int2 fb_size_;
    GLuint fb_texture_{0};
    
    GLDisplay display_;
    GLuint pbo_{0};
    
    CudaOutputBuffer<uchar4> output_buffer_;
    
    void Render() override;
    void Draw() override;
    
    void Resize(const int2& newSize) override;
public:
    explicit Application(const AppProps& props);
    ~Application() = default;

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
};

} // namespace Agate