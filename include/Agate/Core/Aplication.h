//
// Created by 秋鱼头 on 2022/4/11.
//

#pragma once

#include <Agate/Util/Scene.hpp>
#include <Agate/Util/MouseTracker.hpp>
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
    int2 fb_size_;
    GLDisplay display_;
    CudaOutputBuffer<uchar4> output_buffer_;
    std::unique_ptr<OptixRenderer> optix_renderer_;
    
    std::shared_ptr<ModelData> model_;
    std::shared_ptr<Scene> scene_;
    
    std::vector<std::string> optix_modules_;
    
    std::shared_ptr<Camera> camera_;
    std::shared_ptr<MouseTracker> tracker_;
    
    void createOptixState();
    void loadAssets();
    
    void Render() override;
    void Draw() override;
    
    void Resize(const int2& newSize) override;
    void cursorUpdate() override;
public:
    explicit Application(const AppProps& props);
    ~Application() = default;

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    
    void addOptixModule(std::string_view module) { optix_modules_.push_back(module.data()); }
    void finalize();
};

} // namespace Agate