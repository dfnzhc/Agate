//
// Created by 秋鱼头 on 2022/4/11.
//

#include <Agate/Util/Utils.h>
#include <Agate/Util/ReadFile.h>
#include "Agate/Core/Aplication.h"

namespace Agate {

Application::Application(const AppProps& props) :
    AgateWindow{props.windowExtend.x, props.windowExtend.y, props.title},
    optix_renderer_{std::make_unique<OptixRenderer>()},
    display_{BufferImageFormat::UNSIGNED_BYTE4},
    output_buffer_{width_, height_},
    model_(std::make_shared<ModelData>()),
    camera_(std::make_shared<Camera>()),
    tracker_(std::make_shared<MouseTracker>())
{
    fb_size_ = props.windowExtend;

    output_buffer_.Resize(fb_size_.x, fb_size_.y);
    uchar4* result_buffer_data = output_buffer_.Map();
    optix_renderer_->Resize(fb_size_, result_buffer_data);
    output_buffer_.Unmap();

    scene_ = std::make_shared<Scene>(optix_renderer_->getContext());
}

void Application::Render()
{
    optix_renderer_->updateCamera(tracker_->getCamera());
    optix_renderer_->bind("Hello");
    optix_renderer_->Render();
}

void Application::Draw()
{
    display_.Display(width_, height_, fb_size_.x, fb_size_.y, output_buffer_.GetPbo());
}

void Application::Resize(const int2& newSize)
{
    fb_size_ = newSize;

    output_buffer_.Resize(newSize.x, newSize.y);
    optix_renderer_->Resize(newSize, output_buffer_.Map());
}

void Application::finalize()
{
    createOptixState();
    loadAssets();

    camera_ = std::make_shared<Camera>(model_->camera());
    tracker_->setCamera(camera_.get());

    scene_->finalize(1);
}

void Application::createOptixState()
{
    {
        OptixStateInfo info;
        info.ptx_name = "Hello";
        info.raygen = "__raygen__Hello";
        info.miss = "__miss__Hello";
        info.closesthit = "__closesthit__Hello";
        info.anyhit = "__anyhit__Hello";
        info.hitgroup = "__hitgroup__Hello";

        optix_renderer_->finalize(info);
        optix_renderer_->createSBT(info);
    }
}

void Application::loadAssets()
{
    {
        model_->addMeshFromGLTF(GetAssetPath("models/rubber_duck/scene.gltf"));
    }

    model_->finalize();
    scene_->addMeshData(model_);
}

void Application::cursorUpdate()
{
    if (input_.mouse_button != -1) {
        if (input_.mouse_button == GLFW_MOUSE_BUTTON_LEFT) {
            tracker_->setViewMode(ViewMode::Orbit);
        }
        else if (input_.mouse_button == GLFW_MOUSE_BUTTON_RIGHT) {
            tracker_->setViewMode(ViewMode::EyeFixed);
        }

        tracker_->update(input_.posX, input_.posY);
    }
    
    if (input_.scroll != 0) {
        tracker_->zoom(input_.scroll);
        
        input_.scroll = 0;
    }
}

} // namespace Agate