//
// Created by 秋鱼头 on 2022/4/11.
//

#include <Agate/Util/Utils.h>
#include "Agate/Core/Aplication.h"

//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace Agate {

Application::Application(const AppProps& props) :
    AgateWindow{props.windowExtend.x, props.windowExtend.y, props.title},
    optix_renderer_{std::make_unique<OptixRenderer>()},
    display_{BufferImageFormat::UNSIGNED_BYTE4},
    output_buffer_{width_, height_}
{
    glGenBuffers(1, &pbo_);

    fb_size_ = props.windowExtend;

    pixels_.resize(GetSize(fb_size_));

    output_buffer_.Resize(fb_size_.x, fb_size_.y);
    uchar4* result_buffer_data = output_buffer_.Map();
    optix_renderer_->Resize(fb_size_, result_buffer_data);
    output_buffer_.Unmap();

    glGenTextures(1, &fb_texture_);
}

void Application::Render()
{
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
    pixels_.resize(GetSize(newSize));
}

} // namespace Agate