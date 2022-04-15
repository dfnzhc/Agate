//
// Created by 秋鱼头 on 2022/4/11.
//

#include <Agate/Util/Utils.h>
#include "Agate/Core/Aplication.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace Agate {

Application::Application(const AppProps& props) :
    AgateWindow{props.windowExtend.x, props.windowExtend.y, props.title},
    optix_renderer_{std::make_unique<OptixRenderer>()}
{
    Resize(props.windowExtend);
    glGenTextures(1, &fb_texture_);
}

void Application::Render()
{
    optix_renderer_->Render();
}

void Application::Draw()
{
    optix_renderer_->DownloadPixels(pixels_.data());

    glBindTexture(GL_TEXTURE_2D, fb_texture_);
    GLenum texFormat = GL_RGBA;
    GLenum texelType = GL_UNSIGNED_BYTE;
    glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fb_size_.x, fb_size_.y, 0, GL_RGBA,
                 texelType, pixels_.data());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fb_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, fb_size_.x, fb_size_.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float) fb_size_.x, 0.f, (float) fb_size_.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);

        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float) fb_size_.y, 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float) fb_size_.x, (float) fb_size_.y, 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float) fb_size_.x, 0.f, 0.f);
    }
    glEnd();
}

void Application::Resize(const int2& newSize)
{
    fb_size_ = newSize;

    optix_renderer_->Resize(newSize);
    pixels_.resize(GetSize(newSize));
}

} // namespace Agate