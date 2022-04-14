//
// Created by 秋鱼头 on 2022/4/10.
//
#include <Agate/Core/Window.h>
#include <Agate/Core/Error.h>
#include <Agate/Core/OptixRenderer.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

int main()
{
    Agate::Log::Init();

    Agate::AgateWindow window{960, 720};

    Agate::OptixRenderer optix_renderer;
    int2 fbSize{1200, 1024};
    optix_renderer.Resize(fbSize);
    optix_renderer.Draw();

    std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
    optix_renderer.DownloadPixels(pixels.data());

    const std::string fileName = "hello.png";
    stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
                   pixels.data(), fbSize.x * sizeof(uint32_t));

    window.Run();

    Agate::Log::Shutdown();
}
