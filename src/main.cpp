//
// Created by 秋鱼头 on 2022/4/10.
//
#include <Agate/Core/Window.h>
#include <Agate/Core/Error.h>
#include <Agate/Core/OptixRenderer.h>

int main()
{
    Agate::Log::Init();
    
    Agate::AgateWindow window{960, 720};

    Agate::OptixRenderer optix_renderer;

    window.Run();
    
    Agate::Log::Shutdown();
}
