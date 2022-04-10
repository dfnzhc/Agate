//
// Created by 秋鱼头 on 2022/4/10.
//
#include <Agate/Common/Window.h>

int main()
{
    Agate::Log::Init();
    
    Agate::AgateWindow window{960, 720};

    LOG_TRACE("Trace")
    LOG_DEBUG("Debug")
    LOG_INFO("Debug")
    LOG_WARN("Debug")
    LOG_ERROR("Debug")

    window.Run();
    
    Agate::Log::Shutdown();
}
