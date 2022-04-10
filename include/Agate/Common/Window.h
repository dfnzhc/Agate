//
// Created by 秋鱼头 on 2022/4/10.
//

#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

namespace Agate {


class AgateWindow
{
public:
    AgateWindow(int w, int h, std::string_view title = "The Agate Renderer");
    ~AgateWindow();

    AgateWindow(const AgateWindow&) = delete;
    AgateWindow& operator=(const AgateWindow&) = delete;

    /// put the pixels on the screen
    void Draw();

    /// re-render the frame - typically part of draw(), but we keep
    /// this a separate function so render() can focus on optix rendering
    void Render();

    /// open the window and runs the window's events
    void Run();

    void Resize(const float2& newSize);

    bool ShouldClose() { return glfwWindowShouldClose(handle_); }
    bool WasWindowResized() const { return window_resized; }
private:
    
    void InitWindow();
    void SetGLFWCallback();
    
    GLFWwindow* handle_{nullptr};
    
    bool window_resized = false;
    
    int width_, height_;
    std::string title_;
};

} // namespace Agate
