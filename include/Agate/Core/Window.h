//
// Created by 秋鱼头 on 2022/4/10.
//

#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include "Interfaces.h"

namespace Agate {

struct MouseInput
{
    int mouse_button = -1;
    int posX = 0;
    int posY = 0;
    int scroll = 0;
};

class AgateWindow
{
public:
    AgateWindow(int w, int h, std::string_view title = "The Agate Renderer");
    ~AgateWindow();

    AgateWindow(const AgateWindow&) = delete;
    AgateWindow& operator=(const AgateWindow&) = delete;

    /// open the window and runs the window's events
    void Run();

protected:
    /// put the pixels on the screen
    virtual void Draw() {}

    /// re-render the frame - typically part of draw(), but we keep
    /// this a separate function so render() can focus on optix rendering
    virtual void Render() {}

    virtual void Resize(const int2& newSize) {}
    virtual void cursorUpdate() {};

    bool ShouldClose() { return glfwWindowShouldClose(handle_); }
    bool WasWindowResized() const { return window_resized; }

    void InitWindow();
    void SetGLFWCallback();

    GLFWwindow* handle_{nullptr};

    bool window_resized = false;

    int width_, height_;
    std::string title_;

    MouseInput input_;
};

} // namespace Agate
