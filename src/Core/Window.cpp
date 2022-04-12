//
// Created by 秋鱼头 on 2022/4/10.
//
#include "Agate/Core/Window.h"
#include <Agate/Core/Common.h>

namespace Agate {

static void GLFW_Error_Callback(int error, const char* description);

AgateWindow::AgateWindow(int w, int h, std::string_view title)
    : width_{w}, height_{h}, title_{title}
{
    InitWindow();
}

AgateWindow::~AgateWindow()
{
    glfwDestroyWindow(handle_);
    glfwTerminate();
}

void AgateWindow::Draw()
{

}

void AgateWindow::Render()
{

}

void AgateWindow::Run()
{
    int width, height;
    glfwGetFramebufferSize(handle_, &width, &height);
    Resize({width, height});

    while (!ShouldClose()) {
        glfwPollEvents();
        
        Render();
        Draw();

        glfwSwapBuffers(handle_);
    }
}

void AgateWindow::Resize(const int2& newSize)
{

}

void AgateWindow::InitWindow()
{
    LOG_INFO("创建 GLFW 窗口")
    
    glfwSetErrorCallback(GLFW_Error_Callback);

    if (!glfwInit()) {
        throw AgateException("GLFW failed to initialize.");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    handle_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
    if (!handle_) {
        glfwTerminate();
        throw AgateException("Failed to create GLFW window.");
    }

    glfwSetWindowUserPointer(handle_, this);
    glfwMakeContextCurrent(handle_);
    glfwSwapInterval(1);

    SetGLFWCallback();
}

void AgateWindow::SetGLFWCallback()
{
    glfwSetScrollCallback(
        handle_,
        [](GLFWwindow* window, double xoffset, double yoffset)
        {
            // ,,
        }
    );

    glfwSetKeyCallback(
        handle_,
        [](GLFWwindow* window, int key, int scancode, int action, int mods)
        {
            const bool pressed = action != GLFW_RELEASE;
            if (key == GLFW_KEY_ESCAPE && pressed)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    );
}

static void GLFW_Error_Callback(int error, const char* description)
{
    LOG_ERROR("{}: {}.", error, description)
}

} // namespace Agate