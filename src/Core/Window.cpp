//
// Created by 秋鱼头 on 2022/4/10.
//
#include "Agate/Core/Window.h"
#include <Agate/Core/Common.h>
#include <Agate/Core/Error.h>

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

void AgateWindow::InitWindow()
{
    LOG_INFO("创建 GLFW 窗口")

    glfwSetErrorCallback(GLFW_Error_Callback);

    if (!glfwInit()) {
        throw AgateException("GLFW failed to initialize.");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    handle_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
    if (!handle_) {
        glfwTerminate();
        throw AgateException("Failed to create GLFW window.");
    }

    glfwSetWindowUserPointer(handle_, this);
    glfwMakeContextCurrent(handle_);
    glfwSwapInterval(0);

    if (!gladLoadGL()) {
        throw AgateException("GL 初始化失败.");
    }

    GL_CHECK(glClearColor(0.222f, 0.723f, 0.367f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));

    SetGLFWCallback();
}

void AgateWindow::SetGLFWCallback()
{
    glfwSetKeyCallback(
        handle_,
        [](GLFWwindow* window, int key, int scancode, int action, int mods)
        {
            const bool pressed = action != GLFW_RELEASE;
            if (key == GLFW_KEY_ESCAPE && pressed)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    );

    glfwSetFramebufferSizeCallback(
        handle_,
        [](GLFWwindow* window, int width, int height)
        {
            auto* aw = static_cast<AgateWindow*>(glfwGetWindowUserPointer(window));
            AGATE_ASSERT(aw);
            aw->Resize({width, height});
        }
    );

    glfwSetMouseButtonCallback(
        handle_,
        [](GLFWwindow* window, int button, int action, int mods)
        {
            auto* aw = static_cast<AgateWindow*>(glfwGetWindowUserPointer(window));
            AGATE_ASSERT(aw);

            if (action == GLFW_PRESS) {
                aw->input_.mouse_button = button;
            } else {
                aw->input_.mouse_button = -1;
            }
        }
    );

    glfwSetCursorPosCallback(
        handle_,
        [](GLFWwindow* window, double xpos, double ypos)
        {
            auto* aw = static_cast<AgateWindow*>(glfwGetWindowUserPointer(window));
            AGATE_ASSERT(aw);
            aw->input_.posX = static_cast<int>(xpos);
            aw->input_.posY = static_cast<int>(ypos);

            aw->cursorUpdate();
        }
    );

    glfwSetScrollCallback(
        handle_,
        [](GLFWwindow* window, double xscroll, double yscroll)
        {
            auto* aw = static_cast<AgateWindow*>(glfwGetWindowUserPointer(window));
            AGATE_ASSERT(aw);
            aw->input_.scroll = static_cast<int>(yscroll);

            aw->cursorUpdate();
        }
    );
}

static void GLFW_Error_Callback(int error, const char* description)
{
    LOG_ERROR("{}: {}.", error, description)
}

} // namespace Agate