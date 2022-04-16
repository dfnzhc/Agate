//
// Created by 秋鱼头 on 2022/4/15.
//

#include "Agate/Core/Error.h"
#include "Agate/Core/GLDisplay.h"

namespace Agate {

//-----------------------------------------------------------------------------
//
// Helper functions
//
//-----------------------------------------------------------------------------
GLuint CreateGLShader(const std::string& source, GLuint shader_type)
{
    GLuint shader = glCreateShader(shader_type);
    {
        const auto* source_data = reinterpret_cast<const GLchar*>( source.data());
        glShaderSource(shader, 1, &source_data, nullptr);
        glCompileShader(shader);

        GLint is_compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
        if (is_compiled == GL_FALSE) {
            GLint max_length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);

            std::string info_log(max_length, '\0');
            auto* info_log_data = reinterpret_cast<GLchar*>( &info_log[0]);
            glGetShaderInfoLog(shader, max_length, nullptr, info_log_data);

            glDeleteShader(shader);
            std::cerr << "Compilation of shader failed: " << info_log << std::endl;

            return 0;
        }
    }

    GL_CHECK_ERRORS();

    return shader;
}

GLuint CreateGLProgram(
    const std::string& vert_source,
    const std::string& frag_source
)
{
    GLuint vert_shader = CreateGLShader(vert_source, GL_VERTEX_SHADER);
    if (vert_shader == 0)
        return 0;

    GLuint frag_shader = CreateGLShader(frag_source, GL_FRAGMENT_SHADER);
    if (frag_shader == 0) {
        glDeleteShader(vert_shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);

    GLint is_linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
    if (is_linked == GL_FALSE) {
        GLint max_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);

        std::string info_log(max_length, '\0');
        auto* info_log_data = reinterpret_cast<GLchar*>( &info_log[0]);
        glGetProgramInfoLog(program, max_length, nullptr, info_log_data);
        LOG_ERROR("Linking of program failed: {}", info_log);

        glDeleteProgram(program);
        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);

        return 0;
    }

    glDetachShader(program, vert_shader);
    glDetachShader(program, frag_shader);

    GL_CHECK_ERRORS();

    return program;
}

GLint GetGLUniformLocation(GLuint program, const std::string& name)
{
    GLint loc = glGetUniformLocation(program, name.c_str());
    AGATE_ASSERT_MSG(loc != -1, "Failed to get uniform loc for '" + name + "'");
    return loc;
}

//-----------------------------------------------------------------------------
//
// GLDisplay implementation
//
//-----------------------------------------------------------------------------

const std::string GLDisplay::vert_source_ = R"(
#version 460 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string GLDisplay::frag_source_ = R"(
#version 460 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";

GLDisplay::GLDisplay(BufferImageFormat image_format)
    : image_format_(image_format)
{
    GLuint m_vertex_array;
    GL_CHECK(glGenVertexArrays(1, &m_vertex_array));
    GL_CHECK(glBindVertexArray(m_vertex_array));

    program_ = CreateGLProgram(vert_source_, frag_source_);
    render_tex_uniform_loc_ = GetGLUniformLocation(program_, "render_tex");

    GL_CHECK(glGenTextures(1, &render_tex_));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, render_tex_));

    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
    };

    GL_CHECK(glGenBuffers(1, &quad_vertex_buffer_));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer_));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER,
                          sizeof(g_quad_vertex_buffer_data),
                          g_quad_vertex_buffer_data,
                          GL_STATIC_DRAW
    )
    );

    GL_CHECK_ERRORS();
}

void GLDisplay::Display(int32_t screen_res_x,
                        int32_t screen_res_y,
                        int32_t framebuf_res_x,
                        int32_t framebuf_res_y,
                        uint32_t pbo) const
{
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_CHECK(glViewport(0, 0, framebuf_res_x, framebuf_res_y));

    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    GL_CHECK(glUseProgram(program_));

    // Bind our texture in Texture Unit 0
    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, render_tex_));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));
    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 4)); // TODO!!!!!!

    size_t elmt_size = PixelFormatSize(image_format_);
    if (elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if (elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if (elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    bool convertToSrgb = true;

    if (image_format_ == BufferImageFormat::UNSIGNED_BYTE4) {
        // input is assumed to be in srgb since it is only 1 byte per channel in size
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_res_x, screen_res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        convertToSrgb = false;
    } else if (image_format_ == BufferImageFormat::FLOAT3)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, screen_res_x, screen_res_y, 0, GL_RGB, GL_FLOAT, nullptr);
    else if (image_format_ == BufferImageFormat::FLOAT4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res_x, screen_res_y, 0, GL_RGBA, GL_FLOAT, nullptr);
    else
        throw AgateException("Unknown buffer format");

    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    GL_CHECK(glUniform1i(render_tex_uniform_loc_, 0));

    // 1st attribute buffer : vertices
    GL_CHECK(glEnableVertexAttribArray(0));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, quad_vertex_buffer_));
    GL_CHECK(glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*) nullptr     // array buffer offset
    )
    );

    if (convertToSrgb)
        GL_CHECK(glEnable(GL_FRAMEBUFFER_SRGB));
    else
        GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB));

    // Draw the triangles !
    GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 6)); // 2*3 indices starting at 0 -> 2 triangles
    GL_CHECK(glDisableVertexAttribArray(0));
    GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB));

    GL_CHECK_ERRORS();
}

} // namespace Agate
