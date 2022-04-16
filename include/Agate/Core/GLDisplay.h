//
// Created by 秋鱼头 on 2022/4/15.
//

#pragma once

#include <vector_types.h>
#include <Agate/Util/Utils.h>

namespace Agate {

class GLDisplay
{
public:
    GLDisplay(
        BufferImageFormat format = BufferImageFormat::UNSIGNED_BYTE4);

    void Display(
        int32_t screen_res_x,
        int32_t screen_res_y,
        int32_t framebuf_res_x,
        int32_t framebuf_res_y,
        uint32_t pbo) const;

private:
    GLuint render_tex_ = 0u;
    GLuint program_ = 0u;
    GLint render_tex_uniform_loc_ = -1;
    GLuint quad_vertex_buffer_ = 0;

    BufferImageFormat image_format_;

    static const std::string vert_source_;
    static const std::string frag_source_;
};
} // namespace Agate
