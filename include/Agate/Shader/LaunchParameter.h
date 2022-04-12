//
// Created by 秋鱼头 on 2022/4/12.
//

#pragma once

#include "VertexInfo.h"

namespace Agate {

struct OptixLaunchParams
{
    int frameID{0};
    uint32_t* color_buffer{nullptr};
    int2 frame_buffer_size{0, 0};
};

} // namespace Agate