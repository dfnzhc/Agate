//
// Created by 秋鱼头 on 2022/4/18.
//

#pragma once

#include <builtin_types.h>

namespace Agate {

struct MaterialData
{
    float4 base_color{1.0f, 1.0f, 1.0f, 1.0f};
    float metallic = 1.0f;
    float roughness = 1.0f;

    cudaTextureObject_t base_color_tex = 0;
    cudaTextureObject_t metallic_roughness_tex = 0;
    cudaTextureObject_t normal_tex = 0;
};

} // namespace Agate