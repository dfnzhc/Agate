//
// Created by 秋鱼头 on 2022/4/12.
//

#pragma once

#include <cstdint>
#include <vector_types.h>

namespace Agate {

struct VertexAttributes
{
    float3 vertex;
    float3 tangent;
    float3 normal;
    float3 texcoord;
};

struct GeometryInstanceData
{
    VertexAttributes* attributes;

    int3* indices;
//    int materialIndex;
};

} // namespace Agate
