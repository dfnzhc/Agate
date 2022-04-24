//
// Created by 秋鱼头 on 2022/4/12.
//

#pragma once

#include "Geometry.hpp"
#include "Material.hpp"
#include "VertexInfo.h"

namespace Agate {

struct LaunchParams
{
    int frameID{0};
    uchar4* color_buffer{nullptr};
    int2 frame_buffer_size{0, 0};

    float3 eye{};
    float3 U{};
    float3 V{};
    float3 W{};

    OptixTraversableHandle traversable{};
};

struct RayGenData
{
    /// ...
};

struct MissData
{
    /// ...
};

struct HitGroupData
{
    GeometryData geometry_data;
    MaterialData material_data;
};

} // namespace Agate