//
// Created by 秋鱼头 on 2022/4/24.
//

#pragma once

#include "CudaBufferView.hpp"
#include <vector_types.h>

namespace Agate {

struct GeometryData
{
    struct TriangleMesh
    {
        GenericBufferView  indices;
        BufferView<float3> positions;
        BufferView<float3> normals;
        BufferView<float2> texcoords;
    };
    
    
    TriangleMesh triangle_mesh;
};

} // namespace Agate