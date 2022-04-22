//
// Created by 秋鱼头 on 2022/4/18.
//

#pragma once

#include <vector_types.h>
#include "Agate/Shader/vec_math.h"

namespace Agate {

struct Camera
{
    float3 eye{1.0f, 1.0f, 1.0f};
    float3 lookat{0.0f, 0.0f, 0.0f};
    float3 up{0.0f, 1.0f, 0.0f};

    float fovY = 45.0f;
    float aspectRatio = 1.0f;

    Camera() = default;
    Camera(const float3& eye, const float3& lookat, const float3& up, float fovY, float aspectRatio)
        : eye(eye), lookat(lookat), up(up), fovY(fovY), aspectRatio(aspectRatio) {}

    inline float3 direction() const { return normalize(lookat - eye); }
    void setDirection(const float3& dir) { lookat = eye + length(lookat - eye) * dir; }

    // UVW forms an orthogonal, but not orthonormal basis!
    inline void UVWFrame(float3& U, float3& V, float3& W) const
    {
        W = lookat - eye; // Do not normalize W -- it implies focal length
        float wlen = length(W);
        U = normalize(cross(W, up));
        V = normalize(cross(U, W));

        float vlen = wlen * tanf(0.5f * fovY * M_PIf / 180.0f);
        V *= vlen;
        float ulen = vlen * aspectRatio;
        U *= ulen;
    }
};

} // namespace Agate