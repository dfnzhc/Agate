#include <optix.h>

#include "Params.h"
#include "random.h"
#include "vec_math.h"
#include "help.h"

namespace Agate {

extern "C" __constant__ LaunchParams params;

static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(float_as_int(p.x));
    optixSetPayload_1(float_as_int(p.y));
    optixSetPayload_2(float_as_int(p.z));
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = normalize(params.W);
    const float2 d = 2.0f * make_float2(
        static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
        static_cast<float>( idx.y ) / static_cast<float>( dim.y )
    ) - 1.0f;

    origin = params.eye;
    direction = normalize(d.x * U + d.y * V + W);
}

extern "C" __global__ void __closesthit__Hello()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    setPayload(make_float3(barycentrics, 1.0f));
}

extern "C" __global__ void __anyhit__Hello() { /*! for this simple example, this will remain empty */ }

extern "C" __global__ void __miss__Hello() { setPayload(float3{0.2, 0.3, 0.5}); }

extern "C" __global__ void __raygen__Hello()
{
    const int frameID = params.frameID;

    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay(idx, dim, ray_origin, ray_direction);

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
        params.traversable,
        ray_origin,
        ray_direction,
        0.0f,                // Min intersection distance
        1e16f,               // Max intersection distance
        0.0f,                // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset   -- See SBT discussion
        1,                   // SBT stride   -- See SBT discussion
        0,                   // missSBTIndex -- See SBT discussion
        p0, p1, p2);
    float3 result;
    result.x = int_as_float(p0);
    result.y = int_as_float(p1);
    result.z = int_as_float(p2);

    // and write to frame buffer ...
    params.color_buffer[idx.y * params.frame_buffer_size.x + idx.x] = make_color(result);
}

} // namespace Agate
