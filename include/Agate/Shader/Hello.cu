#include <optix.h>

#include "Params.h"
#include "random.h"
#include "vec_math.h"
#include "help.h"

namespace Agate {

extern "C" __constant__ LaunchParams params;

extern "C" __global__ void __closesthit__Hello() { /*! for this simple example, this will remain empty */ }

extern "C" __global__ void __anyhit__Hello() { /*! for this simple example, this will remain empty */ }

extern "C" __global__ void __miss__Hello() { /*! for this simple example, this will remain empty */ }
  
extern "C" __global__ void __raygen__Hello()
{
    const int frameID = params.frameID; 

    const uint3  launch_idx     = optixGetLaunchIndex();
    const uint3  launch_dims    = optixGetLaunchDimensions();
    const float3 eye            = params.eye;
    const float3 U              = params.U;
    const float3 V              = params.V;
    const float3 W              = params.W;

    unsigned int seed = tea<4>( launch_idx.y * launch_dims.x + launch_idx.x, frameID );

    // The center of each pixel is at fraction (0.5,0.5)
    const float2 subpixel_jitter =
        frameID == 0 ? make_float2( 0.5f, 0.5f ) : make_float2( rnd( seed ), rnd( seed ) );

    const float2 d =
        2.0f * make_float2( ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
                           ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y ) )
        - 1.0f;
    
    const float3 ray_direction = normalize( d.x * U + d.y * V + W );
    const float3 ray_origin    = eye;

    // and write to frame buffer ...
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    params.color_buffer[image_index] = make_color(  ray_origin );
}

} // namespace Agate
