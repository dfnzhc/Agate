#include "LaunchParameter.h"


namespace Agate {

extern "C" __constant__ OptixLaunchParams optixLaunchParams;

extern "C" __global__ void __closesthit__radiance(){}

} // namespace Agate
