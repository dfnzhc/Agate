//
// Created by 秋鱼头 on 2022/4/24.
//

#pragma once

#include <optix.h>
#include <driver_types.h>

#include <Agate/Shader/Params.h>

namespace Agate {

struct RecordHeader
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
};

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    T data;
};

struct EmptyData {};
using EmptyRecord       = Record<EmptyData>;

using RaygenRecord      = Record<RayGenData>;
using MissRecord        = Record<MissData>;
using HitGroupRecord    = Record<HitGroupData>;

} // namespace Agate