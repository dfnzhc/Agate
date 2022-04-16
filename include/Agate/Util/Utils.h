//
// Created by 秋鱼头 on 2022/4/15.
//

#pragma once

#include <builtin_types.h>
#include <Agate/Core/Common.h>
namespace Agate {

inline int GetSize(const int2& in)
{
    return in.x * in.y;
}

enum BufferImageFormat
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct ImageBuffer
{
    void* data =      nullptr;
    unsigned int      width = 0;
    unsigned int      height = 0;
    BufferImageFormat pixel_format = UNSIGNED_BYTE4;
};

struct Texture
{
    cudaArray_t         array;
    cudaTextureObject_t texture;
};


inline size_t PixelFormatSize(BufferImageFormat format )
{
    switch( format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
            return sizeof( char ) * 4;
        case BufferImageFormat::FLOAT3:
            return sizeof( float ) * 3;
        case BufferImageFormat::FLOAT4:
            return sizeof( float ) * 4;
        default:
            throw AgateException( "Unrecognized buffer format" );
    }
}

} // namespace Agate