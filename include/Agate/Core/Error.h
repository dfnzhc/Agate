//
// Created by 秋鱼头 on 2022/4/11.
//

#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include "Agate/Core/Common.h"

#define CUDA_CHECK(call) \
{                                                                     \
    call;                                          \
    cudaError_t err = cudaGetLastError();                             \
    if( err != cudaSuccess )                                          \
    {                                                                 \
        throw Agate::AgateException("Error (%s: line %d): %s", __FILE__, __LINE__, cudaGetErrorString( err ) ); \
    }                                                                 \
};

#define OPTIX_CHECK( call )                                           \
{                                                                     \
    OptixResult err = call;                                           \
    if( err != OPTIX_SUCCESS )                                        \
    {                                                                 \
        throw Agate::AgateException("Optix call (%s) failed with code %d (line %d)", #call, err, __LINE__ ); \
    }                                                                 \
};
  
#define CUDA_SYNC_CHECK()                                             \
{                                                                     \
    cudaDeviceSynchronize();                                          \
    cudaError_t err = cudaGetLastError();                             \
    if( err != cudaSuccess )                                          \
    {                                                                 \
        throw Agate::AgateException("Error (%s: line %d): %s", __FILE__, __LINE__, cudaGetErrorString( err ) ); \
    }                                                                 \
};