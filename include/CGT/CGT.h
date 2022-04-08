//
// Created by 秋鱼头 on 2022/3/26.
//
#pragma once

#if defined(_MSC_VER)
/* Disable some warnings on MSVC++ */
#pragma warning(disable : 4127 4702 4100 4515 4800 4146 4512)
#endif

#ifdef CGT_USE_GPU
#define CGT_GPU __device__
#define CGT_CPU_GPU __host__ __device__
#define CGT_CONST __device__ const
#else
#define CGT_GPU
#define CGT_CPU_GPU
#define CGT_CONST const
#endif

#ifdef CGT_DBG_LOGGING
#include <cstdio>
#ifndef CGT_TO_STRING
#define CGT_TO_STRING(x) CGT_TO_STRING2(x)
#define CGT_TO_STRING2(x) #x
#endif  // !CGT_TO_STRING
#ifdef CGT_USE_GPU
#define CGT_DBG(...) printf(__FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)
#else
#define CGT_DBG(...) fprintf(stderr, __FILE__ ":" TO_STRING(__LINE__) ": " __VA_ARGS__)
#endif  // CGT_USE_GPU
#else
#define CGT_DBG(...)
#endif  // CGT_DBG_LOGGING

#include <cstdint>
#include <cstddef>
namespace CGT {
#ifdef CGT_FLOAT_AS_DOUBLE
using Float = double;
#else
using Float = float;
#endif

#ifdef CGT_FLOAT_AS_DOUBLE
using FloatBits = uint64_t;
#else
using FloatBits = uint32_t;
#endif  // CGT_FLOAT_AS_DOUBLE
static_assert(sizeof(Float) == sizeof(FloatBits), "Float and FloatBits must have the same size");
    
    

} // namespace CGT
