//
// Created by 秋鱼头 on 2022/4/9.
//

#pragma once

#if defined(_MSC_VER)
/* Disable some warnings on MSVC++ */
#pragma warning(disable : 4127 4702 4100 4515 4800 4146 4512)
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define AGATE_CPUGPU __host__ __device__
#    define AGATE_INLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
#else
#    define AGATE_CPUGPU
#    define AGATE_INLINE inline
#    define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#endif
