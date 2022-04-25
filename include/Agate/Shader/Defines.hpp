//
// Created by 秋鱼 on 2022/4/25.
//

#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define AGATE_CPUGPU __host__ __device__
#    define AGATE_INLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
#else
#    define AGATE_CPUGPU
#    define AGATE_INLINE inline
#    define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#endif
