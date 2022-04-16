//
// Created by 秋鱼头 on 2022/4/11.
//

#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "Agate/Core/Common.h"


//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw Agate::AgateException( ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )

#define CUDA_SYNC_CHECK()                                                      \
    do                                                                         \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw Agate::AgateException( ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )

#define CUDA_CHECK_NOTHROW(call)                                             \
    do                                                                         \
    {                                                                          \
        cudaError_t error = (call);                                            \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::cerr << "CUDA call (" << #call << " ) failed with error: '"   \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            std::terminate();                                                  \
        }                                                                      \
    } while( 0 )

//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------
#define OPTIX_CHECK(call)                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw Agate::AgateException(res, ss.str().c_str());                \
        }                                                                      \
    } while( 0 )

#define OPTIX_LOG_V() char log[2048]; \
                      size_t sizeof_log = sizeof(log)

#define OPTIX_CHECK_LOG(call)                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        const size_t sizeof_log_returned = sizeof_log;                         \
        sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */    \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) \
               << "\n";                                                        \
            throw Agate::AgateException(res, ss.str().c_str());                \
        }                                                                      \
    } while( 0 )

#define OPTIX_CHECK_NOTHROW(call)                                            \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::cerr << "Optix call '" << #call                               \
                      << "' failed: " __FILE__ ":" << __LINE__ << ")\n";       \
            std::terminate();                                                  \
        }                                                                      \
    } while( 0 )


//------------------------------------------------------------------------------
//
// Assertions
//
//------------------------------------------------------------------------------
#define AGATE_ASSERT(cond)                                                   \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << __FILE__ << " (" << __LINE__ << "): " << #cond;              \
            throw Agate::AgateException( ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )

#define AGATE_ASSERT_MSG(cond, msg)                                          \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << (msg) << ": " << __FILE__ << " (" << __LINE__ << "): " << #cond ; \
            throw Agate::AgateException( ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )


//------------------------------------------------------------------------------
//
// GL error-checking
//
//------------------------------------------------------------------------------

#include <glad/glad.h>

inline const char* GetGLErrorString( GLenum error )
{
    switch( error )
    {
        case GL_NO_ERROR:            return "No error";
        case GL_INVALID_ENUM:        return "Invalid enum";
        case GL_INVALID_VALUE:       return "Invalid value";
        case GL_INVALID_OPERATION:   return "Invalid operation";
            //case GL_STACK_OVERFLOW:      return "Stack overflow";
            //case GL_STACK_UNDERFLOW:     return "Stack underflow";
        case GL_OUT_OF_MEMORY:       return "Out of memory";
            //case GL_TABLE_TOO_LARGE:     return "Table too large";
        default:                     return "Unknown GL error";
    }
}


inline void CheckGLError()
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::ostringstream oss;
        do
        {
            oss << "GL error: " << GetGLErrorString( err ) << "\n";
            err = glGetError();
        }
        while( err != GL_NO_ERROR );

        throw Agate::AgateException( oss.str().c_str() );
    }
}

#define GL_CHECK(call)                                                         \
        do                                                                     \
        {                                                                      \
            call;                                                              \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
                std::stringstream ss;                                          \
                ss << "GL error " <<  GetGLErrorString( err ) << " at " \
                   << __FILE__  << "(" <<  __LINE__  << "): " << #call         \
                   << std::endl;                                               \
                std::cerr << ss.str() << std::endl;                            \
                throw Agate::AgateException( ss.str().c_str() );               \
            }                                                                  \
        }                                                                      \
        while (0)

#define GL_CHECK_ERRORS()                                                      \
        do                                                                     \
        {                                                                      \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
                std::stringstream ss;                                          \
                ss << "GL error " <<  GetGLErrorString( err ) << " at "        \
                   << __FILE__  << "(" <<  __LINE__  << ")";                   \
                std::cerr << ss.str() << std::endl;                            \
                throw Agate::AgateException( ss.str().c_str() );               \
            }                                                                  \
        }                                                                      \
        while (0)
