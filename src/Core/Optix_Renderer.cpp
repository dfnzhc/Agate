//
// Created by 秋鱼头 on 2022/4/11.
//

#include "Agate/Core/Optix_Renderer.h"
#include "Agate/Core/Error.h"

#include <optix_function_table_definition.h>

namespace Agate {

extern "C" char embedded_ptx_code[];

static void ContextLogCB(unsigned int level,
                         const char* tag,
                         const char* message,
                         void*)
{
    LOG_DEBUG("[{}]: {}", tag, message)
}

OptixRenderer::OptixRenderer()
{
    InitOptiX();

    CreateContext();
    
    CreateModule();
}
OptixRenderer::~OptixRenderer()
{

}

void OptixRenderer::InitOptiX()
{
    LOG_INFO("初始化 Optix...")
    /// Initialize CUDA driver API
    CUresult cuRes = cuInit(0);
    if (cuRes != CUDA_SUCCESS) {
        throw AgateException("初始化 CUDA API 失败.");
    }

    int version = 0;
    CUDA_CHECK(cuDriverGetVersion(&version))

    int major = version / 1000;
    int minor = (version - major * 1000) / 10;
    LOG_TRACE("CUDA 版本为：{}.{}", major, minor)

    int deviceCount = 0;
    CUDA_CHECK(cuDeviceGetCount(&deviceCount))
    LOG_TRACE("发现 {} 个 CUDA 设备.", deviceCount)

    OPTIX_CHECK(optixInit())
}

void OptixRenderer::CreateContext()
{
    LOG_INFO("创建 Optix 上下文.")

    const int DeviceID = 0;
    CUDA_CHECK(cudaSetDevice(DeviceID))
    CUDA_CHECK(cudaStreamCreate(&stream_))

    cudaGetDeviceProperties(&device_prop_, DeviceID);
    LOG_TRACE("当前使用的 GPU 设备 {}.", device_prop_.name)

    CUDA_CHECK(cuCtxGetCurrent(&cuda_context_))

    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, nullptr, &optix_context_))
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                    (optix_context_, ContextLogCB, nullptr, 4))
}

void OptixRenderer::CreateModule()
{
    LOG_INFO("创建 Optix module.")

    module_compile_options_.maxRegisterCount = 50;
    module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline_compile_options_ = {};
    pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options_.usesMotionBlur = false;
    pipeline_compile_options_.numPayloadValues = 2;
    pipeline_compile_options_.numAttributeValues = 2;
    pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipeline_link_options_.maxTraceDepth = 2;

    const std::string ptxCode = "";//embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(optix_context_,
                                         &module_compile_options_,
                                         &pipeline_compile_options_,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log, &sizeof_log,
                                         &module_
    ))

    if (sizeof_log > 1) {
        LOG_TRACE(log)
    }
}

void OptixRenderer::CreateRaygenPrograms()
{

}

void OptixRenderer::CreateMissPrograms()
{

}

void OptixRenderer::CreateHitgroupPrograms()
{

}

void OptixRenderer::CreatePipeline()
{

}

void OptixRenderer::BuildSBT()
{

}

} // namespace Agate