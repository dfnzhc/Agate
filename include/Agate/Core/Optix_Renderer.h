//
// Created by 秋鱼头 on 2022/4/11.
//

#pragma once

#include <cuda.h>
#include <optix.h>
#include <driver_types.h>

namespace Agate {

class OptixRenderer
{
    CUcontext          cuda_context_{};
    CUstream           stream_{};
    cudaDeviceProp     device_prop_{};

    OptixDeviceContext optix_context_{};

    OptixPipeline               pipeline_{};
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    OptixPipelineLinkOptions    pipeline_link_options_ = {};

    OptixModule                 module_{};
    OptixModuleCompileOptions   module_compile_options_ = {};

    /// 用于初始化 OptiX
    void InitOptiX();
 
    /// 创建和配置一个 optix 设备上下文
    void CreateContext();

    void CreateModule();

    void CreateRaygenPrograms();
    void CreateMissPrograms();
    void CreateHitgroupPrograms();

    void CreatePipeline();

    void BuildSBT();

public:
    OptixRenderer();
    ~OptixRenderer();

    OptixRenderer(const OptixRenderer&) = delete;
    OptixRenderer& operator=(const OptixRenderer&) = delete;
};

} // namespace Agate

 