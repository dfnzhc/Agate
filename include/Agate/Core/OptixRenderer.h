//
// Created by 秋鱼头 on 2022/4/11.
//

#pragma once


#include <cuda.h>
#include <optix.h>
#include <driver_types.h>
#include <Agate/Util/CudaBuffer.h>
#include <Agate/Shader/Params.h>
#include "Interfaces.h"

namespace Agate {

class OptixRenderer
{
    CUcontext cuda_context_{};
    CUstream stream_{};
    cudaDeviceProp device_prop_{};

    OptixDeviceContext optix_context_{};

    OptixPipeline pipeline_{};
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    OptixPipelineLinkOptions pipeline_link_options_ = {};

    OptixModule module_{};
    OptixModuleCompileOptions module_compile_options_ = {};

    OptixShaderBindingTable sbt = {};

    std::vector<OptixProgramGroup> raygenPGs_;
    std::vector<OptixProgramGroup> missPGs_;
    std::vector<OptixProgramGroup> hitgroupPGs_;

    CudaBuffer raygen_records_buf_{};
    CudaBuffer miss_records_buf_{};
    CudaBuffer hitgroup_records_buf_{};
    
    LaunchParams params_{};
    CudaBuffer params_buffer_{};
    
    CudaBuffer color_buffer_{};

    /// 用于初始化 OptiX
    void InitOptiX();

    /// 创建和配置一个 optix 设备上下文
    void CreateContext();

    void SetCompileOptions();

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
    
    void Render();
    void Resize(const int2& newSize, uchar4* mapped_buffer);
};

} // namespace Agate

 