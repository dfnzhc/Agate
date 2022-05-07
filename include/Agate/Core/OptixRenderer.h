//
// Created by 秋鱼头 on 2022/4/11.
//

#pragma once

#include <cuda.h>
#include <optix.h>
#include <driver_types.h>
#include <Agate/Util/CudaBuffer.h>
#include <Agate/Shader/Params.h>
#include <Agate/Util/Record.hpp>
#include <Agate/Util/Camera.hpp>
#include "Interfaces.h"

namespace Agate {

struct OptixStateInfo
{
    std::string ptx_name{};
    std::string raygen{};
    std::string miss{};
    std::string closesthit{};
    std::string anyhit{};
    std::string hitgroup{};
};

class OptixRenderer
{
    CUcontext cuda_context_{};
    CUstream stream_{};
    cudaDeviceProp device_prop_{};
    OptixDeviceContext optix_context_{};

    struct OptixState
    {
        OptixModule module = {};
        OptixModuleCompileOptions module_compile_options = {};

        OptixPipeline pipeline = {};
        OptixPipelineCompileOptions pipeline_compile_options = {};
        OptixPipelineLinkOptions link_options = {};
    };

    std::unordered_map<std::string, OptixState> states_;
    std::unordered_map<std::string, OptixShaderBindingTable> sbts_;

    struct BindState
    {
        OptixPipeline pipeline = {};
        OptixShaderBindingTable sbt = {};
    };

    BindState bind_state_ = {};

    std::unordered_map<std::string, OptixProgramGroup> program_groups_;

    CudaBuffer raygen_records_buf_{};
    CudaBuffer miss_records_buf_{};
    CudaBuffer hitgroup_records_buf_{};

    LaunchParams params_{};
    CudaBuffer params_buffer_{};

    OptixTraversableHandle gas_handle_ = {};
    CUdeviceptr d_gas_output_buffer_ = {};
    /// 用于初始化 OptiX
    void InitOptiX();

    /// 创建和配置一个 optix 设备上下文
    void CreateContext();

    void createModule(std::string_view ptxName /* settings... */);

    void CreateRaygenPrograms(OptixModule module, const OptixStateInfo& info);
    void CreateMissPrograms(OptixModule module, const OptixStateInfo& info);
    void CreateHitGroupPrograms(OptixModule module, const OptixStateInfo& info);
    void setPipeline(const OptixStateInfo& info);
public:
    OptixRenderer();
    ~OptixRenderer();

    OptixRenderer(const OptixRenderer&) = delete;
    OptixRenderer& operator=(const OptixRenderer&) = delete;
    OptixDeviceContext getContext() { return optix_context_; }

    void finalize(const OptixStateInfo& info);
    void clearup();

    void bind(std::string_view name);
    void updateCamera(const Camera* camera);
    void Render();
    void Resize(const int2& newSize, uchar4* mapped_buffer);

    void createSBT(const OptixStateInfo& info);

    void addSBT(std::string_view name, OptixShaderBindingTable sbt);
};

} // namespace Agate

 