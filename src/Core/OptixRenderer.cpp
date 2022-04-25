//
// Created by 秋鱼头 on 2022/4/11.
//

#include "Agate/Core/OptixRenderer.h"
#include "Agate/Core/Error.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <Agate/Shader/VertexInfo.h>
#include <Agate/Util/ReadFile.h>

namespace Agate {

/// 对于不携带数据的着色器程序，只需要一个 header
struct SbtRecordHeader
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
};

/// 用于携带数据的着色器程序，模板函数方便传递不同类型的信息
template<typename T>
struct SbtRecordData
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]{};
    T data;
};

using SbtRecordGeometryInstanceData = SbtRecordData<GeometryInstanceData>;

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

    SetCompileOptions();

    CreateRaygenPrograms();
    CreateMissPrograms();
    CreateHitgroupPrograms();

    CreatePipeline();

    BuildSBT();

    params_buffer_.resize(sizeof(params_));

    LOG_INFO("Optix 渲染器初始化完成！");
}
OptixRenderer::~OptixRenderer()
{

}

void OptixRenderer::InitOptiX()
{
    LOG_INFO("初始化 Optix...")

    int version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&version));

    int major = version / 1000;
    int minor = (version - major * 1000) / 10;
    LOG_TRACE("CUDA 版本为：{}.{}", major, minor)

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    LOG_TRACE("发现 {} 个 CUDA 设备.", deviceCount)

    OPTIX_CHECK(optixInit());
}

void OptixRenderer::CreateContext()
{
    LOG_INFO("创建 Optix 上下文...")

    const int DeviceID = 0;
    CUDA_CHECK(cudaSetDevice(DeviceID));
    CUDA_CHECK(cudaStreamCreate(&stream_));

    cudaGetDeviceProperties(&device_prop_, DeviceID);
    LOG_TRACE("当前使用的 GPU 设备 {}.", device_prop_.name)

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &ContextLogCB;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, &options, &optix_context_));
}

void OptixRenderer::SetCompileOptions()
{
    LOG_INFO("创建 Optix module...")

    module_compile_options_.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline_compile_options_ = {};
    // 与 OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS 有什么区别？
    pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options_.usesMotionBlur = false;
    pipeline_compile_options_.numPayloadValues = 2;
    pipeline_compile_options_.numAttributeValues = 2;

    pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";

    // TODO 传入 ptx 文件参数
    const std::string ptxCode = ReadPTX("optixHello");

    OPTIX_LOG_V();
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(optix_context_,
                                             &module_compile_options_,
                                             &pipeline_compile_options_,
                                             ptxCode.c_str(),
                                             ptxCode.size(),
                                             log, &sizeof_log,
                                             &module_
    ));
}

void OptixRenderer::CreateRaygenPrograms()
{
    LOG_INFO("创建 Raygen 着色器程序...")

    raygenPGs_.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgDesc.raygen.module = module_;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    OPTIX_LOG_V();
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &raygenPGs_[0]
    ));
}

void OptixRenderer::CreateMissPrograms()
{
    LOG_INFO("创建 Miss 着色器程序...")

    missPGs_.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgDesc.raygen.module = module_;
    pgDesc.raygen.entryFunctionName = "__miss__radiance";

    OPTIX_LOG_V();
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &missPGs_[0]
    ));
}

void OptixRenderer::CreateHitgroupPrograms()
{
    LOG_INFO("创建 Hitgroup 着色器程序组...")

    hitgroupPGs_.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgDesc.hitgroup.moduleCH = module_;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module_;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    OPTIX_LOG_V();
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &hitgroupPGs_[0]
    ));
}

void OptixRenderer::CreatePipeline()
{
    LOG_INFO("创建着色器流水线...")

    OPTIX_LOG_V();
    std::vector<OptixProgramGroup> program_groups;
    for (auto pg : raygenPGs_)
        program_groups.push_back(pg);
    for (auto pg : missPGs_)
        program_groups.push_back(pg);
    for (auto pg : hitgroupPGs_)
        program_groups.push_back(pg);

    const uint32_t max_trace_depth = 0;
    pipeline_link_options_.maxTraceDepth = max_trace_depth;
    pipeline_link_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_CHECK(optixPipelineCreate(optix_context_,
                                    &pipeline_compile_options_,
                                    &pipeline_link_options_,
                                    program_groups.data(),
                                    (int) program_groups.size(),
                                    log, &sizeof_log,
                                    &pipeline_
    ));

    // STACK SIZES
    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0,  // maxCCDepth
                                           0,  // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));

    OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          2  // maxTraversableDepth
    ));
}

void OptixRenderer::BuildSBT()
{
    LOG_INFO("创建 Shader Binging Table(SBT)...")

    /// ------------------------------------------------------------------
    /// build ray generation program records
    /// ------------------------------------------------------------------
    std::vector<SbtRecordHeader> raygenRecords;
    for (auto& raygenPG : raygenPGs_) {
        SbtRecordHeader rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rec));
        raygenRecords.push_back(rec);
    }
    raygen_records_buf_.allocAndUpload(raygenRecords);
    sbt_.raygenRecord = raygen_records_buf_.get();

    /// ------------------------------------------------------------------
    /// build miss program records
    /// ------------------------------------------------------------------
    std::vector<SbtRecordHeader> missRecords;
    for (auto& missPGs : raygenPGs_) {
        SbtRecordHeader rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs, &rec));
        missRecords.push_back(rec);
    }
    miss_records_buf_.allocAndUpload(missRecords);
    sbt_.missRecordBase = miss_records_buf_.get();
    sbt_.missRecordCount = static_cast<int>(missRecords.size());
    sbt_.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));


    /// ------------------------------------------------------------------
    /// build hitgroup program records
    /// ------------------------------------------------------------------
    std::vector<SbtRecordHeader> hitgroupRecords;

    // TODO 设置物体信息
    for (auto& missPGs : raygenPGs_) {
        SbtRecordHeader rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs, &rec));
        hitgroupRecords.push_back(rec);
    }
    hitgroup_records_buf_.allocAndUpload(hitgroupRecords);
    sbt_.hitgroupRecordBase = hitgroup_records_buf_.get();
    sbt_.hitgroupRecordCount = static_cast<int>(hitgroupRecords.size());
    sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));
}

void OptixRenderer::Render()
{
    if (params_.frame_buffer_size.x == 0)
        return;

    params_buffer_.upload(&params_, 1);
    params_.frameID++;

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        pipeline_, stream_,
        /*! parameters and SBT */
        params_buffer_.get(),
        params_buffer_.byte_size,
        &sbt_,
        /*! dimensions of the launch: */
        params_.frame_buffer_size.x,
        params_.frame_buffer_size.y,
        1));

    CUDA_SYNC_CHECK();
}

void OptixRenderer::Resize(const int2& newSize, uchar4* mapped_buffer)
{
    if (newSize.x == 0 || newSize.y == 0)
        return;

    // update the launch parameters that we'll pass to the optix
    // launch:
    params_.frame_buffer_size = newSize;
    params_.color_buffer = mapped_buffer;
}

void OptixRenderer::createModule(std::string_view name)
{
    LOG_INFO("创建 Optix module: {}...", name)
    
    ModuleState moduleState;
    
    moduleState.compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleState.compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleState.compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    PipelineState pipelineState;
    pipelineState.compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineState.compile_options.usesMotionBlur = false;
    pipelineState.compile_options.numPayloadValues = 2;
    pipelineState.compile_options.numAttributeValues = 2;

    pipelineState.compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineState.compile_options.pipelineLaunchParamsVariableName = "params";

    const std::string ptxCode = ReadPTX(name);

    OPTIX_LOG_V();
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(optix_context_,
                                             &moduleState.compile_options,
                                             &pipelineState.compile_options,
                                             ptxCode.c_str(),
                                             ptxCode.size(),
                                             log, &sizeof_log,
                                             &moduleState.module
    ));
    
    modules_[name.data()] = moduleState;
    pipelines_[name.data()] = pipelineState;
}

void OptixRenderer::createPipeline(std::string_view name,
                                   const std::vector<OptixProgramGroup>& raygenPGs,
                                   const std::vector<OptixProgramGroup>& missPGs,
                                   const std::vector<OptixProgramGroup>& hitgroupPGs)
{
    LOG_INFO("创建着色器流水线: {}...", name)

    std::vector<OptixProgramGroup> program_groups;
    for (auto pg : raygenPGs)
        program_groups.push_back(pg);
    for (auto pg : missPGs)
        program_groups.push_back(pg);
    for (auto pg : hitgroupPGs)
        program_groups.push_back(pg);

    PipelineState pipelineState = pipelines_[name.data()];
    
    const uint32_t max_trace_depth = 0;
    pipelineState.link_options.maxTraceDepth = max_trace_depth;
    pipelineState.link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_LOG_V();
    OPTIX_CHECK(optixPipelineCreate(optix_context_,
                                    &pipelineState.compile_options,
                                    &pipelineState.link_options,
                                    program_groups.data(),
                                    (int) program_groups.size(),
                                    log, &sizeof_log,
                                    &pipelineState.pipeline
    ));

    // STACK SIZES
    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0,  // maxCCDepth
                                           0,  // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));

    OPTIX_CHECK(optixPipelineSetStackSize(pipelineState.pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          2  // maxTraversableDepth
    ));
}

void OptixRenderer::addSBT(std::string_view name, OptixShaderBindingTable sbt)
{
    sbts_[name.data()] = sbt;
}

} // namespace Agate