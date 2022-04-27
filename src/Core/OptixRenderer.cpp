//
// Created by 秋鱼头 on 2022/4/11.
//

#include "Agate/Core/OptixRenderer.h"
#include "Agate/Core/Error.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <Agate/Shader/VertexInfo.h>
#include <Agate/Util/ReadFile.h>
#include <Agate/Util/Record.hpp>

namespace Agate {

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

    params_buffer_.resize(sizeof(params_));

    LOG_INFO("Optix 初始化完成！");
}

OptixRenderer::~OptixRenderer()
{
    clearup();
}

void OptixRenderer::finalize(const OptixStateInfo& info)
{
    createModule(info.ptx_name);
    auto m = states_.find(info.ptx_name);
    AGATE_ASSERT(m != states_.end() && "创建模组失败");

    OptixState state = m->second;
    /// 分别创建三种着色器
    CreateRaygenPrograms(state.module, info);
    CreateMissPrograms(state.module, info);
    CreateHitGroupPrograms(state.module, info);

    setPipeline(info);
    createSBT(info);
}

void OptixRenderer::clearup()
{
    for (auto it = states_.begin(); it != states_.end(); ++it) {
        OPTIX_CHECK(optixPipelineDestroy(it->second.pipeline));
        it->second.pipeline = nullptr;

        OPTIX_CHECK(optixModuleDestroy(it->second.module));
        it->second.module = nullptr;
    }

    for (auto it = program_groups_.begin(); it != program_groups_.end(); ++it) {
        OPTIX_CHECK(optixProgramGroupDestroy(it->second));
        it->second = nullptr;
    }

    for (auto it = sbts_.begin(); it != sbts_.end(); ++it) {
        if (it->second.raygenRecord) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( it->second.raygenRecord )));
            it->second.raygenRecord = 0;
        }
        if (it->second.missRecordBase) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( it->second.missRecordBase )));
            it->second.missRecordBase = 0;
        }
        if (it->second.hitgroupRecordBase) {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>( it->second.hitgroupRecordBase )));
            it->second.hitgroupRecordBase = 0;
        }
    }
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

void OptixRenderer::createModule(std::string_view ptxName)
{
    LOG_INFO("创建 Optix module: {}...", ptxName)

    OptixState state;

    state.module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    state.module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    state.module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.numPayloadValues = 2;
    state.pipeline_compile_options.numAttributeValues = 2;

    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    const std::string ptxCode = ReadPTX(ptxName);

    OPTIX_LOG_V();
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(optix_context_,
                                             &state.module_compile_options,
                                             &state.pipeline_compile_options,
                                             ptxCode.c_str(),
                                             ptxCode.size(),
                                             log, &sizeof_log,
                                             &state.module
    ));

    states_.insert({ptxName.data(), state});
}

void OptixRenderer::CreateRaygenPrograms(OptixModule module, const OptixStateInfo& info)
{
    LOG_INFO("创建着色器程序: {}...", info.raygen)

    OptixProgramGroup pg;

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = info.raygen.c_str();

    OPTIX_LOG_V();
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &pg
    ));

    program_groups_.insert({info.raygen, pg});
}

void OptixRenderer::CreateMissPrograms(OptixModule module, const OptixStateInfo& info)
{
    LOG_INFO("创建着色器程序: {}...", info.miss)

    OptixProgramGroup pg;
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = info.miss.c_str();

    OPTIX_LOG_V();
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &pg
    ));

    program_groups_.insert({info.miss, pg});
}

void OptixRenderer::CreateHitGroupPrograms(OptixModule module, const OptixStateInfo& info)
{
    LOG_INFO("创建着色器程序: {}...", info.hitgroup)

    OptixProgramGroup pg;

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = info.closesthit.c_str();
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = info.anyhit.c_str();

    OPTIX_LOG_V();
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &pg
    ));

    program_groups_.insert({info.hitgroup, pg});
}

void OptixRenderer::setPipeline(const OptixStateInfo& info)
{
    LOG_INFO("创建着色器流水线: {}...", info.ptx_name)

    std::vector<OptixProgramGroup> groups;
    groups.push_back(program_groups_[info.raygen]);
    groups.push_back(program_groups_[info.miss]);
    groups.push_back(program_groups_[info.hitgroup]);

    OptixState& state = states_[info.ptx_name];

    const uint32_t max_trace_depth = 0;
    state.link_options.maxTraceDepth = max_trace_depth;
    state.link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_LOG_V();
    OPTIX_CHECK(optixPipelineCreate(optix_context_,
                                    &state.pipeline_compile_options,
                                    &state.link_options,
                                    groups.data(),
                                    (int) groups.size(),
                                    log, &sizeof_log,
                                    &state.pipeline
    ));

    // STACK SIZES
    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : groups) {
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

    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          2  // maxTraversableDepth
    ));
}

void OptixRenderer::createSBT(const OptixStateInfo& info)
{
    auto state = states_.find(info.ptx_name);
    AGATE_ASSERT(state != states_.end() && "没有相应的 Optix State，不能创建 SBT");

    // TODO: 能够添加 sbt 信息
    LOG_INFO("创建 Shader Binging Table(SBT)...")

    OptixShaderBindingTable sbt = {};
    /// ------------------------------------------------------------------
    /// build ray generation program records
    /// ------------------------------------------------------------------
    {
        auto raygenPG = program_groups_[info.raygen];
        EmptyRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rec));
        raygen_records_buf_.allocAndUpload(&rec);
        sbt.raygenRecord = raygen_records_buf_.get();
    }

    /// ------------------------------------------------------------------
    /// build miss program records
    /// ------------------------------------------------------------------
    {
        auto missPG = program_groups_[info.miss];
        EmptyRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &rec));
        miss_records_buf_.allocAndUpload(&rec);
        sbt.missRecordBase = miss_records_buf_.get();
        sbt.missRecordCount = 1;
        sbt.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(EmptyRecord));
    }


    /// ------------------------------------------------------------------
    /// build hitgroup program records
    /// ------------------------------------------------------------------
    std::vector<EmptyRecord> hitgroupRecords;
    {
        auto chPG = program_groups_[info.hitgroup];
        EmptyRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(chPG, &rec));
        hitgroupRecords.push_back(rec);
    }

    hitgroup_records_buf_.allocAndUpload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroup_records_buf_.get();
    sbt.hitgroupRecordCount = static_cast<int>(hitgroupRecords.size());
    sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(EmptyRecord));

    sbts_.insert({info.ptx_name, sbt});
}

void OptixRenderer::addSBT(std::string_view name, OptixShaderBindingTable sbt)
{
    sbts_[name.data()] = sbt;
}

void OptixRenderer::bind(std::string_view name)
{
    auto st = states_.find(name.data());
    auto sb = sbts_.find(name.data());
    AGATE_ASSERT(st != states_.end() &&
        sb != sbts_.end() &&
        "没有相应的 Optix 状态和 SBT");

    bind_state_.pipeline = st->second.pipeline;
    bind_state_.sbt = sb->second;
}

void OptixRenderer::Render()
{
    if (params_.frame_buffer_size.x == 0)
        return;

    params_buffer_.upload(&params_, 1);
    params_.frameID++;

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        bind_state_.pipeline, stream_,
        /*! parameters and SBT */
        params_buffer_.get(),
        params_buffer_.byte_size,
        &bind_state_.sbt,
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

} // namespace Agate