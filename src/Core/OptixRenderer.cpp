//
// Created by 秋鱼头 on 2022/4/11.
//

#include "Agate/Core/OptixRenderer.h"
#include "Agate/Core/Error.h"

#include <optix_function_table_definition.h>
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
    LOG_INFO("创建 Optix 上下文...")

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

void OptixRenderer::SetCompileOptions()
{
    LOG_INFO("创建 Optix module...")

    module_compile_options_.maxRegisterCount = 50;
    module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline_compile_options_ = {};
    // 与 OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING 有什么区别？
    pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options_.usesMotionBlur = false;
    pipeline_compile_options_.numPayloadValues = 2;
    pipeline_compile_options_.numAttributeValues = 2;
    pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_.pipelineLaunchParamsVariableName = "SystemParameters";
    pipeline_link_options_.maxTraceDepth = 2;

}

void OptixRenderer::CreateRaygenPrograms()
{
    LOG_INFO("创建 Raygen 着色器程序...")

    raygenPGs_.resize(1);

    // TODO 载入 ptx
    const std::string ptxCode = ReadPTX("./PTX/RayGeneration.ptx");

    OptixModule moduleRaygeneration;
    OPTIX_CHECK(optixModuleCreateFromPTX(optix_context_,
                                         &module_compile_options_,
                                         &pipeline_compile_options_,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         nullptr, nullptr,
                                         &moduleRaygeneration
    ))

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgDesc.raygen.module = moduleRaygeneration;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        nullptr, nullptr,
                                        &raygenPGs_[0]
    ))
}

void OptixRenderer::CreateMissPrograms()
{
    LOG_INFO("创建 Miss 着色器程序...")

    missPGs_.resize(1);

    // TODO 载入 ptx
    const std::string ptxCode = ReadPTX("./PTX/Miss.ptx");

    OptixModule moduleMiss;
    OPTIX_CHECK(optixModuleCreateFromPTX(optix_context_,
                                         &module_compile_options_,
                                         &pipeline_compile_options_,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         nullptr, nullptr,
                                         &moduleMiss
    ))

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgDesc.raygen.module = moduleMiss;
    pgDesc.raygen.entryFunctionName = "__miss__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        nullptr, nullptr,
                                        &missPGs_[0]
    ))
}

void OptixRenderer::CreateHitgroupPrograms()
{
    LOG_INFO("创建 Hitgroup 着色器程序组...")

    hitgroupPGs_.resize(1);

    // TODO 载入 ptx
    const std::string ptxCode_ClosestHit = ReadPTX("./PTX/ClosestHit.ptx");
    const std::string ptxCode_AnyHit = ReadPTX("./PTX/AnyHit.ptx");

    OptixModule moduleClosestHit;
    OptixModule moduleAnyHit;
    OPTIX_CHECK(optixModuleCreateFromPTX(optix_context_,
                                         &module_compile_options_,
                                         &pipeline_compile_options_,
                                         ptxCode_ClosestHit.c_str(),
                                         ptxCode_ClosestHit.size(),
                                         nullptr, nullptr,
                                         &moduleClosestHit
    ))

    OPTIX_CHECK(optixModuleCreateFromPTX(optix_context_,
                                         &module_compile_options_,
                                         &pipeline_compile_options_,
                                         ptxCode_AnyHit.c_str(),
                                         ptxCode_AnyHit.size(),
                                         nullptr, nullptr,
                                         &moduleAnyHit
    ))

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgDesc.hitgroup.moduleCH = moduleClosestHit;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = moduleAnyHit;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(optix_context_,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        nullptr, nullptr,
                                        &hitgroupPGs_[0]
    ))
}

void OptixRenderer::CreatePipeline()
{
    LOG_INFO("创建着色器流水线...")
    std::vector<OptixProgramGroup> program_groups;
    for (auto pg : raygenPGs_)
        program_groups.push_back(pg);
    for (auto pg : missPGs_)
        program_groups.push_back(pg);
    for (auto pg : hitgroupPGs_)
        program_groups.push_back(pg);

    OPTIX_CHECK(optixPipelineCreate(optix_context_,
                                    &pipeline_compile_options_,
                                    &pipeline_link_options_,
                                    program_groups.data(),
                                    (int) program_groups.size(),
                                    nullptr, nullptr,
                                    &pipeline_
    ));


    // STACK SIZES
    OptixStackSizes stackSizesPipeline = {};
    for (auto& program_group : program_groups) {
        OptixStackSizes stackSizes;

        OPTIX_CHECK(optixProgramGroupGetStackSize(program_group, &stackSizes));

        stackSizesPipeline.cssRG = std::max(stackSizesPipeline.cssRG, stackSizes.cssRG);
        stackSizesPipeline.cssMS = std::max(stackSizesPipeline.cssMS, stackSizes.cssMS);
        stackSizesPipeline.cssCH = std::max(stackSizesPipeline.cssCH, stackSizes.cssCH);
        stackSizesPipeline.cssAH = std::max(stackSizesPipeline.cssAH, stackSizes.cssAH);
        stackSizesPipeline.cssIS = std::max(stackSizesPipeline.cssIS, stackSizes.cssIS);
        stackSizesPipeline.cssCC = std::max(stackSizesPipeline.cssCC, stackSizes.cssCC);
        stackSizesPipeline.dssDC = std::max(stackSizesPipeline.dssDC, stackSizes.dssDC);
    }

    // Temporaries
    const unsigned int cssCCTree =
        stackSizesPipeline.cssCC; // Should be 0. No continuation callables in this pipeline. // maxCCDepth == 0
    const unsigned int cssCHOrMSPlusCCTree = std::max(stackSizesPipeline.cssCH, stackSizesPipeline.cssMS) + cssCCTree;

    // Arguments
    const unsigned int directCallableStackSizeFromTraversal =
        stackSizesPipeline.dssDC; // maxDCDepth == 1 // FromTraversal: DC is invoked from IS or AH.      // Possible stack size optimizations.
    const unsigned int directCallableStackSizeFromState =
        stackSizesPipeline.dssDC; // maxDCDepth == 1 // FromState:     DC is invoked from RG, MS, or CH. // Possible stack size optimizations.
    const unsigned int continuationStackSize = stackSizesPipeline.cssRG + cssCCTree
        + cssCHOrMSPlusCCTree * (std::max(1u, pipeline_link_options_.maxTraceDepth) - 1u) +
        std::min(1u, pipeline_link_options_.maxTraceDepth)
            * std::max(cssCHOrMSPlusCCTree, stackSizesPipeline.cssAH + stackSizesPipeline.cssIS);
    // "The maxTraversableGraphDepth responds to the maximum number of traversables visited when calling optixTrace. 
    // Every acceleration structure and motion transform count as one level of traversal."
    // Render Graph is at maximum: IAS -> GAS
    const unsigned int maxTraversableGraphDepth = 1;

//    OPTIX_CHECK(optixPipelineSetStackSize(pipeline_,
//                                          directCallableStackSizeFromTraversal,
//                                          directCallableStackSizeFromState,
//                                          continuationStackSize,
//                                          maxTraversableGraphDepth));
    OPTIX_CHECK(optixPipelineSetStackSize
                    (/* [in] The pipeline to configure the stack size for */
                        pipeline_,
                        /* [in] The direct stack size requirement for direct
                           callables invoked from IS or AH. */
                        2 * 1024,
                        /* [in] The direct stack size requirement for direct
                           callables invoked from RG, MS, or CH.  */
                        2 * 1024,
                        /* [in] The continuation stack requirement. */
                        2 * 1024,
                        /* [in] The maximum depth of a traversable graph
                           passed to trace. */
                        1));
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
    raygen_records_buf_.Alloc_And_Upload(raygenRecords);
    sbt.raygenRecord = raygen_records_buf_.device_ptr();

    /// ------------------------------------------------------------------
    /// build miss program records
    /// ------------------------------------------------------------------
    std::vector<SbtRecordHeader> missRecords;
    for (auto& missPGs : raygenPGs_) {
        SbtRecordHeader rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs, &rec));
        missRecords.push_back(rec);
    }
    miss_records_buf_.Alloc_And_Upload(missRecords);
    sbt.missRecordBase = miss_records_buf_.device_ptr();
    sbt.missRecordBase = static_cast<int>(missRecords.size());
    sbt.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));


    /// ------------------------------------------------------------------
    /// build hitgroup program records
    /// ------------------------------------------------------------------
    std::vector<SbtRecordHeader> hitgroupRecords;

    // TODO 设置物体信息
    for (auto& missPGs : raygenPGs_) {
        SbtRecordHeader rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs, &rec))
        hitgroupRecords.push_back(rec);
    }
    hitgroup_records_buf_.Alloc_And_Upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroup_records_buf_.device_ptr();
    sbt.hitgroupRecordBase = static_cast<int>(hitgroupRecords.size());
    sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(SbtRecordHeader));
}

} // namespace Agate