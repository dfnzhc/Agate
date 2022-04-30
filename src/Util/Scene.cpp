//
// Created by 秋鱼头 on 2022/4/18.
//

#include "include/Agate/Util/Scene.hpp"
#include "include/Agate/Util/Record.hpp"
#include <Agate/Core/Error.h>

namespace Agate {

Scene::~Scene()
{
    cleanup();
}

void Scene::addMeshData(std::shared_ptr<ModelData>& model)
{
    auto meshes = model->meshes();
    auto materials = model->materials();
    
    meshes_.resize(meshes.size());
    materials_.resize(materials.size());
    
    std::copy(meshes.begin(), meshes.end(), meshes_.begin());
    std::copy(materials.begin(), materials.end(), materials.begin());
}

void Scene::finalize(int rayTypeCount)
{
    buildMeshAccels();
    buildInstanceAccel(rayTypeCount);
}

void Scene::cleanup()
{
    
}


void Scene::buildMeshAccels(uint32_t triangle_input_flags)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    for (auto& mesh : meshes_) {
        const size_t num_subMeshes = mesh->indices.size();
        std::vector<OptixBuildInput> buildInputs(num_subMeshes);

        assert(mesh->positions.size() == num_subMeshes &&
            mesh->normals.size() == num_subMeshes &&
            mesh->texcoords.size() == num_subMeshes);

        for (size_t j = 0; j < num_subMeshes; ++j) {
            OptixBuildInput& triangle_input = buildInputs[j];
            memset(&triangle_input, 0, sizeof(OptixBuildInput));
            triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes = mesh->positions[j].byte_stride ?
                                                               mesh->positions[j].byte_stride :
                                                               sizeof(float3);
            triangle_input.triangleArray.numVertices = mesh->positions[j].count;
            triangle_input.triangleArray.vertexBuffers = &(mesh->positions[j].data);
            triangle_input.triangleArray.indexFormat = mesh->indices[j].elmt_byte_size == 2 ?
                                                       OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 :
                                                       OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes = mesh->indices[j].byte_stride ?
                                                              mesh->indices[j].byte_stride :
                                                              mesh->indices[j].elmt_byte_size * 3;
            triangle_input.triangleArray.numIndexTriplets = mesh->indices[j].count / 3;
            triangle_input.triangleArray.indexBuffer = mesh->indices[j].data;
            triangle_input.triangleArray.flags = &triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;
        }

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &accel_options, buildInputs.data(),
                                                 static_cast<unsigned int>(num_subMeshes), &gas_buffer_sizes));

        CudaBuffer d_temp_compactedSizes;
        d_temp_compactedSizes.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = d_temp_compactedSizes.get();

        CudaBuffer d_temp;
        d_temp.alloc(gas_buffer_sizes.tempSizeInBytes);

        CudaBuffer d_temp_output;
        d_temp_output.alloc(gas_buffer_sizes.outputSizeInBytes);

        OPTIX_CHECK(optixAccelBuild(context_,
                                    /* stream */ nullptr,
                                    &accel_options,
                                    buildInputs.data(),
                                    static_cast<unsigned int>(num_subMeshes),
                                    d_temp.get(),
                                    d_temp.byte_size,
                                    d_temp_output.get(),
                                    d_temp_output.byte_size,

                                    &mesh->gas_handle,
                                    &emitProperty,
                                    1));

        CUDA_SYNC_CHECK();

        uint64_t compactedSize;
        d_temp_compactedSizes.download(&compactedSize, 1);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mesh->d_gas_output), compactedSize));
        OPTIX_CHECK(optixAccelCompact(context_,
                                      /*stream:*/nullptr,
                                      mesh->gas_handle,
                                      mesh->d_gas_output,
                                      compactedSize,
                                      &mesh->gas_handle));

        CUDA_SYNC_CHECK();
    }
}

void Scene::buildInstanceAccel(int rayTypeCount)
{
    const size_t num_instances = meshes_.size();

    std::vector<OptixInstance> optix_instances(num_instances);

    unsigned int sbt_offset = 0;
    for (size_t i = 0; i < meshes_.size(); ++i) {
        auto mesh = meshes_[i];
        auto& optix_instance = optix_instances[i];
        memset(&optix_instance, 0, sizeof(OptixInstance));

        optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId = static_cast<unsigned int>(i);
        optix_instance.sbtOffset = sbt_offset;
        optix_instance.visibilityMask = 1;
        optix_instance.traversableHandle = mesh->gas_handle;
        memcpy(optix_instance.transform, mesh->transform.getData(), sizeof(float) * 12);

        // one sbt record per GAS build input per RAY_TYPE
        sbt_offset += static_cast<unsigned int>(mesh->indices.size()) * rayTypeCount;
    }

    const size_t instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
    CUdeviceptr d_instances;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_instances),
        optix_instances.data(),
        instances_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_,
        &accel_options,
        &instance_input,
        1, // num build inputs
        &ias_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer),
        ias_buffer_sizes.tempSizeInBytes
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_ias_output_buffer_),
        ias_buffer_sizes.outputSizeInBytes
    ));

    OPTIX_CHECK(optixAccelBuild(
        context_,
        nullptr,                  // CUDA stream
        &accel_options,
        &instance_input,
        1,                        // num build inputs
        d_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes,
        d_ias_output_buffer_,
        ias_buffer_sizes.outputSizeInBytes,
        &ias_handle_,
        nullptr,            // emitted property list
        0                   // num emitted properties
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
}

OptixShaderBindingTable Scene::createSBT(const std::vector<OptixProgramGroup>& raygenPGs,
                                  const std::vector<OptixProgramGroup>& missPGs,
                                  const std::vector<OptixProgramGroup>& hitgroupPGs)
{
    OptixShaderBindingTable sbt = {};
    
    std::vector<RaygenRecord> raygenRecords;
    for (auto raygenPG : raygenPGs) {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rec));
        /// ...
        raygenRecords.push_back(rec);
    }
    raygenRecordBuffer.allocAndUpload(raygenRecords);
    sbt.raygenRecord = raygenRecordBuffer.get();
    
    std::vector<MissRecord> missRecords;
    for (auto missPG : missPGs) {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &rec));
        /// ...
        missRecords.push_back(rec);
    }
    missRecordBuffer.allocAndUpload(missRecords);
    sbt.missRecordBase = missRecordBuffer.get();
    sbt.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(MissRecord));
    sbt.missRecordCount = static_cast<unsigned int>(missRecords.size());
    
    std::vector<HitGroupRecord> hitgroupRecords;
    for (const auto& mesh : meshes_) {
        for (size_t i = 0; i < mesh->material_idx.size(); ++i) {
            for (auto hitgroupPG : hitgroupPGs) {
                HitGroupRecord rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, &rec));
                rec.data.geometry_data.triangle_mesh.positions = mesh->positions[i];
                rec.data.geometry_data.triangle_mesh.normals = mesh->normals[i];
                rec.data.geometry_data.triangle_mesh.texcoords = mesh->texcoords[i];
                rec.data.geometry_data.triangle_mesh.indices = mesh->indices[i];
                
                const int32_t mat_idx = mesh->material_idx[i];
                if (mat_idx >= 0) {
                    rec.data.material_data = materials_[mat_idx];
                }
                else
                    rec.data.material_data = MaterialData{};
                
                hitgroupRecords.push_back(rec);
            }
        }
    }
    hitgroupRecordBuffer.allocAndUpload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordBuffer.get();
    sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(HitGroupRecord));
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroupRecords.size());
    
    return sbt;
}


} // namespace Agate