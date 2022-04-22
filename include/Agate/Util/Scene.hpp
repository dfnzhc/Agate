//
// Created by 秋鱼头 on 2022/4/18.
//

#pragma once

#include <Agate/Shader/Matrix.h>
#include <optix_types.h>
#include "CudaBufferView.hpp"
#include "Camera.hpp"
#include "AABB.hpp"
#include "Material.hpp"
namespace Agate {

class Scene
{
public:
    Scene() = default;
    ~Scene();
    
    struct MeshData
    {
        std::string name;
        Matrix4x4 transform;

        std::vector<GenericBufferView> indices;
        std::vector<BufferView<float3>> positions;
        std::vector<BufferView<float3>> normals;
        std::vector<BufferView<float2>> texcoords;

        std::vector<int32_t> material_idx;

        OptixTraversableHandle gas_handle = 0;
        CUdeviceptr d_gas_output = 0;

        AABB object_aabb;
        AABB world_aabb;
    };

    void addCamera(const Camera& camera) { cameras_.push_back(camera); }
    void addMesh(std::shared_ptr<MeshData> mesh) { meshes_.push_back(mesh); }
    void addMaterial(const MaterialData& mtl) { materials_.push_back(mtl); }
    void addBuffer(uint64_t buf_size, const void* data);
    void addImage(
        int32_t width,
        int32_t height,
        int32_t bits_per_component,
        int32_t num_components,
        const void* data
    );
    void addSampler(
        cudaTextureAddressMode address_s,
        cudaTextureAddressMode address_t,
        cudaTextureFilterMode filter_mode,
        int32_t image_idx
    );

    CUdeviceptr getBuffer(int32_t buffer_index) const;
    cudaArray_t getImage(int32_t image_index) const;
    cudaTextureObject_t getSampler(int32_t sampler_index) const;

    void finalize();
    void cleanup();

    Camera camera() const;
    OptixPipeline pipeline() const { return pipeline_; }
    const OptixShaderBindingTable* sbt() const { return &sbt_; }
    OptixTraversableHandle traversableHandle() const { return ias_handle_; }
    AABB aabb() const { return scene_aabb_; }
    OptixDeviceContext context() const { return context_; }
    const std::vector<MaterialData>& materials() const { return materials_; }
    const std::vector<std::shared_ptr<MeshData>>& meshes() const { return meshes_; }

    void createContext();
    void buildMeshAccels(uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    void buildInstanceAccel(int rayTypeCount);

private:
    void createPTXModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();

    // TODO: custom geometry support

    std::vector<Camera> cameras_;
    std::vector<std::shared_ptr<MeshData> > meshes_;
    std::vector<MaterialData> materials_;
    std::vector<CUdeviceptr> buffers_;
    std::vector<cudaTextureObject_t> samplers_;
    std::vector<cudaArray_t> images_;
    AABB scene_aabb_;

    OptixDeviceContext context_ = nullptr;
    OptixShaderBindingTable sbt_ = {};
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    OptixPipeline pipeline_ = nullptr;
    OptixModule ptx_module_ = nullptr;

//    OptixProgramGroup raygen_prog_group = nullptr;
//    OptixProgramGroup radiance_miss_group = nullptr;
//    OptixProgramGroup occlusion_miss_group = nullptr;
//    OptixProgramGroup radiance_hit_group = nullptr;
//    OptixProgramGroup occlusion_hit_group = nullptr;
    OptixTraversableHandle ias_handle_ = 0;
    CUdeviceptr d_ias_output_buffer_ = 0;
};

} // namespace Agate
