//
// Created by 秋鱼头 on 2022/4/18.
//

#pragma once

#include <Agate/Shader/Matrix.h>
#include <optix_types.h>
#include "Agate/Shader/CudaBufferView.hpp"
#include "Camera.hpp"
#include "AABB.hpp"
#include "Agate/Shader/Material.hpp"

#include "CudaBuffer.h"

namespace Agate {
class Scene
{
public:
    Scene() = default;
    explicit Scene(OptixDeviceContext& context) : context_{context}{}
    
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

    CUdeviceptr getBuffer(int32_t buffer_index) const { return buffers_[buffer_index]; }
    cudaArray_t getImage(int32_t image_index) const { return images_[image_index]; }
    cudaTextureObject_t getSampler(int32_t sampler_index) const { return samplers_[sampler_index]; }

    void finalize(int rayTypeCount);
    void cleanup();

    Camera camera() const;
    OptixTraversableHandle traversableHandle() const { return ias_handle_; }
    AABB aabb() const { return scene_aabb_; }
    OptixDeviceContext context() const { return context_; }
    const std::vector<MaterialData>& materials() const { return materials_; }
    const std::vector<std::shared_ptr<MeshData>>& meshes() const { return meshes_; }

    void buildMeshAccels(uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    void buildInstanceAccel(int rayTypeCount);

    OptixShaderBindingTable createSBT(const std::vector<OptixProgramGroup>& raygenPGs,
                                      const std::vector<OptixProgramGroup>& missPGs,
                                      const std::vector<OptixProgramGroup>& hitgroupPGs, int rayTypeCount);
private:
    // TODO: custom geometry support
    std::vector<Camera> cameras_;
    std::vector<std::shared_ptr<MeshData> > meshes_;
    std::vector<MaterialData> materials_;
    std::vector<CUdeviceptr> buffers_;
    std::vector<cudaTextureObject_t> samplers_;
    std::vector<cudaArray_t> images_;
    AABB scene_aabb_;
    
    CudaBuffer raygenRecordBuffer{};
    CudaBuffer missRecordBuffer{};
    CudaBuffer hitgroupRecordBuffer{};

    OptixDeviceContext& context_;
    OptixTraversableHandle ias_handle_ = 0;
    CUdeviceptr d_ias_output_buffer_ = 0;
};

} // namespace Agate
