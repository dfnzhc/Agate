//
// Created by 秋鱼 on 2022/4/28.
//

#pragma once

#include <Agate/Shader/Matrix.h>
#include <Agate/Shader/CudaBufferView.hpp>
#include <optix_types.h>
#include <Agate/Shader/Material.hpp>
#include "AABB.hpp"
#include "Camera.hpp"

namespace Agate {

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

class ModelData
{
public:
    ModelData() = default;
    ~ModelData();
    
    void cleanup();
    
    void addMeshFromGLTF(std::string_view filename);

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

    CUdeviceptr getBuffer(int32_t buffer_index, int32_t offset = -1) const;
    cudaArray_t getImage(int32_t image_index) const { return images_[image_index]; }
    cudaTextureObject_t getSampler(int32_t sampler_index) const { return samplers_[sampler_index]; }
    Camera camera() const;
    AABB aabb() const { return aabb_; }
    const std::vector<MaterialData>& materials() const { return materials_; }
    const std::vector<std::shared_ptr<MeshData>>& meshes() const { return meshes_; }
    
private:
    std::vector<Camera> cameras_;
    std::vector<std::shared_ptr<MeshData>> meshes_;
    std::vector<MaterialData> materials_;
    std::vector<CUdeviceptr> buffers_;
    std::vector<int32_t> buffer_offsets_;
    std::vector<cudaTextureObject_t> samplers_;
    std::vector<cudaArray_t> images_;
    AABB aabb_;
};

} // namespace Agate
