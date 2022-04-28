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
#include "ModelData.hpp"

namespace Agate {
class Scene
{
public:
    Scene() = default;
    explicit Scene(OptixDeviceContext& context) : context_{context}{}
    ~Scene();
    
    void addMeshData(const ModelData& model);
    void finalize(int rayTypeCount);
    void cleanup();

    OptixDeviceContext context() const { return context_; }


    void buildMeshAccels(uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    void buildInstanceAccel(int rayTypeCount);

    OptixShaderBindingTable createSBT(const std::vector<OptixProgramGroup>& raygenPGs,
                                      const std::vector<OptixProgramGroup>& missPGs,
                                      const std::vector<OptixProgramGroup>& hitgroupPGs);
private:
    std::vector<std::shared_ptr<MeshData>> meshes_;
    std::vector<MaterialData> materials_;
    
    CudaBuffer raygenRecordBuffer{};
    CudaBuffer missRecordBuffer{};
    CudaBuffer hitgroupRecordBuffer{};

    OptixDeviceContext& context_;
    OptixTraversableHandle ias_handle_ = 0;
    CUdeviceptr d_ias_output_buffer_ = 0;
};

} // namespace Agate
