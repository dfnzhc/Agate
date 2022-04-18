//
// Created by 秋鱼头 on 2022/4/18.
//

#include "Scene.hpp"

namespace Agate {

Scene::~Scene()
{

}

void Scene::addBuffer(const uint64_t buf_size, const void* data)
{

}

void Scene::addImage(int32_t width,
                     int32_t height,
                     int32_t bits_per_component,
                     int32_t num_components,
                     const void* data)
{

}

void Scene::addSampler(cudaTextureAddressMode address_s,
                       cudaTextureAddressMode address_t,
                       cudaTextureFilterMode filter_mode,
                       int32_t image_idx)
{

}

CUdeviceptr Scene::getBuffer(int32_t buffer_index) const
{
    return 0;
}

cudaArray_t Scene::getImage(int32_t image_index) const
{
    return nullptr;
}

cudaTextureObject_t Scene::getSampler(int32_t sampler_index) const
{
    return 0;
}

void Scene::finalize()
{

}

void Scene::cleanup()
{

}

Camera Scene::camera() const
{
    return Camera(float3(), float3(), float3(), 0, 0);
}

void Scene::createContext()
{

}

void Scene::buildMeshAccels(uint32_t triangle_input_flags)
{

}

void Scene::buildInstanceAccel(int rayTypeCount)
{

}

void Scene::createPTXModule()
{

}

void Scene::createProgramGroups()
{

}

void Scene::createPipeline()
{

}

void Scene::createSBT()
{

}
} // namespace Agate