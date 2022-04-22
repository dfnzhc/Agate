﻿//
// Created by 秋鱼头 on 2022/4/18.
//

#include "Scene.hpp"
#include "Quaternion.hpp"
#include <Agate/Core/Error.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined( WIN32 )
#    pragma warning( push )
#    pragma warning( disable : 4267 )
#endif
#include <tiny_gltf.h>
#if defined( WIN32 )
#    pragma warning( pop )
#endif

namespace Agate {

float3 make_float3_from_double(double x, double y, double z)
{
    return make_float3(static_cast<float>( x ), static_cast<float>( y ), static_cast<float>( z ));
}

float4 make_float4_from_double(double x, double y, double z, double w)
{
    return make_float4(static_cast<float>( x ),
                       static_cast<float>( y ),
                       static_cast<float>( z ),
                       static_cast<float>( w ));
}

template<typename T>
BufferView<T> BufferViewFromGLTF(const tinygltf::Model& model, Scene& scene, const int32_t accessor_idx)
{
    if (accessor_idx == -1)
        return BufferView<T>();

    const auto& gltf_accessor = model.accessors[accessor_idx];
    const auto& gltf_buffer_view = model.bufferViews[gltf_accessor.bufferView];

    const int32_t elmt_byte_size =
        gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 2 :
        gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT ? 4 :
        gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT ? 4 :
        0;
    if (!elmt_byte_size) {
        throw AgateException("gltf accessor component type not supported");
    }

    const CUdeviceptr buffer_base = scene.getBuffer(gltf_buffer_view.buffer);
    BufferView<T> buffer_view;
    buffer_view.data = buffer_base + gltf_buffer_view.byteOffset + gltf_accessor.byteOffset;
    buffer_view.byte_stride = static_cast<uint16_t>( gltf_buffer_view.byteStride );
    buffer_view.count = static_cast<uint32_t>( gltf_accessor.count );
    buffer_view.elmt_byte_size = static_cast<uint16_t>( elmt_byte_size );

    return buffer_view;
}

void ProcessGLTFNode(
    Scene& scene,
    const tinygltf::Model& model,
    const tinygltf::Node& gltf_node,
    const Matrix4x4& parent_matrix
)
{
    const Matrix4x4 translation = gltf_node.translation.empty() ?
                                  Matrix4x4::identity() :
                                  Matrix4x4::translate(make_float3_from_double(
                                      gltf_node.translation[0],
                                      gltf_node.translation[1],
                                      gltf_node.translation[2]
                                  ));

    const Matrix4x4 rotation = gltf_node.rotation.empty() ?
                               Matrix4x4::identity() :
                               Quaternion(
                                   static_cast<float>( gltf_node.rotation[3] ),
                                   static_cast<float>( gltf_node.rotation[0] ),
                                   static_cast<float>( gltf_node.rotation[1] ),
                                   static_cast<float>( gltf_node.rotation[2] )
                               ).rotationMatrix();

    const Matrix4x4 scale = gltf_node.scale.empty() ?
                            Matrix4x4::identity() :
                            Matrix4x4::scale(make_float3_from_double(
                                gltf_node.scale[0],
                                gltf_node.scale[1],
                                gltf_node.scale[2]
                            ));

    std::vector<float> gltf_matrix;
    for (double x : gltf_node.matrix)
        gltf_matrix.push_back(static_cast<float>( x ));
    const Matrix4x4 matrix = gltf_node.matrix.empty() ?
                             Matrix4x4::identity() :
                             Matrix4x4(reinterpret_cast<float*>( gltf_matrix.data())).transpose();

    const Matrix4x4 node_xform = parent_matrix * matrix * translation * rotation * scale;

    if (gltf_node.camera != -1) {
        const auto& gltf_camera = model.cameras[gltf_node.camera];
        LOG_INFO("Processing camera '{}' \n\ttype: {}", gltf_camera.name, gltf_camera.type);

        if (gltf_camera.type != "perspective") {
            LOG_WARN("\tskipping non-perpective camera");
            return;
        }

        const float3 eye = make_float3(node_xform * make_float4_from_double(0.0f, 0.0f, 0.0f, 1.0f));
        const float3 up = make_float3(node_xform * make_float4_from_double(0.0f, 1.0f, 0.0f, 0.0f));
        const float yfov = static_cast<float>( gltf_camera.perspective.yfov ) * 180.0f / static_cast<float>( M_PI );

        LOG_TRACE("\teye   : {}, {}, {}", eye.x, eye.y, eye.z);
        LOG_TRACE("\tup    : {}, {}, {}", up.x, up.y, up.z);
        LOG_TRACE("\tfov   : {}", yfov);
        LOG_TRACE("\taspect: {}", gltf_camera.perspective.aspectRatio);

        Camera camera;
        camera.fovY = yfov;
        camera.aspectRatio = static_cast<float>(gltf_camera.perspective.aspectRatio);
        camera.eye = eye;
        camera.up = up;
        scene.addCamera(camera);
    } else if (gltf_node.mesh != -1) {
        const auto& gltf_mesh = model.meshes[gltf_node.mesh];
        LOG_INFO("Processing glTF mesh: '{}'", gltf_mesh.name);
        LOG_TRACE("\tNum mesh primitive groups: {}", gltf_mesh.primitives.size());
        for (auto& gltf_primitive : gltf_mesh.primitives) {
            if (gltf_primitive.mode != TINYGLTF_MODE_TRIANGLES) // Ignore non-triangle meshes
            {
                LOG_WARN("\tNon-triangle primitive: skipping\n");
                continue;
            }

            auto mesh = std::make_shared<Scene::MeshData>();
            scene.addMesh(mesh);

            mesh->name = gltf_mesh.name;
            mesh->indices.push_back(BufferViewFromGLTF<uint32_t>(model, scene, gltf_primitive.indices));
            mesh->material_idx.push_back(gltf_primitive.material);
            mesh->transform = node_xform;
            LOG_TRACE("\t\tNum triangles: {}", mesh->indices.back().count / 3);

            assert(gltf_primitive.attributes.find("POSITION") != gltf_primitive.attributes.end());
            const int32_t pos_accessor_idx = gltf_primitive.attributes.at("POSITION");
            mesh->positions.push_back(BufferViewFromGLTF<float3>(model, scene, pos_accessor_idx));

            const auto& pos_gltf_accessor = model.accessors[pos_accessor_idx];
            mesh->object_aabb = AABB(
                make_float3_from_double(
                    pos_gltf_accessor.minValues[0],
                    pos_gltf_accessor.minValues[1],
                    pos_gltf_accessor.minValues[2]
                ),
                make_float3_from_double(
                    pos_gltf_accessor.maxValues[0],
                    pos_gltf_accessor.maxValues[1],
                    pos_gltf_accessor.maxValues[2]
                ));
            mesh->world_aabb = mesh->object_aabb;
            mesh->world_aabb.transform(node_xform);

            auto normal_accessor_iter = gltf_primitive.attributes.find("NORMAL");
            if (normal_accessor_iter != gltf_primitive.attributes.end()) {
                LOG_TRACE("\t\tHas vertex normals: true");
                mesh->normals.push_back(BufferViewFromGLTF<float3>(model, scene, normal_accessor_iter->second));
            } else {
                LOG_TRACE("\t\tHas vertex normals: false");
                mesh->normals.push_back(BufferViewFromGLTF<float3>(model, scene, -1));
            }

            auto texcoord_accessor_iter = gltf_primitive.attributes.find("TEXCOORD_0");
            if (texcoord_accessor_iter != gltf_primitive.attributes.end()) {
                LOG_TRACE("\t\tHas texcoords: true");
                mesh->texcoords.push_back(BufferViewFromGLTF<float2>(model, scene, texcoord_accessor_iter->second));
            } else {
                LOG_TRACE("\t\tHas texcoords: false");
                mesh->texcoords.push_back(BufferViewFromGLTF<float2>(model, scene, -1));
            }
        }
    } else if (!gltf_node.children.empty()) {
        for (int32_t child : gltf_node.children) {
            ProcessGLTFNode(scene, model, model.nodes[child], node_xform);
        }
    }
}

void LoadScene(const std::string& filename, Scene& scene)
{
    scene.cleanup();

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!warn.empty()) {
        LOG_WARN("glTF Warning: {}", warn);
    }
    if (!ret) {
        throw AgateException("Failed to load GLTF {} : {}.", filename, err);
    }

    // -------------
    // Buffer data
    // -------------
    for (const auto& gltf_buffer : model.buffers) {
        const uint64_t buf_size = gltf_buffer.data.size();
        LOG_INFO("Processing glTF buffer '{}' \n\tbyte size: {} \n\turi      : {}",
                 gltf_buffer.name,
                 buf_size,
                 gltf_buffer.uri);

        scene.addBuffer(buf_size, gltf_buffer.data.data());
    }

    // ---------
    // Images
    // ---------
    for (const auto& gltf_image : model.images) {
        LOG_INFO("Processing image '{}' \n\t({}x{})x{} \n\tbits: {}",
                 gltf_image.name,
                 gltf_image.width,
                 gltf_image.height,
                 gltf_image.component,
                 gltf_image.bits);

        assert(gltf_image.component == 4);
        assert(gltf_image.bits == 8 || gltf_image.bits == 16);

        scene.addImage(
            gltf_image.width,
            gltf_image.height,
            gltf_image.bits,
            gltf_image.component,
            gltf_image.image.data()
        );
    }

    // -----------
    // Textures
    // -----------
    for (const auto& gltf_texture : model.textures) {
        if (gltf_texture.sampler == -1) {
            scene.addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, gltf_texture.source);
            continue;
        }

        const auto& gltf_sampler = model.samplers[gltf_texture.sampler];
        const cudaTextureAddressMode address_s = gltf_sampler.wrapS == GL_CLAMP_TO_EDGE ? cudaAddressModeClamp :
                                                 gltf_sampler.wrapS == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
                                                 cudaAddressModeWrap;
        const cudaTextureAddressMode address_t = gltf_sampler.wrapT == GL_CLAMP_TO_EDGE ? cudaAddressModeClamp :
                                                 gltf_sampler.wrapT == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
                                                 cudaAddressModeWrap;
        const cudaTextureFilterMode filter = gltf_sampler.minFilter == GL_NEAREST ? cudaFilterModePoint :
                                             cudaFilterModeLinear;
        scene.addSampler(address_s, address_t, filter, gltf_texture.source);
    }

    // ----------
    // Materials
    // ----------
    for (auto& gltf_material : model.materials) {
        LOG_INFO("Processing glTF material: '{}'", gltf_material.name);
        MaterialData mtl;

        {
            const auto base_color_it = gltf_material.values.find("baseColorFactor");
            if (base_color_it != gltf_material.values.end()) {
                const tinygltf::ColorValue c = base_color_it->second.ColorFactor();
                mtl.base_color = make_float4_from_double(c[0], c[1], c[2], c[3]);
                LOG_TRACE("\tBase color: ({}, {}, {})", mtl.base_color.x, mtl.base_color.y, mtl.base_color.z);
            } else {

                LOG_TRACE("\tUsing default base color factor");
            }
        }

        {
            const auto base_color_it = gltf_material.values.find("baseColorTexture");
            if (base_color_it != gltf_material.values.end()) {
                LOG_TRACE("\tFound base color texture: {}", base_color_it->second.TextureIndex());
                mtl.base_color_tex = scene.getSampler(base_color_it->second.TextureIndex());
            } else {
                LOG_TRACE("\tNo base color texture");
            }
        }

        {
            const auto roughness_it = gltf_material.values.find("roughnessFactor");
            if (roughness_it != gltf_material.values.end()) {
                mtl.roughness = static_cast<float>( roughness_it->second.Factor());
                LOG_TRACE("\tRougness:  {}", mtl.roughness);
            } else {
                LOG_TRACE("\tUsing default roughness factor");
            }
        }

        {
            const auto metallic_it = gltf_material.values.find("metallicFactor");
            if (metallic_it != gltf_material.values.end()) {
                mtl.metallic = static_cast<float>( metallic_it->second.Factor());
                LOG_TRACE("\tMetallic:  {}", mtl.metallic);
            } else {
                LOG_TRACE("\tUsing default metallic factor");
            }
        }

        {
            const auto metallic_roughness_it = gltf_material.values.find("metallicRoughnessTexture");
            if (metallic_roughness_it != gltf_material.values.end()) {
                LOG_TRACE("\tFound metallic roughness texture: {}", metallic_roughness_it->second.TextureIndex());
                mtl.metallic_roughness_tex = scene.getSampler(metallic_roughness_it->second.TextureIndex());
            } else {
                LOG_TRACE("\tNo metllic roughness texture");
            }
        }

        {
            const auto normal_it = gltf_material.additionalValues.find("normalTexture");
            if (normal_it != gltf_material.additionalValues.end()) {
                LOG_TRACE("\tFound normal texture: {}", normal_it->second.TextureIndex());
                mtl.normal_tex = scene.getSampler(normal_it->second.TextureIndex());
            } else {
                LOG_TRACE("\tNo normal texture");
            }
        }

        {
            const auto occlusion_it = gltf_material.additionalValues.find("occlusionTexture");
            if (occlusion_it != gltf_material.additionalValues.end()) {
                LOG_TRACE("\tFound occlusion texture: {}", occlusion_it->second.TextureIndex());
                mtl.occlusion_tex = scene.getSampler(occlusion_it->second.TextureIndex());
            } else {
                LOG_TRACE("\tNo normal texture");
            }
        }

        {
            const auto emissive_it = gltf_material.additionalValues.find("emissiveTexture");
            if (emissive_it != gltf_material.additionalValues.end()) {
                LOG_TRACE("\tFound emissive texture: {}", emissive_it->second.TextureIndex());
                mtl.emissive_tex = scene.getSampler(emissive_it->second.TextureIndex());
            } else {
                LOG_TRACE("\tNo emissive texture");
            }
        }

        scene.addMaterial(mtl);
    }

    // -------------
    // Process nodes
    // -------------
    std::vector<int32_t> root_nodes(model.nodes.size(), 1);
    for (auto& gltf_node : model.nodes)
        for (int32_t child : gltf_node.children)
            root_nodes[child] = 0;

    for (size_t i = 0; i < root_nodes.size(); ++i) {
        if (!root_nodes[i])
            continue;
        auto& gltf_node = model.nodes[i];

        ProcessGLTFNode(scene, model, gltf_node, Matrix4x4::identity());
    }
}

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