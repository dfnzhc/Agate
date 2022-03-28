//
// Created by 秋鱼头 on 2022/3/26.
//

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <filesystem>
#include "CGT/common.h"
#include "CGT/util/texture.h"

namespace fs = std::filesystem;

TEST_CASE("Read File", "[ReadFile]")
{
    auto path = fs::current_path();
    
    std::vector<char> data;
    CGT::ReadFile("123.txt", data);
    
}

TEST_CASE("Texture", "[Texture 2D]")
{
    CGT::Texture tex{"awesomeface.png"};
//    tex.bitmap_.SaveHDR("awesomeface.EXR");
    tex.bitmap_.Save("awesomefacetest.png");
    
    CGT::Texture tex1{"immenstadter_horn_2k.hdr", CGT::TEXTURE_TYPE::TEX_HDR};
    tex1.bitmap_.Save("hdrSaveTest.hdr");
    
    auto vcTexture = CGT::ConvertEquirectangularMapToVerticalCross(tex1.bitmap_);
    
    vcTexture.Save("hdrCrossTest.hdr");
}
