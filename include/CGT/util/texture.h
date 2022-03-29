//
// Created by 秋鱼头 on 2022/3/28.
//

#pragma once

#include "bitmap.h"

namespace CGT {
enum class TEXTURE_TYPE{
    TEX_2D, TEX_HDR
};

struct Texture
{
    explicit Texture(const std::string& fileName, TEXTURE_TYPE type = TEXTURE_TYPE::TEX_2D);
    
    Bitmap bitmap_;
    TEXTURE_TYPE type_ = TEXTURE_TYPE::TEX_2D;
};
} // namespace CGT


 