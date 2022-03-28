//
// Created by 秋鱼头 on 2022/3/28.
//

#include "CGT/util/texture.h"

#include <stb_image.h>

namespace CGT {

/// Draw a checkerboard on a pre-allocated square RGB image.
uint8_t* GenDefaultCheckerboardImage(int* width, int* height)
{
    const int w = 128;
    const int h = 128;

    uint8_t* imgData = (uint8_t*) malloc(w * h * 3); // stbi_load() uses malloc(), so this is safe

    assert(imgData && w > 0 && h > 0);
    assert(w == h);

    if (!imgData || w <= 0 || h <= 0) return nullptr;
    if (w != h) return nullptr;

    for (int i = 0; i < w * h; i++) {
        const int row = i / w;
        const int col = i % w;
        imgData[i * 3 + 0] = imgData[i * 3 + 1] = imgData[i * 3 + 2] = 0xFF * ((row + col) % 2);
    }

    if (width) *width = w;
    if (height) *height = h;

    return imgData;
}

Texture::Texture(const std::string& fileName, TEXTURE_TYPE type)
    : type_(type)
{
    switch (type) {
        case TEXTURE_TYPE::TEX_2D: {
            int w, h, comp;

            uint8_t* img = stbi_load(fileName.c_str(), &w, &h, &comp, STBI_rgb_alpha);
            if (!img)
                throw CGTException("WARNING: could not load image `{}`.", fileName);

            bitmap_ = Bitmap{w, h, comp, BitmapFormat::UnsignedByte, img};
            stbi_image_free((void*) img);
        }
            break;
        case TEXTURE_TYPE::TEX_HDR: {
            int w, h, comp;

            const float* img = stbi_loadf(fileName.c_str(), &w, &h, &comp, STBI_rgb);
            if (!img)
                throw CGTException("WARNING: could not load image `{}`.", fileName);

            bitmap_ = Bitmap{w, h, comp, BitmapFormat::Float, img};
        }
            break;
        default:throw CGTException("No properly texture type implements.");
    }
}

} // namespace CGT