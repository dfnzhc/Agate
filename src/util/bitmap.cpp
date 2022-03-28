//
// Created by 秋鱼头 on 2022/3/28.
//

#include "CGT/util/bitmap.h"
#include <stb_image_write.h>

#include <tbb/tbb.h>
#include <CGT/util/uMath.h>

namespace CGT {
void Bitmap::SaveHDR(const char* filename) const
{
    int ret = stbi_write_hdr(filename, w_, h_, comp_, (const float*) data_.data());
    if (ret == 0) {
        throw CGTException("Bitmap::SaveHDR(): Could not save EXR file {}.", filename);
    }
}

void Bitmap::SavePNG(const char* filename) const
{
    uint8_t* rgba = new uint8_t[comp_ * w_ * h_];
    uint8_t* dst = rgba;
    for (int i = 0; i < h_; ++i) {
        for (int j = 0; j < w_; ++j) {
            Vector4f p = GetPixel(j, i);
            for (int c = 0; c < comp_; ++c)
                dst[c] = (uint8_t) Clamp(255.f * p[c], 0.f, 255.f);
            dst += comp_;
        }
    }

    int ret = stbi_write_png(filename, w_, h_, comp_, rgba, comp_ * (int) w_);
    if (ret == 0) {
        throw CGTException("Bitmap::SavePNG(): Could not save PNG file {}.", filename);
    }

    delete[] rgba;
}

void Bitmap::Save(const std::string& filename) const
{
    switch (fmt_) {
        case BitmapFormat::UnsignedByte:
            SavePNG(filename.c_str());
            break;
        case BitmapFormat::Float:SaveHDR(filename.c_str());
            break;
    }
}

Vector3f texCoordsToXYZ(int i, int j, int faceID, int faceSize)
{
    const float Pi = 2.0f * float(i) / faceSize;
    const float Pj = 2.0f * float(j) / faceSize;

    if (faceID == 0) return {Pi - 1.0f, -1.0f, 1.0f - Pj};
    if (faceID == 1) return {1.0f, Pi - 1.0f, 1.0f - Pj};
    if (faceID == 2) return {1.0f - Pi, 1.0f, 1.0f - Pj};
    if (faceID == 3) return {-1.0f, 1.0 - Pi, 1.0 - Pj};
    if (faceID == 4) return {Pj - 1.0f, Pi - 1.0f, 1.0f};
    if (faceID == 5) return {1.0f - Pj, Pi - 1.0f, -1.0f};

    return {};
}

Bitmap ConvertEquirectangularMapToVerticalCross(const Bitmap& b)
{
    const int faceSize = b.w_ / 4;

    const int w = faceSize * 4;
    const int h = faceSize * 3;

    Bitmap result(w, h, b.comp_, b.fmt_);

    const Vector2f kFaceOffsets[] =
        {
            Vector2f(0, faceSize),
            Vector2f(faceSize, faceSize),
            Vector2f(2 * faceSize, faceSize),
            Vector2f(3 * faceSize, faceSize),
            Vector2f(faceSize, 0),
            Vector2f(faceSize, 2 * faceSize)
        };

    const int clampW = b.w_ - 1;
    const int clampH = b.h_ - 1;

    for (int face = 0; face != 6; face++) {
        tbb::parallel_for(0, faceSize,
                          [&result, &kFaceOffsets, &b, face, faceSize, clampW, clampH](int i)
                          {
                              for (int j = 0; j != faceSize; j++) {
                                  const Vector3f P = texCoordsToXYZ(i, j, face, faceSize);
                                  const float R = hypot(P.x, P.y);
                                  const float theta = atan2(P.y, P.x);
                                  const float phi = atan2(P.z, R);
                                  //	float point source coordinates
                                  const auto Uf = float(2.0f * faceSize * (theta + PI) / PI);
                                  const auto Vf = float(2.0f * faceSize * (PI / 2.0f - phi) / PI);
                                  // 4-samples for bilinear interpolation
                                  const int U1 = glm::clamp(int(floor(Uf)), 0, clampW);
                                  const int V1 = glm::clamp(int(floor(Vf)), 0, clampH);
                                  const int U2 = glm::clamp(U1 + 1, 0, clampW);
                                  const int V2 = glm::clamp(V1 + 1, 0, clampH);
                                  // fractional part
                                  const float s = Uf - U1;
                                  const float t = Vf - V1;
                                  // fetch 4-samples
                                  const Vector4f A = b.GetPixel(U1, V1);
                                  const Vector4f B = b.GetPixel(U2, V1);
                                  const Vector4f C = b.GetPixel(U1, V2);
                                  const Vector4f D = b.GetPixel(U2, V2);
                                  // bilinear interpolation
                                  const Vector4f color = A * (1 - s) * (1 - t) + B * (s) * (1 - t) + C * (1 - s) * t + D
                                      * (s) * (t);
                                  result.SetPixel(i + kFaceOffsets[face].x, j + kFaceOffsets[face].y, color);
                              }
                          });
    }

    return result;
}

Bitmap ConvertVerticalCrossToCubeMapFaces(const Bitmap& b)
{
    const int faceWidth = b.w_ / 4;
    const int faceHeight = b.h_ / 3;

    Bitmap cubemap(faceWidth, faceHeight, 6, b.comp_, b.fmt_);

    const uint8_t* src = b.data_.data();
    uint8_t* dst = cubemap.data_.data();

    /*
          ------
          | +Y |
     ---------------------
     | -X | -Z | +X | +Z |
     ---------------------
          | -Y |
          ------
    */
    const int pixelSize = cubemap.comp_ * Bitmap::GetBytesPerComponent(cubemap.fmt_);
    for (int face = 0; face != 6; ++face) {
        for (int j = 0; j != faceHeight; ++j) {
            for (int i = 0; i != faceWidth; ++i) {
                int x = 0;
                int y = 0;

                switch (face) {
                    // GL_TEXTURE_CUBE_MAP_POSITIVE_X
                    case 0:x = i;
                        y = faceHeight + j;
                        break;

                        // GL_TEXTURE_CUBE_MAP_NEGATIVE_X
                    case 1:x = 2 * faceWidth + i;
                        y = 1 * faceHeight + j;
                        break;

                        // GL_TEXTURE_CUBE_MAP_POSITIVE_Y
                    case 2:x = 2 * faceWidth - (i + 1);
                        y = 1 * faceHeight - (j + 1);
                        break;

                        // GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
                    case 3:x = 2 * faceWidth - (i + 1);
                        y = 3 * faceHeight - (j + 1);
                        break;

                        // GL_TEXTURE_CUBE_MAP_POSITIVE_Z
                    case 4:x = 3 * faceWidth + i;
                        y = 1 * faceHeight + j;
                        break;

                        // GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
                    case 5:x = faceWidth + i;
                        y = faceHeight + j;
                        break;
                }

                memcpy(dst, src + (y * b.w_ + x) * pixelSize, pixelSize);

                dst += pixelSize;
            }
        }
    }

    return cubemap;
}
} // namespace CGT