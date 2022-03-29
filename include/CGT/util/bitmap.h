//
// Created by 秋鱼头 on 2022/3/28.
//

#pragma once

#include <CGT/common.h>
#include <dxgiformat.h>
namespace CGT {

enum class BitmapFormat : uint32_t
{
    UnsignedByte, Float,
};

struct BitmapInfo
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t comp = 3;
    uint32_t mipMapCount = 0;
};

struct Bitmap
{
    Bitmap() = default;

    Bitmap(int w, int h, int comp, BitmapFormat fmt)
        : fmt_(fmt), data_(w * h * comp * GetBytesPerComponent(fmt))
    {
        initGetSetFuncs();

        info_.width = w;
        info_.height = h;
        info_.comp = comp;
        info_.mipMapCount = GetMipCount(w, h);
    }

    Bitmap(int w, int h, int d, int comp, BitmapFormat fmt)
        : fmt_(fmt), data_(w * h * d * comp * GetBytesPerComponent(fmt))
    {
        initGetSetFuncs();

        info_.width = w;
        info_.height = h;
        info_.depth = d;
        info_.comp = comp;
        info_.mipMapCount = GetMipCount(w, h);
    }

    Bitmap(int w, int h, int comp, BitmapFormat fmt, const void* ptr)
        : Bitmap(w, h, comp, fmt)
    {
        memcpy(data_.data(), ptr, data_.size());
    }

    BitmapInfo info_;
    BitmapFormat fmt_ = BitmapFormat::UnsignedByte;
    std::vector<uint8_t> data_;
    float cutoff_ = 1.0f;
    float alpha_test_coverage_ = 1.0;
    std::vector<Bitmap> mipmaps_;

    constexpr static int GetBytesPerComponent(BitmapFormat fmt)
    {
        if (fmt == BitmapFormat::UnsignedByte) return 1;
        if (fmt == BitmapFormat::Float) return 4;
        return 0;
    }

    constexpr static uint32_t GetMipCount(uint32_t width, uint32_t height)
    {
        uint32_t mipmap_count = 0;
        while (true) {
            mipmap_count += 1;
            if (width > 1) width >>= 1;
            if (height > 1) height >>= 1;
            if (width == 1 && height == 1)
                break;
        }

        return mipmap_count;
    }

    void SetPixel(int x, int y, const Vector4f& c)
    {
        (*this.*setPixelFunc)(x, y, c);
    }

    Vector4f GetPixel(int x, int y) const
    {
        return ((*this.*getPixelFunc)(x, y));
    }

    void Save(const std::string& filename) const;

    void SetAlphaTest(float cutoff)
    {
        this->cutoff_ = cutoff;
        if (this->cutoff_ < 1.0f)
            alpha_test_coverage_ = GetAlphaCoverage(info_.width, info_.height, 1.0, (int) (255 * cutoff));
    }

private:
    using setPixel_t = void (Bitmap::*)(int, int, const Vector4f&);
    using getPixel_t = Vector4f(Bitmap::*)(int, int) const;
    setPixel_t setPixelFunc = &Bitmap::setPixelUnsignedByte;
    getPixel_t getPixelFunc = &Bitmap::getPixelUnsignedByte;

    void SaveHDR(const char* filename) const;

    void SavePNG(const char* filename) const;

    void initGetSetFuncs()
    {
        switch (fmt_) {
            case BitmapFormat::UnsignedByte:setPixelFunc = &Bitmap::setPixelUnsignedByte;
                getPixelFunc = &Bitmap::getPixelUnsignedByte;
                break;
            case BitmapFormat::Float:setPixelFunc = &Bitmap::setPixelFloat;
                getPixelFunc = &Bitmap::getPixelFloat;
                break;
        }
    }

    void setPixelFloat(int x, int y, const Vector4f& c)
    {
        const int ofs = info_.comp * (y * info_.width + x);
        auto* data = reinterpret_cast<float*>(data_.data());
        if (info_.comp > 0) data[ofs + 0] = c.x;
        if (info_.comp > 1) data[ofs + 1] = c.y;
        if (info_.comp > 2) data[ofs + 2] = c.z;
        if (info_.comp > 3) data[ofs + 3] = c.w;
    }

    Vector4f getPixelFloat(int x, int y) const
    {
        const int ofs = info_.comp * (y * info_.width + x);
        const auto* data = reinterpret_cast<const float*>(data_.data());
        return {
            info_.comp > 0 ? data[ofs + 0] : 0.0f,
            info_.comp > 1 ? data[ofs + 1] : 0.0f,
            info_.comp > 2 ? data[ofs + 2] : 0.0f,
            info_.comp > 3 ? data[ofs + 3] : 0.0f};
    }

    void setPixelUnsignedByte(int x, int y, const Vector4f& c)
    {
        const int ofs = info_.comp * (y * info_.width + x);
        if (info_.comp > 0) data_[ofs + 0] = uint8_t(c.x * 255.0f);
        if (info_.comp > 1) data_[ofs + 1] = uint8_t(c.y * 255.0f);
        if (info_.comp > 2) data_[ofs + 2] = uint8_t(c.z * 255.0f);
        if (info_.comp > 3) data_[ofs + 3] = uint8_t(c.w * 255.0f);
    }

    Vector4f getPixelUnsignedByte(int x, int y) const
    {
        const int ofs = info_.comp * (y * info_.width + x);
        return {
            info_.comp > 0 ? float(data_[ofs + 0]) / 255.0f : 0.0f,
            info_.comp > 1 ? float(data_[ofs + 1]) / 255.0f : 0.0f,
            info_.comp > 2 ? float(data_[ofs + 2]) / 255.0f : 0.0f,
            info_.comp > 3 ? float(data_[ofs + 3]) / 255.0f : 0.0f};
    }

    float GetAlphaCoverage(uint32_t width, uint32_t height, float scale, int cutoff) const
    {
        double val = 0;

        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                auto pixel = GetPixel(x, y);

                int alpha = (int) (scale * pixel[3]);
                if (alpha > 255) alpha = 255;
                if (alpha <= cutoff)
                    continue;

                val += alpha;
            }
        }

        return (float) (val / (height * width * 255));
    }
};

/// some useful tools for cubemap
Bitmap ConvertEquirectangularMapToVerticalCross(const Bitmap& b);
Bitmap ConvertVerticalCrossToCubeMapFaces(const Bitmap& b);

} // namespace CGT


 