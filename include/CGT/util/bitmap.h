//
// Created by 秋鱼头 on 2022/3/28.
//

#pragma once

#include <CGT/common.h>
namespace CGT {

enum class BitmapFormat
{
    UnsignedByte, Float,
};

struct Bitmap
{
    Bitmap() = default;

    Bitmap(int w, int h, int comp, BitmapFormat fmt)
        : w_(w), h_(h), comp_(comp), fmt_(fmt), data_(w * h * comp * GetBytesPerComponent(fmt))
    {
        initGetSetFuncs();
    }

    Bitmap(int w, int h, int d, int comp, BitmapFormat fmt)
        : w_(w), h_(h), d_(d), comp_(comp), fmt_(fmt), data_(w * h * d * comp * GetBytesPerComponent(fmt))
    {
        initGetSetFuncs();
    }

    Bitmap(int w, int h, int comp, BitmapFormat fmt, const void* ptr)
        : w_(w), h_(h), comp_(comp), fmt_(fmt), data_(w * h * comp * GetBytesPerComponent(fmt))
    {
        initGetSetFuncs();
        memcpy(data_.data(), ptr, data_.size());
    }

    int w_ = 0;
    int h_ = 0;
    int d_ = 1;
    int comp_ = 3;
    BitmapFormat fmt_ = BitmapFormat::UnsignedByte;
    std::vector<uint8_t> data_;

    static int GetBytesPerComponent(BitmapFormat fmt)
    {
        if (fmt == BitmapFormat::UnsignedByte) return 1;
        if (fmt == BitmapFormat::Float) return 4;
        return 0;
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
            case BitmapFormat::UnsignedByte: 
                setPixelFunc = &Bitmap::setPixelUnsignedByte;
                getPixelFunc = &Bitmap::getPixelUnsignedByte;
                break;
            case BitmapFormat::Float: 
                setPixelFunc = &Bitmap::setPixelFloat;
                getPixelFunc = &Bitmap::getPixelFloat;
                break;
        }
    }

    void setPixelFloat(int x, int y, const Vector4f& c)
    {
        const int ofs = comp_ * (y * w_ + x);
        auto* data = reinterpret_cast<float*>(data_.data());
        if (comp_ > 0) data[ofs + 0] = c.x;
        if (comp_ > 1) data[ofs + 1] = c.y;
        if (comp_ > 2) data[ofs + 2] = c.z;
        if (comp_ > 3) data[ofs + 3] = c.w;
    }

    Vector4f getPixelFloat(int x, int y) const
    {
        const int ofs = comp_ * (y * w_ + x);
        const auto* data = reinterpret_cast<const float*>(data_.data());
        return {
            comp_ > 0 ? data[ofs + 0] : 0.0f,
            comp_ > 1 ? data[ofs + 1] : 0.0f,
            comp_ > 2 ? data[ofs + 2] : 0.0f,
            comp_ > 3 ? data[ofs + 3] : 0.0f};
    }

    void setPixelUnsignedByte(int x, int y, const Vector4f& c)
    {
        const int ofs = comp_ * (y * w_ + x);
        if (comp_ > 0) data_[ofs + 0] = uint8_t(c.x * 255.0f);
        if (comp_ > 1) data_[ofs + 1] = uint8_t(c.y * 255.0f);
        if (comp_ > 2) data_[ofs + 2] = uint8_t(c.z * 255.0f);
        if (comp_ > 3) data_[ofs + 3] = uint8_t(c.w * 255.0f);
    }

    Vector4f getPixelUnsignedByte(int x, int y) const
    {
        const int ofs = comp_ * (y * w_ + x);
        return {
            comp_ > 0 ? float(data_[ofs + 0]) / 255.0f : 0.0f,
            comp_ > 1 ? float(data_[ofs + 1]) / 255.0f : 0.0f,
            comp_ > 2 ? float(data_[ofs + 2]) / 255.0f : 0.0f,
            comp_ > 3 ? float(data_[ofs + 3]) / 255.0f : 0.0f};
    }
};

/// some useful tools for cubemap
Bitmap ConvertEquirectangularMapToVerticalCross(const Bitmap& b);
Bitmap ConvertVerticalCrossToCubeMapFaces(const Bitmap& b);

} // namespace CGT


 