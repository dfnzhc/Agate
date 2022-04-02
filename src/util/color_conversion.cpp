//
// Created by 秋鱼头 on 2022/4/2.
//

#include "CGT/misc/color_conversion.h"

namespace CGT {
float ColorSpacePrimaries[4][4][2] = {
    //Rec709
    {
        0.3127f, 0.3290f, // White point
        0.64f, 0.33f, // Red point
        0.30f, 0.60f, // Green point
        0.15f, 0.06f, // Blue point
    },
    //P3
    {
        0.3127f, 0.3290f, // White point
        0.680f, 0.320f, // Red point
        0.265f, 0.690f, // Green point
        0.150f, 0.060f, // Blue point
    },
    //Rec2020
    {
        0.3127f, 0.3290f, // White point
        0.708f, 0.292f, // Red point
        0.170f, 0.797f, // Green point
        0.131f, 0.046f, // Blue point
    },
    //Display Specific zeroed out now Please fill them in once you query them and want to use them for matrix calculations
    {
        0.0f, 0.0f, // White point
        0.0f, 0.0f, // Red point
        0.0f, 0.0f, // Green point
        0.0f, 0.0f // Blue point
    }
};

Matrix4x4f CalculateRGBToXYZMatrix(float xw,
                                   float yw,
                                   float xr,
                                   float yr,
                                   float xg,
                                   float yg,
                                   float xb,
                                   float yb,
                                   bool scaleLumaFlag)
{
    float Xw = xw / yw;
    float Yw = 1;
    float Zw = (1 - xw - yw) / yw;

    float Xr = xr / yr;
    float Yr = 1;
    float Zr = (1 - xr - yr) / yr;

    float Xg = xg / yg;
    float Yg = 1;
    float Zg = (1 - xg - yg) / yg;

    float Xb = xb / yb;
    float Yb = 1;
    float Zb = (1 - xb - yb) / yb;

    Matrix4x4f XRGB = Matrix4x4f(
        Vector4f(Xr, Xg, Xb, 0),
        Vector4f(Yr, Yg, Yb, 0),
        Vector4f(Zr, Zg, Zb, 0),
        Vector4f(0, 0, 0, 1));
    Matrix4x4f XRGBInverse = glm::inverse(XRGB);

    Vector4f referenceWhite = Vector4f(Xw, Yw, Zw, 0);
    Vector4f SRGB = glm::transpose(XRGBInverse) * referenceWhite;

    Matrix4x4f RegularResult = Matrix4x4f(
        Vector4f(SRGB.x * Xr, SRGB.y * Xg, SRGB.z * Xb, 0),
        Vector4f(SRGB.x * Yr, SRGB.y * Yg, SRGB.z * Yb, 0),
        Vector4f(SRGB.x * Zr, SRGB.y * Zg, SRGB.z * Zb, 0),
        Vector4f(0, 0, 0, 1));

    if (!scaleLumaFlag)
        return RegularResult;

    Vector3f Scale{100, 100, 100};
    Matrix4x4f Result = glm::scale(RegularResult, Scale);
    return Result;
}

Matrix4x4f CalculateXYZToRGBMatrix(float xw,
                                   float yw,
                                   float xr,
                                   float yr,
                                   float xg,
                                   float yg,
                                   float xb,
                                   float yb,
                                   bool scaleLumaFlag)
{
    auto RGBToXYZ = CalculateRGBToXYZMatrix(xw, yw, xr, yr, xg, yg, xb, yb, scaleLumaFlag);
    return glm::inverse(RGBToXYZ);
}

void FillDisplaySpecificPrimaries(float xw, float yw, float xr, float yr, float xg, float yg, float xb, float yb)
{
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_WHITE][ColorPrimariesCoordinates_X] = xw;
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_WHITE][ColorPrimariesCoordinates_Y] = yw;

    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_RED][ColorPrimariesCoordinates_X] = xr;
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_RED][ColorPrimariesCoordinates_Y] = yr;

    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_GREEN][ColorPrimariesCoordinates_X] = xg;
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_GREEN][ColorPrimariesCoordinates_Y] = yg;

    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_BLUE][ColorPrimariesCoordinates_X] = xb;
    ColorSpacePrimaries[ColorSpace_Display][ColorPrimaries_BLUE][ColorPrimariesCoordinates_Y] = yb;
}

void SetupGamutMapperMatrices(ColorSpace gamutIn, ColorSpace gamutOut, Matrix4x4f* inputToOutputRecMatrix)
{
    Matrix4x4f intputGamut_To_XYZ = CalculateRGBToXYZMatrix(
        ColorSpacePrimaries[gamutIn][ColorPrimaries_WHITE][ColorPrimariesCoordinates_X],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_WHITE][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_RED][ColorPrimariesCoordinates_X],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_RED][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_GREEN][ColorPrimariesCoordinates_X],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_GREEN][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_BLUE][ColorPrimariesCoordinates_X],
        ColorSpacePrimaries[gamutIn][ColorPrimaries_BLUE][ColorPrimariesCoordinates_Y],
        false);

    Matrix4x4f XYZ_To_OutputGamut = CalculateXYZToRGBMatrix(
        ColorSpacePrimaries[gamutOut][ColorPrimaries_WHITE][ColorPrimariesCoordinates_X],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_WHITE][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_RED][ColorPrimariesCoordinates_X],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_RED][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_GREEN][ColorPrimariesCoordinates_X],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_GREEN][ColorPrimariesCoordinates_Y],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_BLUE][ColorPrimariesCoordinates_X],
        ColorSpacePrimaries[gamutOut][ColorPrimaries_BLUE][ColorPrimariesCoordinates_Y],
        false);

    Matrix4x4f intputGamut_To_OutputGamut = intputGamut_To_XYZ * XYZ_To_OutputGamut;
    *inputToOutputRecMatrix = glm::transpose(intputGamut_To_OutputGamut);
}
} // namespace CGT