//
// Created by 秋鱼头 on 2022/4/2.
//

#pragma once

#include <CGT/common.h>
namespace CGT {

enum ColorSpace
{
    ColorSpace_REC709,
    ColorSpace_P3,
    ColorSpace_REC2020,
    ColorSpace_Display
};

enum ColorPrimaries
{
    ColorPrimaries_WHITE,
    ColorPrimaries_RED,
    ColorPrimaries_GREEN,
    ColorPrimaries_BLUE
};

enum ColorPrimariesCoordinates
{
    ColorPrimariesCoordinates_X,
    ColorPrimariesCoordinates_Y
};

extern float ColorSpacePrimaries[4][4][2];

void FillDisplaySpecificPrimaries(float xw, float yw, float xr, float yr, float xg, float yg, float xb, float yb);

Matrix4x4f CalculateRGBToXYZMatrix(float xw,
                                   float yw,
                                   float xr,
                                   float yr,
                                   float xg,
                                   float yg,
                                   float xb,
                                   float yb,
                                   bool scaleLumaFlag);

Matrix4x4f CalculateXYZToRGBMatrix(float xw,
                                   float yw,
                                   float xr,
                                   float yr,
                                   float xg,
                                   float yg,
                                   float xb,
                                   float yb,
                                   bool scaleLumaFlag);

void SetupGamutMapperMatrices(ColorSpace gamutIn, ColorSpace gamutOut, Matrix4x4f* inputToOutputRecMatrix);

} // namespace CGT
