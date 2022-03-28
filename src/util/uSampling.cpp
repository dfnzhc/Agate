//
// Created by 秋鱼头 on 2022/3/28.
//
#include "CGT/util/uSampling.h"

namespace CGT {
Point2f SquareToUniformSquare(const Point2f& sample)
{
    return sample;
}

float SquareToUniformSquarePdf(const Point2f& sample)
{
    return (sample.length() >= 0.0 && sample.length() <= 1.0) ? 1.0f : 0.0f;
}

Point2f SquareToTent(const Point2f& sample)
{
    Point2f res{0};

    for (int i = 0; i < 2; i++) {
        float s = sample[i];
        if (s < 0.5f) res[i] = sqrtf(2.0f * s) - 1.0f;
        else res[i] = 1 - sqrtf(2.0f - 2.0f * s);
    }

    return res;
}

float SquareToTentPdf(const Point2f& p)
{
    float a = abs(p[0]) <= 1 ? 1.0f - abs(p[0]) : 0;
    float b = abs(p[1]) <= 1 ? 1.0f - abs(p[1]) : 0;

    return a * b;
}

Point2f SquareToUniformDisk(const Point2f& sample)
{
    float r = sqrtf(sample[0]);
    float theta = sample[1] * PI * 2.0f;

    return Point2f{r * cosf(theta), r * sinf(theta)};
}

float SquareToUniformDiskPdf(const Point2f& p)
{
    if (p.length() <= 1.0f + EPS_F) return INV_PI;

    return 0.0f;
}

Point2f SquareToUniformConcentric(const Point2f& sample)
{
    Point2f uOffset = 2.f * sample - Vector2f{1.f};

    if (uOffset.x == 0 && uOffset.y == 0)
        return Point2f{0.0f};

    float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PI_OVER4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PI_OVER2 - PI_OVER4 * (uOffset.x / uOffset.y);
    }

    return r * Point2f{std::cos(theta), std::sin(theta)};
}

float SquareToUniformConcentricPdf(const Point2f& p)
{
    if (p.length() <= 1.0f + EPS_F) return INV_PI;

    return 0.0f;
}

Vector3f SquareToUniformSphere(const Point2f& sample)
{
    float theta = acosf(sample[0] * 2.0f - 1.0f);
    float phi = sample[1] * PI * 2.0f;

    float cosTheta = cosf(theta);
    float sinTheta = sinf(theta);

    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    return Vector3f{sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}

float SquareToUniformSpherePdf(const Vector3f& v)
{
    if (v.length() <= 1.0f + EPS_F) return INV_PI * 0.25f;

    return 0.0f;
}

Vector3f SquareToUniformHemisphere(const Point2f& sample)
{
    float theta = acosf(sample[0]);
    float phi = sample[1] * PI * 2.0f;

    float cosTheta = cosf(theta);
    float sinTheta = sinf(theta);

    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    return Vector3f{sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}

float SquareToUniformHemispherePdf(const Vector3f& v)
{
    if (v.z >= EPS_F && v.length() <= 1.0f + EPS_F) return INV_PI * 0.5f;

    return 0.0f;
}

// 《Advanced global illumination》 p66
Vector3f SquareToCosineHemisphere(const Point2f& sample)
{
    // Vector2f unifDisk = SquareToUniformDisk(sample);
    //
    // return Vector3f{unifDisk.x(), unifDisk.y(), sqrtf(1.0f - unifDisk.squaredNorm())};
    float theta = acosf(sqrtf(sample[0]));
    float phi = sample[1] * PI * 2.0f;

    float cosTheta = cosf(theta);
    float sinTheta = sinf(theta);

    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    return Vector3f{sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}

float SquareToCosineHemispherePdf(const Vector3f& v)
{
    if (v.z >= EPS_F && v.length() <= 1.0f + EPS_F) return INV_PI * v.z;

    return 0.0f;
}

Vector3f SquareToBeckmann(const Point2f& sample, float alpha)
{
    float phi = sample[0] * 2.0f * PI;
    float theta = acosf(1.0f / sqrtf(1.0f - alpha * alpha * log(sample[1])));

    float cosTheta = cosf(theta);
    float sinTheta = sinf(theta);

    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    return Vector3f{sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}

float SquareToBeckmannPdf(const Vector3f& m, float alpha)
{
    if (m.z <= 0.0f)
        return 0.0f;

    float cosTheta = m.z;
    float cos2Theta = cosTheta * cosTheta;
    float tan2Theta = (1.0f - cos2Theta) / cos2Theta;

    float alpha2 = alpha * alpha;

    float azimuthal = 0.5f * INV_PI;
    float longitudinal = 2.0f * exp(-tan2Theta / alpha2) / (alpha2 * cosTheta * cos2Theta);

    return azimuthal * longitudinal;
}

Vector3f SquareToCone(const Point2f& sample, float cosThetaMax)
{
    float cosTheta = (1.0 - sample[0]) + sample[0] * cosThetaMax;
    float sinTheta = std::sqrtf(1.0f - cosTheta * cosTheta);

    float phi = sample[1] * 2.0f * PI;

    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    return Vector3f{sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}

float SquareToConePdf(const Vector3f& m, float cosThetaMax)
{
    if (cosThetaMax == 1) return 0;
   
    return 1.0f / (2.0f * PI * (1.0f - cosThetaMax));
}

Point2f SquareToTriangle(const Point2f& sample)
{
    float su0 = std::sqrtf(sample[0]);

    return Point2f{1.0f - su0, sample[1] * su0};
}

Vector3f SquareToPhongLobe(const Point2f& sample, float exponent)
{
    float theta = acos(pow(1 - sample[0], 1 / (exponent + 1)));
    float phi = 2 * PI * sample[1];

    float cosTheta = cosf(theta);
    float sinTheta = sinf(theta);

    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    return Vector3f{sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}

float SquareToPhongLobePdf(const Vector3f& m, float exponent)
{
    return (exponent + 1) * pow(m.z, exponent) / (2 * PI);
}
} // namespace CGT