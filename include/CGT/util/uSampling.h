//
// Created by 秋鱼头 on 2022/3/28.
//

#pragma once

#include <CGT/common.h>

namespace CGT {

    /// Dummy warping function: takes uniformly distributed points in a square and just returns them
    static Point2f SquareToUniformSquare(const Point2f& sample);

    /// Probability density of \ref SquareToUniformSquare()
    static float SquareToUniformSquarePdf(const Point2f& p);

    /// Sample a 2D tent distribution
    static Point2f SquareToTent(const Point2f& sample);

    /// Probability density of \ref SquareToTent()
    static float SquareToTentPdf(const Point2f& p);

    /// Uniformly sample a vector on a 2D disk with radius 1, centered around the origin
    static Point2f SquareToUniformDisk(const Point2f& sample);

    /// Probability density of \ref SquareToUniformDisk()
    static float SquareToUniformDiskPdf(const Point2f& p);

    /// Uniformly sample a vector on a 2D disk with radius 1, centered around the origin
    static Point2f SquareToUniformConcentric(const Point2f& sample);

    /// Probability density of \ref SquareToUniformDisk()
    static float SquareToUniformConcentricPdf(const Point2f& p);

    /// Uniformly sample a vector on the unit sphere with respect to solid angles
    static Vector3f SquareToUniformSphere(const Point2f& sample);

    /// Probability density of \ref SquareToUniformSphere()
    static float SquareToUniformSpherePdf(const Vector3f& v);

    /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to solid angles
    static Vector3f SquareToUniformHemisphere(const Point2f& sample);

    /// Probability density of \ref SquareToUniformHemisphere()
    static float SquareToUniformHemispherePdf(const Vector3f& v);

    /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to projected solid angles
    static Vector3f SquareToCosineHemisphere(const Point2f& sample);

    /// Probability density of \ref SquareToCosineHemisphere()
    static float SquareToCosineHemispherePdf(const Vector3f& v);

    /// Warp a uniformly distributed square sample to a Beckmann distribution * cosine for the given 'alpha' parameter
    static Vector3f SquareToBeckmann(const Point2f& sample, float alpha);

    /// Probability density of \ref SquareToBeckmann()
    static float SquareToBeckmannPdf(const Vector3f& m, float alpha);

    static Vector3f SquareToCone(const Point2f& sample, float cosThetaMax);

    static float SquareToConePdf(const Vector3f& m, float cosThetaMax);

    static Point2f SquareToTriangle(const Point2f& sample);

    static Vector3f SquareToPhongLobe(const Point2f& sample, float exponent);

    static float SquareToPhongLobePdf(const Vector3f& m, float exponent);

} // namespace CGT
 