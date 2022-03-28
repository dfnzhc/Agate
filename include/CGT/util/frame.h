//
// Created by 秋鱼头 on 2022/3/28.
//

#pragma once

#include <CGT/common.h>
#include "uTransform.h"
#include "uMath.h"
namespace CGT {

/**
 * \brief Stores a three-dimensional orthonormal coordinate frame
 *
 * This class is mostly used to quickly convert between different
 * cartesian coordinate systems and to efficiently compute certain
 * quantities (e.g. \ref cosTheta(), \ref tanTheta, ..).
 */
struct Frame
{
    Vector3f s, t;
    Normal3f n;

    /// Default constructor -- performs no initialization!
    Frame() = default;

    /// Construct a new coordinate frame from a single vector
    explicit Frame(const Vector3f& n) : n(n)
    {
        CoordinateSystem(n, s, t);
    }

    /// Convert from world coordinates to local coordinates
    Vector3f toLocal(const Vector3f& v) const
    {
        return {
            glm::dot(v, s), glm::dot(v, t), glm::dot(v, n)
        };
    }

    /// Convert from local coordinates to world coordinates
    Vector3f toWorld(const Vector3f& v) const
    {
        return s * v.x + t * v.y + n * v.z;
    }

    /** \brief Assuming that the given direction is in the local coordinate 
     * system, return the cosine of the angle between the normal and v */
    static float cosTheta(const Vector3f& v)
    {
        return v.z;
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the sine of the angle between the normal and v */
    static float sinTheta(const Vector3f& v)
    {
        float temp = sinTheta2(v);
        if (temp <= 0.0f)
            return 0.0f;
        return std::sqrt(temp);
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the tangent of the angle between the normal and v */
    static float tanTheta(const Vector3f& v)
    {
        float temp = 1 - v.z * v.z;
        if (temp <= 0.0f)
            return 0.0f;
        return std::sqrt(temp) / v.z;
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared sine of the angle between the normal and v */
    static float sinTheta2(const Vector3f& v)
    {
        return 1.0f - v.z * v.z;
    }

    /** \brief Assuming that the given direction is in the local coordinate 
     * system, return the sine of the phi parameter in spherical coordinates */
    static float sinPhi(const Vector3f& v)
    {
        float sinTheta = Frame::sinTheta(v);
        if (sinTheta == 0.0f)
            return 1.0f;
        return Clamp(v.y / sinTheta, -1.0f, 1.0f);
    }

    /** \brief Assuming that the given direction is in the local coordinate 
     * system, return the cosine of the phi parameter in spherical coordinates */
    static float cosPhi(const Vector3f& v)
    {
        float sinTheta = Frame::sinTheta(v);
        if (sinTheta == 0.0f)
            return 1.0f;
        return Clamp(v.x / sinTheta, -1.0f, 1.0f);
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared sine of the phi parameter in  spherical
     * coordinates */
    static float sinPhi2(const Vector3f& v)
    {
        return Clamp(v.y * v.y / sinTheta2(v), 0.0f, 1.0f);
    }

    /** \brief Assuming that the given direction is in the local coordinate
     * system, return the squared cosine of the phi parameter in  spherical
     * coordinates */
    static float cosPhi2(const Vector3f& v)
    {
        return Clamp(v.x * v.x / sinTheta2(v), 0.0f, 1.0f);
    }

    /// Equality test
    bool operator==(const Frame& frame) const
    {
        return frame.s == s && frame.t == t && frame.n == n;
    }

    /// Inequality test
    bool operator!=(const Frame& frame) const
    {
        return !operator==(frame);
    }

};

} // namespace CGT


 