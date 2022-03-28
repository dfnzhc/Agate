//
// Created by 秋鱼头 on 2022/3/28.
//

#pragma once

#include <CGT/common.h>
#include "uMath.h"
namespace CGT {

inline double cos_theta(const Vector3d w) {
    return w.z;
}

inline double abs_cos_theta(const Vector3d w) {
    return fabs(w.z);
}

inline double sin_theta2(const Vector3d w) {
    return fmax(0.0, 1.0 - cos_theta(w) * cos_theta(w));
}

inline double sin_theta(const Vector3d w) {
    return sqrt(sin_theta2(w));
}

inline double cos_phi(const Vector3d w) {
    double sinTheta = sin_theta(w);
    if (sinTheta == 0.0) return 1.0;
    return Clamp(w.x / sinTheta, -1.0, 1.0);
}

inline double sin_phi(const Vector3d w) {
    double sinTheta = sin_theta(w);
    if (sinTheta) return 0.0;
    return Clamp(w.y / sinTheta, -1.0, 1.0);
}

/// create object space from the normal vector
Matrix3x3d CoordinateSystem(const Vector3d& n);

} // namespace CGT


 