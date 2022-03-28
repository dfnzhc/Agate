//
// Created by 秋鱼头 on 2022/3/28.
//

#include "CGT/util/uTransform.h"

namespace CGT {

void CoordinateSystem(const Vector3f &a, Vector3f &b, Vector3f &c) {
    if (std::abs(a.x) > std::abs(a.y)) {
        float invLen = 1.0f / std::sqrt(a.x * a.x + a.z * a.z);
        c = Vector3f{ a.z * invLen, 0.0f, -a.x * invLen };
    } else {
        float invLen = 1.0f / std::sqrt(a.y * a.y + a.z * a.z);
        c = Vector3f{ 0.0f, a.z * invLen, -a.y * invLen };
    }
    b = glm::cross(c, a);
}
} // namespace CGT