//
// Created by 秋鱼头 on 2022/3/28.
//

#include "CGT/util/uTransform.h"

namespace CGT {

Matrix3x3d CoordinateSystem(const Vector3d& n)
{
    Vector3d z = Vector3d(n.x, n.y, n.z);
    Vector3d h = z;
    if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
        h.x = 1.0;
    else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
        h.y = 1.0;
    else
        h.z = 1.0;

    glm::normalize(z);
    Vector3d y = cross(h, z);
    glm::normalize(y);
    Vector3d x = cross(z, y);
    glm::normalize(x);

    Matrix3x3d o2w;
    o2w[0] = x;
    o2w[1] = y;
    o2w[2] = z;
    
    return o2w;
}

} // namespace CGT