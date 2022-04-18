//
// Created by 秋鱼头 on 2022/4/18.
//

#pragma once

#include "Agate/Shader/vec_math.h"
#include "Agate/Shader/Matrix.h"

#ifndef __CUDACC__
#  include <assert.h>
#  define AGATE_AABB_ASSERT assert
#else
#  define AGATE_AABB_ASSERT(x)
#endif

namespace Agate {

class AABB
{
public:

    /** Construct an invalid box */
    AGATE_CPUGPU AABB();

    /** Construct from min and max vectors */
    AGATE_CPUGPU AABB(const float3& min, const float3& max);

    /** Construct from three points (e.g. triangle) */
    AGATE_CPUGPU AABB(const float3& v0, const float3& v1, const float3& v2);

    /** Exact equality */
    AGATE_CPUGPU bool operator==(const AABB& other) const;

    /** Array access */
    AGATE_CPUGPU float3& operator[](int i);

    /** Const array access */
    AGATE_CPUGPU const float3& operator[](int i) const;

    /** Direct access */
    AGATE_CPUGPU float* data();

    /** Set using two vectors */
    AGATE_CPUGPU void set(const float3& min, const float3& max);

    /** Set using three points (e.g. triangle) */
    AGATE_CPUGPU void set(const float3& v0, const float3& v1, const float3& v2);

    /** Invalidate the box */
    AGATE_CPUGPU void invalidate();

    /** Check if the box is valid */
    AGATE_CPUGPU bool valid() const;

    /** Check if the point is in the box */
    AGATE_CPUGPU bool contains(const float3& p) const;

    /** Check if the box is fully contained in the box */
    AGATE_CPUGPU bool contains(const AABB& bb) const;

    /** Extend the box to include the given point */
    AGATE_CPUGPU void include(const float3& p);

    /** Extend the box to include the given box */
    AGATE_CPUGPU void include(const AABB& other);

    /** Extend the box to include the given box */
    AGATE_CPUGPU void include(const float3& min, const float3& max);

    /** Compute the box center */
    AGATE_CPUGPU float3 center() const;

    /** Compute the box center in the given dimension */
    AGATE_CPUGPU float center(int dim) const;

    /** Compute the box extent */
    AGATE_CPUGPU float3 extent() const;

    /** Compute the box extent in the given dimension */
    AGATE_CPUGPU float extent(int dim) const;

    /** Compute the volume of the box */
    AGATE_CPUGPU float volume() const;

    /** Compute the surface area of the box */
    AGATE_CPUGPU float area() const;

    /** Compute half the surface area of the box */
    AGATE_CPUGPU float halfArea() const;

    /** Get the index of the longest axis */
    AGATE_CPUGPU int longestAxis() const;

    /** Get the extent of the longest axis */
    AGATE_CPUGPU float maxExtent() const;

    /** Check for intersection with another box */
    AGATE_CPUGPU bool intersects(const AABB& other) const;

    /** Make the current box be the intersection between this one and another one */
    AGATE_CPUGPU void intersection(const AABB& other);

    /** Enlarge the box by moving both min and max by 'amount' */
    AGATE_CPUGPU void enlarge(float amount);

    AGATE_CPUGPU void transform(const Matrix3x4& m);
    AGATE_CPUGPU void transform(const Matrix4x4& m);

    /** Check if the box is flat in at least one dimension  */
    AGATE_CPUGPU bool isFlat() const;

    /** Compute the minimum Euclidean distance from a point on the
     surface of this Aabb to the point of interest */
    AGATE_CPUGPU float distance(const float3& x) const;

    /** Compute the minimum squared Euclidean distance from a point on the
     surface of this Aabb to the point of interest */
    AGATE_CPUGPU float distance2(const float3& x) const;

    /** Compute the minimum Euclidean distance from a point on the surface
      of this Aabb to the point of interest.
      If the point of interest lies inside this Aabb, the result is negative  */
    AGATE_CPUGPU float signedDistance(const float3& x) const;

    /** Min bound */
    float3 m_min;
    /** Max bound */
    float3 m_max;
};

AGATE_INLINE AGATE_CPUGPU
AABB::AABB()
{
    invalidate();
}

AGATE_INLINE AGATE_CPUGPU
AABB::AABB(const float3& min, const float3& max)
{
    set(min, max);
}

AGATE_INLINE AGATE_CPUGPU
AABB::AABB(const float3& v0, const float3& v1, const float3& v2)
{
    set(v0, v1, v2);
}

AGATE_INLINE AGATE_CPUGPU
bool AABB::operator==(const AABB& other) const
{
    return m_min.x == other.m_min.x &&
        m_min.y == other.m_min.y &&
        m_min.z == other.m_min.z &&
        m_max.x == other.m_max.x &&
        m_max.y == other.m_max.y &&
        m_max.z == other.m_max.z;
}

AGATE_INLINE AGATE_CPUGPU
float3& AABB::operator[](int i)
{
    AGATE_AABB_ASSERT(i >= 0 && i <= 1);
    return (&m_min)[i];
}

AGATE_INLINE AGATE_CPUGPU
const float3& AABB::operator[](int i) const
{
    AGATE_AABB_ASSERT(i >= 0 && i <= 1);
    return (&m_min)[i];
}

AGATE_INLINE AGATE_CPUGPU
float* AABB::data()
{
    return &m_min.x;
}

AGATE_INLINE AGATE_CPUGPU
void AABB::set(const float3& min, const float3& max)
{
    m_min = min;
    m_max = max;
}

AGATE_INLINE AGATE_CPUGPU
void AABB::set(const float3& v0, const float3& v1, const float3& v2)
{
    m_min = fminf(v0, fminf(v1, v2));
    m_max = fmaxf(v0, fmaxf(v1, v2));
}

AGATE_INLINE AGATE_CPUGPU
void AABB::invalidate()
{
    m_min = make_float3(1e37f);
    m_max = make_float3(-1e37f);
}

AGATE_INLINE AGATE_CPUGPU
bool AABB::valid() const
{
    return m_min.x <= m_max.x &&
        m_min.y <= m_max.y &&
        m_min.z <= m_max.z;
}

AGATE_INLINE AGATE_CPUGPU
bool AABB::contains(const float3& p) const
{
    return p.x >= m_min.x && p.x <= m_max.x &&
        p.y >= m_min.y && p.y <= m_max.y &&
        p.z >= m_min.z && p.z <= m_max.z;
}

AGATE_INLINE AGATE_CPUGPU
bool AABB::contains(const AABB& bb) const
{
    return contains(bb.m_min) && contains(bb.m_max);
}

AGATE_INLINE AGATE_CPUGPU
void AABB::include(const float3& p)
{
    m_min = fminf(m_min, p);
    m_max = fmaxf(m_max, p);
}

AGATE_INLINE AGATE_CPUGPU
void AABB::include(const AABB& other)
{
    m_min = fminf(m_min, other.m_min);
    m_max = fmaxf(m_max, other.m_max);
}

AGATE_INLINE AGATE_CPUGPU
void AABB::include(const float3& min, const float3& max)
{
    m_min = fminf(m_min, min);
    m_max = fmaxf(m_max, max);
}

AGATE_INLINE AGATE_CPUGPU
float3 AABB::center() const
{
    AGATE_AABB_ASSERT(valid());
    return (m_min + m_max) * 0.5f;
}

AGATE_INLINE AGATE_CPUGPU
float AABB::center(int dim) const
{
    AGATE_AABB_ASSERT(valid());
    AGATE_AABB_ASSERT(dim >= 0 && dim <= 2);
    return (((const float*) (&m_min))[dim] + ((const float*) (&m_max))[dim]) * 0.5f;
}

AGATE_INLINE AGATE_CPUGPU
float3 AABB::extent() const
{
    AGATE_AABB_ASSERT(valid());
    return m_max - m_min;
}

AGATE_INLINE AGATE_CPUGPU
float AABB::extent(int dim) const
{
    AGATE_AABB_ASSERT(valid());
    return ((const float*) (&m_max))[dim] - ((const float*) (&m_min))[dim];
}

AGATE_INLINE AGATE_CPUGPU
float AABB::volume() const
{
    AGATE_AABB_ASSERT(valid());
    const float3 d = extent();
    return d.x * d.y * d.z;
}

AGATE_INLINE AGATE_CPUGPU
float AABB::area() const
{
    return 2.0f * halfArea();
}

AGATE_INLINE AGATE_CPUGPU
float AABB::halfArea() const
{
    AGATE_AABB_ASSERT(valid());
    const float3 d = extent();
    return d.x * d.y + d.y * d.z + d.z * d.x;
}

AGATE_INLINE AGATE_CPUGPU
int AABB::longestAxis() const
{
    AGATE_AABB_ASSERT(valid());
    const float3 d = extent();

    if (d.x > d.y)
        return d.x > d.z ? 0 : 2;
    return d.y > d.z ? 1 : 2;
}

AGATE_INLINE AGATE_CPUGPU
float AABB::maxExtent() const
{
    return extent(longestAxis());
}

AGATE_INLINE AGATE_CPUGPU
bool AABB::intersects(const AABB& other) const
{
    if (other.m_min.x > m_max.x || other.m_max.x < m_min.x) return false;
    if (other.m_min.y > m_max.y || other.m_max.y < m_min.y) return false;
    if (other.m_min.z > m_max.z || other.m_max.z < m_min.z) return false;
    return true;
}

AGATE_INLINE AGATE_CPUGPU
void AABB::intersection(const AABB& other)
{
    m_min.x = fmaxf(m_min.x, other.m_min.x);
    m_min.y = fmaxf(m_min.y, other.m_min.y);
    m_min.z = fmaxf(m_min.z, other.m_min.z);
    m_max.x = fminf(m_max.x, other.m_max.x);
    m_max.y = fminf(m_max.y, other.m_max.y);
    m_max.z = fminf(m_max.z, other.m_max.z);
}

AGATE_INLINE AGATE_CPUGPU
void AABB::enlarge(float amount)
{
    AGATE_AABB_ASSERT(valid());
    m_min -= make_float3(amount);
    m_max += make_float3(amount);
}

AGATE_INLINE AGATE_CPUGPU
void AABB::transform(const Matrix3x4& m)
{
    // row-major matrix -> column vectors:
    // x ={ m[0], m[4], m[8] }
    // y ={ m[1], m[5], m[9] }
    // z ={ m[2], m[6], m[10] }
    // 3,7,11 translation

    // no need to initialize, will be overwritten completely
    AABB result;
    const float loxx = m[0] * m_min.x;
    const float hixx = m[0] * m_max.x;
    const float loyx = m[1] * m_min.y;
    const float hiyx = m[1] * m_max.y;
    const float lozx = m[2] * m_min.z;
    const float hizx = m[2] * m_max.z;
    result.m_min.x = fminf(loxx, hixx) + fminf(loyx, hiyx) + fminf(lozx, hizx) + m[3];
    result.m_max.x = fmaxf(loxx, hixx) + fmaxf(loyx, hiyx) + fmaxf(lozx, hizx) + m[3];
    const float loxy = m[4] * m_min.x;
    const float hixy = m[4] * m_max.x;
    const float loyy = m[5] * m_min.y;
    const float hiyy = m[5] * m_max.y;
    const float lozy = m[6] * m_min.z;
    const float hizy = m[6] * m_max.z;
    result.m_min.y = fminf(loxy, hixy) + fminf(loyy, hiyy) + fminf(lozy, hizy) + m[7];
    result.m_max.y = fmaxf(loxy, hixy) + fmaxf(loyy, hiyy) + fmaxf(lozy, hizy) + m[7];
    const float loxz = m[8] * m_min.x;
    const float hixz = m[8] * m_max.x;
    const float loyz = m[9] * m_min.y;
    const float hiyz = m[9] * m_max.y;
    const float lozz = m[10] * m_min.z;
    const float hizz = m[10] * m_max.z;
    result.m_min.z = fminf(loxz, hixz) + fminf(loyz, hiyz) + fminf(lozz, hizz) + m[11];
    result.m_max.z = fmaxf(loxz, hixz) + fmaxf(loyz, hiyz) + fmaxf(lozz, hizz) + m[11];
    *this = result;
}

AGATE_INLINE AGATE_CPUGPU
void AABB::transform(const Matrix4x4& m)
{
    const float3 b000 = m_min;
    const float3 b001 = make_float3(m_min.x, m_min.y, m_max.z);
    const float3 b010 = make_float3(m_min.x, m_max.y, m_min.z);
    const float3 b011 = make_float3(m_min.x, m_max.y, m_max.z);
    const float3 b100 = make_float3(m_max.x, m_min.y, m_min.z);
    const float3 b101 = make_float3(m_max.x, m_min.y, m_max.z);
    const float3 b110 = make_float3(m_max.x, m_max.y, m_min.z);
    const float3 b111 = m_max;

    invalidate();
    include(make_float3(m * make_float4(b000, 1.0f)));
    include(make_float3(m * make_float4(b001, 1.0f)));
    include(make_float3(m * make_float4(b010, 1.0f)));
    include(make_float3(m * make_float4(b011, 1.0f)));
    include(make_float3(m * make_float4(b100, 1.0f)));
    include(make_float3(m * make_float4(b101, 1.0f)));
    include(make_float3(m * make_float4(b110, 1.0f)));
    include(make_float3(m * make_float4(b111, 1.0f)));
}

AGATE_INLINE AGATE_CPUGPU
bool AABB::isFlat() const
{
    return m_min.x == m_max.x ||
        m_min.y == m_max.y ||
        m_min.z == m_max.z;
}

AGATE_INLINE AGATE_CPUGPU
float AABB::distance(const float3& x) const
{
    return sqrtf(distance2(x));
}

AGATE_INLINE AGATE_CPUGPU
float AABB::signedDistance(const float3& x) const
{
    if (m_min.x <= x.x && x.x <= m_max.x &&
        m_min.y <= x.y && x.y <= m_max.y &&
        m_min.z <= x.z && x.z <= m_max.z) {
        float distance_x = fminf(x.x - m_min.x, m_max.x - x.x);
        float distance_y = fminf(x.y - m_min.y, m_max.y - x.y);
        float distance_z = fminf(x.z - m_min.z, m_max.z - x.z);

        float min_distance = fminf(distance_x, fminf(distance_y, distance_z));
        return -min_distance;
    }

    return distance(x);
}

AGATE_INLINE AGATE_CPUGPU
float AABB::distance2(const float3& x) const
{
    float3 box_dims = m_max - m_min;

    // compute vector from min corner of box
    float3 v = x - m_min;

    float dist2 = 0;
    float excess;

    // project vector from box min to x on each axis,
    // yielding distance to x along that axis, and count
    // any excess distance outside box extents

    excess = 0;
    if (v.x < 0)
        excess = v.x;
    else if (v.x > box_dims.x)
        excess = v.x - box_dims.x;
    dist2 += excess * excess;

    excess = 0;
    if (v.y < 0)
        excess = v.y;
    else if (v.y > box_dims.y)
        excess = v.y - box_dims.y;
    dist2 += excess * excess;

    excess = 0;
    if (v.z < 0)
        excess = v.z;
    else if (v.z > box_dims.z)
        excess = v.z - box_dims.z;
    dist2 += excess * excess;

    return dist2;
}

} // namespace Agate