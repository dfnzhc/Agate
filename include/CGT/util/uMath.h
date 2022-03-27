//
// Created by 秋鱼头 on 2022/3/27.
//

#pragma once
#include <CGT/common.h>

namespace CGT {

/// convert radians to degrees
template<typename T>
inline T radToDeg(T value)
{
    return value * (180 / PI);
}

/// convert degrees to radians
template<typename T>
inline T degToRad(T value)
{
    return value * (PI / 180);
}

/// get sin and cos value with given value
template<typename T>
inline auto getCosSin(T theta)
{
    return std::make_pair(std::cos(theta), std::sin(theta));
}

/// clamp value
template<typename T, typename U, typename V>
inline T Clamp(T value, U low, V high)
{
    if (value < low)
        return low;
    else if (value > high)
        return high;
    else
        return value;
}

/// float equation
template<typename T>
inline bool FLT_EQUAL(T a, T b)
{
    return std::abs(a - b) < EPS_F;
//    if constexpr(std::is_same_v<float, T>) {
//        return std::abs(a - b) < EPS_F;
//    } else if constexpr(std::is_same_v<double, T>) {
//        return std::abs(a - b) < EPS_D;
//    }
}

uint32_t RoundUpPow2(uint32_t v);

} // namespace CGT

 