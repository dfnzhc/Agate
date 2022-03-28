//
// Created by 秋鱼头 on 2022/3/27.
//

#pragma once
#include <CGT/common.h>

namespace CGT {

/// convert radians to degrees
template<typename T>
inline T RadToDeg(T value)
{
    return value * (180 / PI);
}

/// convert degrees to radians
template<typename T>
inline T DegToRad(T value)
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

/// align val to the next multiple of alignment
template<typename T>
inline T AlignUp(T val, T alignment)
{
    return (val + alignment - (T) 1) & ~(alignment - (T) 1);
}

/// align val to the previous multiple of alignment
template<typename T>
inline T AlignDown(T val, T alignment)
{
    return val & ~(alignment - (T) 1);
}

template<typename T>
inline T DivideRoundingUp(T a, T b)
{
    return (a + b - (T) 1) / b;
}

uint32_t RoundUpPow2(uint32_t v);

} // namespace CGT

 