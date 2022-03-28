//
// Created by 秋鱼头 on 2022/3/27.
//
#include "CGT/util/uMath.h"

namespace CGT {

uint32_t RoundUpPow2(uint32_t v)
{
    v--;
    v |= v >> 1; v |= v >> 2;
    v |= v >> 4; v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

} // namespace CGT