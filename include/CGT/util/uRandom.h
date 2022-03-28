//
// Created by 秋鱼头 on 2022/3/28.
//

#pragma once

#include <random>
#include "uMath.h"

namespace CGT {

static std::mersenne_twister_engine<std::uint_fast32_t, 32, 624, 397, 31, 0x9908b0df,
                                    11, 0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18,
                                    1812433253> minstd_engine;

static double rmax = 1.0 / (minstd_engine.max() - minstd_engine.min());

/**
 * Returns a number distributed uniformly over [0, 1].
 */
inline double Random_uniform()
{
    return Clamp(double(minstd_engine() - minstd_engine.min()) * rmax, 0.0000001, 0.99999999);
}

/**
 * Returns true with probability p and false with probability 1 - p.
 */
inline bool Coin_Flip(double p)
{
    return Random_uniform() < p;
}
} // namespace CGT


 