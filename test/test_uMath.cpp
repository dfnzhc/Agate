//
// Created by 秋鱼头 on 2022/3/27.
//

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include "CGT/util/uMath.h"

TEST_CASE("Degree and radians", "[RadToDeg & DegToRad]")
{
    REQUIRE(CGT::RadToDeg(CGT::PI) == 180.0);
    REQUIRE(CGT::DegToRad(45.0) == CGT::PI_OVER4);
}

TEST_CASE("Cosin Cosin", "[getCosSin]")
{
    auto [cos, sin] = CGT::getCosSin(CGT::PI_OVER4);
    REQUIRE(CGT::FLT_EQUAL(cos, CGT::INV_SQRT2));
    REQUIRE(CGT::FLT_EQUAL(sin, CGT::INV_SQRT2));
}

TEST_CASE("Clamp", "[Clamp]")
{
    REQUIRE(CGT::Clamp(256, 0, 255) == 255);
    REQUIRE(CGT::Clamp(-1, 0, 255) == 0);
}

TEST_CASE("Roundup", "[RoundUpPow2]")
{
    REQUIRE(CGT::RoundUpPow2(31) == 32);
    REQUIRE(CGT::RoundUpPow2(45) == 64);
}
