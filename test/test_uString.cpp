//
// Created by 秋鱼头 on 2022/3/27.
//

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <CGT/util/uString.h>

TEST_CASE("Test endWith", "[endsWith]")
{
    REQUIRE(CGT::endsWith("123321.abc", "abc") == true);
    REQUIRE(CGT::endsWith("123321.xyz", "xyz") == true);
    REQUIRE(CGT::endsWith("123321.xz", "xyz") == false);
}

TEST_CASE("Test toBool", "[toBool]")
{
    REQUIRE(CGT::toBool("true") == true);
    REQUIRE(CGT::toBool("false") == false);
}

TEST_CASE("Test toInt", "[toInt]")
{
    REQUIRE(CGT::toInt("123321") == 123321);
    REQUIRE(CGT::toInt("992") == 992);
}

TEST_CASE("Test toFloat", "[toFloat]")
{
    REQUIRE(CGT::toFloat("123.521") == 123.521f);
}

TEST_CASE("Test toVector3f", "[toVector3f]")
{
    auto v = CGT::toVector3f("1, 2, 3");
    REQUIRE(v == glm::vec3{1, 2, 3});
}
