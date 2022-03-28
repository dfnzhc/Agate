//
// Created by 秋鱼头 on 2022/3/27.
//

#pragma once

#include <numbers>
#include <glm/glm.hpp>
namespace filesystem {
class path;
class resolver;
}

namespace CGT {

constexpr float EPS_F = 1e-5;
constexpr float INF_F = std::numeric_limits<float>::infinity();
constexpr double EPS_D = 1e-16;
constexpr double INF_D = std::numeric_limits<double>::infinity();

constexpr double PI = std::numbers::pi;
constexpr double INV_PI = 1.0 / PI;
constexpr double INV_2PI = 0.5 / PI;
constexpr double INV_4PI = 0.25 / PI;
constexpr double PI_OVER2 = 0.5 * PI;
constexpr double PI_OVER4 = 0.25 * PI;
constexpr double SQRT2 = std::numbers::sqrt2;
constexpr double INV_SQRT2 = 1.0 / SQRT2;
constexpr double SQRT3 = std::numbers::sqrt3;
constexpr double INV_SQRT3 = 1.0 / SQRT3;

/// define some data structure
using Vector1f = glm::vec1;
using Vector2f = glm::vec2;
using Vector3f = glm::vec3;
using Vector4f = glm::vec4;
using Vector1d = glm::highp_vec1;
using Vector2d = glm::highp_vec2;
using Vector3d = glm::highp_vec3;
using Vector4d = glm::highp_vec4;
using Point1f = glm::vec1;
using Point2f = glm::vec2;
using Point3f = glm::vec3;
using Point4f = glm::vec4;
using Point1d = glm::highp_vec1;
using Point2d = glm::highp_vec2;
using Point3d = glm::highp_vec3;
using Point4d = glm::highp_vec4;


/// for debugging purposes
using std::cout;
using std::cerr;
using std::endl;

/// Simple exception class
class CGTException : public std::runtime_error
{
public:
    /// Variadic template constructor to support printf-style arguments
    template<typename... Args>
    explicit CGTException(std::string_view fmt_str, const Args& ... args)
        : std::runtime_error(std::format(fmt_str, args...)) {}
};

bool ReadFile(std::string_view name, std::vector<char>& data, bool isbinary = true);
bool SaveFile(std::string_view name, void const* data, bool isbinary = true);

}