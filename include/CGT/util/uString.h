//
// Created by 秋鱼头 on 2022/3/27.
//

#pragma once
#include <glm/glm.hpp>

namespace CGT {

/// check if a string ends with given ending
bool endsWith(std::string_view str, std::string_view ending);

/// convert a string to lower case
std::string toLower(std::string_view str);

/// Convert a string into an boolean value
bool toBool(std::string_view str);

/// Convert a string into a signed integer value
int toInt(std::string_view str);

/// Convert a string into an unsigned integer value
unsigned int toUInt(std::string_view str);

/// Convert a string into a floating point value
float toFloat(std::string_view str);

/// Convert a string into a 3D vector
glm::vec3 toVector3f(std::string_view str);

/// Tokenize a string into a list by splitting at 'delim'
std::vector<std::string> tokenize(std::string_view str, std::string_view delim = ", ", bool includeEmpty = false);

/// Convert a time value in milliseconds into a human-readable string
std::string timeString(double time, bool precise = false);

/// Convert a memory amount in bytes into a human-readable string
std::string memString(size_t size, bool precise = false);
} // namespace CGT

 