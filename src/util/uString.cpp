//
// Created by 秋鱼头 on 2022/3/27.
//
#include <CGT/common.h>
#include "pch.h"
#include "CGT/util/uString.h"

namespace CGT {

// check if a string ends with given ending
bool endsWith(std::string_view str, std::string_view ending)
{
    return str.ends_with(ending);
}

// convert string to lower case
std::string toLower(std::string_view str)
{
    std::string result;
    result.resize(str.size());
    std::transform(std::execution::par, str.begin(), str.end(), result.begin(), ::tolower);

    return result;
}

bool toBool(std::string_view str)
{
    std::string value = toLower(str);
    if (value == "false")
        return false;
    else if (value == "true")
        return true;
    else
        throw CGTException("Could not parse boolean value \"%s\"", str);
}

int toInt(std::string_view str)
{
    size_t pos;
    int result = std::stoi(str.data(), &pos);
    if (pos != str.size())
        throw CGTException("Could not parse integer value \"%s\"", str);

    return result;
}

unsigned int toUInt(std::string_view str)
{
    size_t pos;
    unsigned int result = std::stoul(str.data(), &pos);
    if (pos != str.size())
        throw CGTException("Could not parse integer value \"%s\"", str);

    return result;
}

float toFloat(std::string_view str)
{
    size_t pos;
    float result = std::stof(str.data(), &pos);
    if (pos != str.size())
        throw CGTException("Could not parse float value \"%s\"", str);

    return result;
}

glm::vec3 toVector3f(std::string_view str)
{
    std::vector<std::string> tokens = tokenize(str);
    if (tokens.size() != 3)
        throw CGTException("Expected 3 values");
    glm::vec3 result;
    for (int i = 0; i < 3; ++i)
        result[i] = toFloat(tokens[i]);
    return result;
}

std::vector<std::string> tokenize(std::string_view str, std::string_view delim, bool includeEmpty)
{
    std::string::size_type lastPos = 0, pos = str.find_first_of(delim, lastPos);
    std::vector<std::string> tokens;

    while (lastPos != std::string::npos) {
        if (pos != lastPos || includeEmpty)
            tokens.emplace_back(str.substr(lastPos, pos - lastPos));
        lastPos = pos;
        if (lastPos != std::string::npos) {
            lastPos += 1;
            pos = str.find_first_of(delim, lastPos);
        }
    }

    return tokens;
}

std::string timeString(double time, bool precise)
{
    if (std::isnan(time) || std::isinf(time))
        return "inf";

    std::string suffix = "ms";
    if (time > 1000) {
        time /= 1000;suffix = "s";
        if (time > 60) {
            time /= 60;suffix = "m";
            if (time > 60) {
                time /= 60;suffix = "h";
                if (time > 12) {
                    time /= 12;suffix = "d";
                }
            }
        }
    }

    std::ostringstream os;
    os << std::setprecision(precise ? 4 : 1)
       << std::fixed << time << suffix;

    return os.str();
}

std::string memString(size_t size, bool precise)
{
    auto value = (double) size;
    const char *suffixes[] = {
        "B", "KiB", "MiB", "GiB", "TiB", "PiB"
    };
    int suffix = 0;
    while (suffix < 5 && value > 1024.0f) {
        value /= 1024.0f; ++suffix;
    }

    std::ostringstream os;
    os << std::setprecision(suffix == 0 ? 0 : (precise ? 4 : 1))
       << std::fixed << value << " " << suffixes[suffix];

    return os.str();
}

} // namespace CGT
