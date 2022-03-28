//
// Created by 秋鱼头 on 2022/3/27.
//

#pragma once
#include "CGT/common.h"

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
Vector3f toVector3f(std::string_view str);

/// Tokenize a string into a list by splitting at 'delim'
std::vector<std::string> Tokenize(std::string_view str, std::string_view delim = ", ", bool includeEmpty = false);

/// Convert a time value in milliseconds into a human-readable string
std::string timeString(double time, bool precise = false);

/// Convert a memory amount in bytes into a human-readable string
std::string memString(size_t size, bool precise = false);

/// 替换字符串中的子串
std::string ReplaceAll(std::string_view str, std::string_view oldSubStr, std::string_view newSubStr);

// From https://stackoverflow.com/a/64152990/1182653
// Delete a list of items from std::vector with indices in 'selection'
template<class T, class Index = int>
inline void EraseSelected(std::vector<T>& v, const std::vector<Index>& selection)
// e.g., eraseSelected({1, 2, 3, 4, 5}, {1, 3})  ->   {1, 3, 5}
//                         ^     ^    2 and 4 get deleted
{
    // cut off the elements moved to the end of the vector by std::stable_partition
    v.resize(std::distance(
        v.begin(),
        // the stable_partition moves any element whose index is in 'selection' to the end
        std::stable_partition(v.begin(), v.end(),
                              [&selection, &v](const T& item)
                              {
                                  return !std::binary_search(
                                      selection.begin(), selection.end(),
                                      /* std::distance(std::find(v.begin(), v.end(), item), v.begin()) - if you don't like the pointer arithmetic below */
                                      static_cast<Index>(static_cast<const T*>(&item) - &v[0]));
                              })));
}

inline bool CheckFileExist(const std::string& fileName)
{
    //return std::filesystem::exists(fileName);

    // fast check file if exists
    // https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exists-using-standard-c-c11-14-17-c
    struct stat buffer;
    return (stat(fileName.c_str(), &buffer) == 0);
}
} // namespace CGT

 