//
// Created by 秋鱼头 on 2022/4/2.
//

#include "CGT/util/hash.h"

namespace CGT {

size_t Hash(const void* ptr, size_t size, size_t result)
{
    for (size_t i = 0; i < size; ++i) {
        result = (result * 16777619) ^ ((char*) ptr)[i];
    }

    return result;
}

size_t HashString(const char* str, size_t result)
{
    return Hash(str, strlen(str), result);
}

size_t HashString(std::string_view str, size_t result)
{
    return HashString(str.data(), result);
}

size_t HashInt(int type, size_t result) { return Hash(&type, sizeof(int), result); }
size_t HashFloat(float type, size_t result) { return Hash(&type, sizeof(float), result); }
size_t HashPtr(const void* type, size_t result) { return Hash(&type, sizeof(void*), result); }

} // namespace CGT