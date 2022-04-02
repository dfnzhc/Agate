//
// Created by 秋鱼头 on 2022/4/2.
//

#pragma once

namespace CGT {

constexpr size_t HASH_SEED = 2166136261;

size_t Hash(const void* ptr, size_t size, size_t result = HASH_SEED);
size_t HashString(const char* str, size_t result = HASH_SEED);
size_t HashString(std::string_view str, size_t result = HASH_SEED);
size_t HashInt(int type, size_t result = HASH_SEED);
size_t HashFloat(float type, size_t result = HASH_SEED);
size_t HashPtr(const void* type, size_t result = HASH_SEED);

} // namespace CGT

 