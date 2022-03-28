//
// Created by 秋鱼头 on 2022/3/27.
//

#include "pch.h"
#include "CGT/common.h"

namespace CGT {

bool ReadFile(std::string_view name, std::vector<char>& data, bool isbinary)
{
    std::ifstream file{name.data(), std::ios::in | (isbinary ? std::ios::binary : 0)};
    if (!file)
        return false;
    
    file.seekg(0, std::ios_base::end);
    auto fileLength = file.tellg();
    file.seekg(0, std::ios_base::beg);
    
    data.resize(fileLength);
    file.read(data.data(), fileLength * sizeof (char));
    
    return true;
}
bool SaveFile(std::string_view name, const void* data, bool isbinary)
{
    return false;
}

} // namespace CGT

