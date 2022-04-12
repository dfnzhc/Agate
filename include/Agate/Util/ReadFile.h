//
// Created by 秋鱼头 on 2022/4/12.
//

#pragma once

#include <string_view>
#include <fstream>
#include <sstream>
#include "Agate/Core/Log.h"
namespace Agate {

inline std::string ReadPTX(std::string_view fileName)
{
    std::ifstream input{fileName.data()};

    if (!input) {
        LOG_ERROR("读取 PTX 文件失败. {}", fileName);
        return {};
    }

    std::stringstream ptx;

    ptx << input.rdbuf();

    if (input.fail()) {
        LOG_ERROR("读取 PTX 文件失败. {}", fileName);
        return {};
    }
    
    return ptx.str();
}

} // namespace Agate