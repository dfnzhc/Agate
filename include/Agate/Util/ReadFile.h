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
    std::string path = "./ptx/";
    path += fileName;
    path += ".ptx";

    std::ifstream input{path};

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

inline bool FileExist(const char* filename)
{
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

inline std::string GetAssetPath(std::string_view assetName)
{
    std::string path = std::string(AGATE_ASSETS_PATH);
    path += '/';
    path += assetName;

    if (FileExist(path.c_str())) {
        return path;
    }
    
    throw AgateException("文件不存在.");
}

} // namespace Agate