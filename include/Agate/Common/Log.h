//
// Created by 秋鱼头 on 2022/4/10.
//

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#define LOGGER_FORMAT "[%^%l%$] %v"
#define PROJECT_NAME "Agate"

// Mainly for IDEs
#ifndef ROOT_PATH_SIZE
#	define ROOT_PATH_SIZE 0
#endif

#define __FILENAME__ (static_cast<const char *>(__FILE__) + ROOT_PATH_SIZE)

#define LOG_INFO(...)	spdlog::info(__VA_ARGS__);
#define LOG_WARN(...)	spdlog::warn(__VA_ARGS__);
#define LOG_ERROR(...)	spdlog::error("[{}:{}] {}", __FILENAME__, __LINE__, fmt::format(__VA_ARGS__));