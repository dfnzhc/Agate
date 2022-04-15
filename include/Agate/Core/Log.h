//
// Created by 秋鱼头 on 2022/4/10.
//

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace Agate {

class Log final
{
public:
    Log()
    {
        Init();
    }

    ~Log()
    {
        Shutdown();
    }

    static void Init()
    {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("%^[%T]: %v%$");
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs.txt", true);
        file_sink->set_pattern("[%T] [%l]: %v");

        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
        auto logger = std::make_shared<spdlog::logger>("Agate", sinks.begin(), sinks.end());
        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::trace);
        spdlog::register_logger(logger);
    }

    static void Shutdown()
    {

        spdlog::shutdown();
    }
};

} // namespace Agate


// Mainly for IDEs
#ifndef ROOT_PATH_SIZE
#	define ROOT_PATH_SIZE 0
#endif

#define __FILENAME__ (static_cast<const char *>(__FILE__) + ROOT_PATH_SIZE)

#define LOG_TRACE(...)    if (spdlog::get("Agate") != nullptr) {spdlog::get("Agate")->trace(__VA_ARGS__);}
#define LOG_DEBUG(...)    if (spdlog::get("Agate") != nullptr) {spdlog::get("Agate")->debug(__VA_ARGS__);}
#define LOG_INFO(...)    if (spdlog::get("Agate") != nullptr) {spdlog::get("Agate")->info(__VA_ARGS__);}
#define LOG_WARN(...)    if (spdlog::get("Agate") != nullptr) {spdlog::get("Agate")->warn(__VA_ARGS__);}
#define LOG_ERROR(...)    if (spdlog::get("Agate") != nullptr) \
    {spdlog::get("Agate")->error("[{}:{}] {}", __FILENAME__, __LINE__, fmt::format(__VA_ARGS__));}