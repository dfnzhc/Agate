//
// Created by 秋鱼头 on 2022/4/8.
//

#pragma once

#include <string>
#include <vector>
#include "CGT/CGT.h"

namespace CGT {
// LogLevel Definition
enum class LogLevel { Verbose, Error, Fatal, Invalid };

std::string ToString(LogLevel level);
LogLevel LogLevelFromString(const std::string& s);

void ShutdownLogging();
void InitLogging(LogLevel level, std::string logFile, bool logUtilization, bool useGPU);

#ifdef CGT_USE_GPU

struct GPULogItem
{
    LogLevel level;
    char file[64];
    int line;
    char message[128];
};

std::vector<GPULogItem> ReadGPULogs();

#endif

// LogLevel Global Variable Declaration
namespace logging {
extern LogLevel logLevel;
extern FILE* logFile;
}  // namespace logging

// Logging Function Declarations
CGT_CPU_GPU
void Log(LogLevel level, const char* file, int line, const char* s);

CGT_CPU_GPU [[noreturn]] 
void LogFatal(LogLevel level, const char* file, int line,
                                       const char* s);

template<typename... Args>
CGT_CPU_GPU inline 
void Log(LogLevel level, const char* file, int line, const char* fmt,
                            Args&& ...args);

template<typename... Args>
CGT_CPU_GPU [[noreturn]] inline 
void LogFatal(LogLevel level, const char* file, int line,
                                              const char* fmt, Args&& ...args);

#define TO_STRING(x) TO_STRING2(x)
#define TO_STRING2(x) #x

#ifdef CGT_USE_GPU

extern __constant__ LogLevel LOGGING_LogLevelGPU;
#define LOG_VERBOSE(...)                               \
    (pbrt::LogLevel::Verbose >= LOGGING_LogLevelGPU && \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_ERROR(...)                               \
    (pbrt::LogLevel::Error >= LOGGING_LogLevelGPU && \
     (pbrt::Log(LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_FATAL(...) \
    pbrt::LogFatal(pbrt::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

#else

// Logging Macros
#define LOG_VERBOSE(...)                             \
    (pbrt::LogLevel::Verbose >= logging::logLevel && \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_ERROR(...)                                   \
    (pbrt::LogLevel::Error >= pbrt::logging::logLevel && \
     (pbrt::Log(LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_FATAL(...) \
    pbrt::LogFatal(pbrt::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

#endif
} // namespace CGT