//
// Created by 秋鱼头 on 2022/4/10.
//

#pragma once

namespace filesystem {
class path;
class resolver;
}

#include <spdlog/fmt/fmt.h>
#include <optix_types.h>
#include <optix_stubs.h>
namespace Agate {

#define EPS_F = 1e-5;
#define INF_F = std::numeric_limits<float>::infinity();
#define EPS_D = 1e-16;
#define INF_D = std::numeric_limits<double>::infinity();

/// for debugging purposes
using std::cout;
using std::cerr;
using std::endl;

/// Simple exception class
class AgateException : public std::runtime_error
{
public:
    template<typename... Args>
    explicit AgateException(std::string_view fmt_str, const Args& ... args)
        : std::runtime_error(fmt::format(fmt_str, args...)) {}
        
    explicit AgateException( OptixResult res, std::string_view msg )
         : std::runtime_error(CreateMessage( res, msg ).c_str()) {}
        
private:
    inline std::string CreateMessage(OptixResult res, std::string_view msg)
    {
        std::ostringstream out;
        out << optixGetErrorName(res) << ": " << msg;
        return out.str();
    }
};

} // namespace Agate
 