//
// Created by 秋鱼头 on 2022/4/10.
//

#pragma once

#include <chrono>
#include <string>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace Agate {

class Timer
{
    // 创建一个新的 timer，并且重置
    Timer() { Reset(); }

    // 重置 timer 到当前时间
    void Reset() { start = std::chrono::system_clock::now(); }

    // 返回到上一次 Reset 之间的毫秒数
    double Elapsed() const
    {
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
        return (double) duration.count();
    }

    // 返回时间间隔的字符串
    std::string ElapsedString(bool precise = false) const
    {
        double time = Elapsed();

        if (std::isnan(time) || std::isinf(time))
            return "inf";

        std::string suffix = "ms";
        if (time > 1000) {
            time /= 1000; suffix = "s";
            if (time > 60) {
                time /= 60; suffix = "m";
                if (time > 60) {
                    time /= 60; suffix = "h";
                    if (time > 12) {
                        time /= 12; suffix = "d";
                    }
                }
            }
        }
        
        std::ostringstream os;
        os << std::setprecision(precise ? 4 : 1)
           << std::fixed << time << suffix;

        return os.str();
    }

private:
    std::chrono::system_clock::time_point start;
};

} // namespace Agate
