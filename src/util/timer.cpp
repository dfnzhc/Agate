//
// Created by 秋鱼头 on 2022/3/28.
//

#include "CGT/util/timer.h"
#include "CGT/util/uString.h"

namespace CGT {

Timer::Timer()
{
    Reset();
}

void Timer::Reset()
{
    start = std::chrono::system_clock::now();
}

double Timer::Elapsed() const
{
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    return (double) duration.count();
}

std::string Timer::ElapsedString(bool precise) const
{
    return timeString(Elapsed(), precise);
}

} // namespace CGT
