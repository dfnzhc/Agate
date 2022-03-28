//
// Created by 秋鱼头 on 2022/3/28.
//

#pragma once

namespace CGT {
/**
 * \brief Simple timer with millisecond precision
 *
 * This class is convenient for collecting performance data
 */
class Timer
{
public:
    /// Create a new timer and Reset it
    Timer();

    /// Reset the timer to the current time
    void Reset();

    /// Return the number of milliseconds Elapsed since the timer was last Reset
    double Elapsed() const;

    /// Like \ref Elapsed(), but return a human-readable string
    std::string ElapsedString(bool precise = false) const;
    
private:
    std::chrono::system_clock::time_point start;
};
} // namespace CGT

 