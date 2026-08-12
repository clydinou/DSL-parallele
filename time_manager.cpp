#include "time_manager.hpp"

void top_(TimePoint* tp) {
    *tp = std::chrono::steady_clock::now();
}

double cpu_time_(TimePoint t1, TimePoint t2) {
    std::chrono::duration<double> elapsed = t2 - t1;
    return elapsed.count(); // durée en secondes
}

unsigned long get_temp_residuel(void) {
    // Résidu quasi-nul avec steady_clock
    TimePoint a = std::chrono::steady_clock::now();
    TimePoint b = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(b - a);
    return (unsigned long)diff.count();
}
