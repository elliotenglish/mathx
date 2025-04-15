#include "time.h"

#include <chrono>
#include <thread>

namespace mathx {

Time GetTime() {
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::time_point<Clock> TimePoint;
  static TimePoint start = Clock::now();
  TimePoint current = Clock::now();
  std::chrono::duration<double> diff = current - start;
  return diff.count();
}

void Sleep(const Time &dt) {
  std::this_thread::sleep_for(std::chrono::microseconds((int)(dt * 1e6)));
}

} // namespace mathx
