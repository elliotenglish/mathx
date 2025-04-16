#include "log.h"

#include <iomanip>
#include <iostream>
#include <mutex>

#include "time.h"

namespace mathx {

std::string header(const std::string header, const int width) {
  char sep = '-';
  int sep_width = width - header.size() - 2;
  int begin_sep_width = sep_width / 2;
  int end_sep_width = sep_width - begin_sep_width;
  std::string r;
  for (int idx = 0; idx < begin_sep_width; idx++)
    r += sep;
  r += " " + header + " ";
  for (int idx = 0; idx < end_sep_width; idx++)
    r += sep;
  return r;
}

void log_implementation(const std::string &x) {
  static std::mutex log_mutex;
  std::lock_guard<std::mutex> lock(log_mutex);

  bool last_char_newline = true;
  for (unsigned int idx = 0; idx < x.size(); idx++) {
    if (last_char_newline)
      std::cout << "[" << std::setw(12) << std::setprecision(9) << GetTime()
                << "] ";

    last_char_newline = (x[idx] == '\n');
    std::cout << x[idx];
  }
  std::cout << std::flush;
}

std::string ProgramHeader(const char *name, const int argc, const char **argv) {
  std::stringstream ss;
  ss << name << std::endl;
  for (int i = 0; i < argc; i++)
    ss << argv[i] << " ";
  return ss.str();
}

} // namespace mathx
