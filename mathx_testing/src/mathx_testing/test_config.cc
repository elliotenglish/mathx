#include <mathx_core/log.h>

#include <cstdlib>
#include <string>

namespace code {
namespace testing {

bool DebugVisualize() {
  char* val = std::getenv("DEBUG_VISUALIZE");
  // log("DEBUG_VISUALIZE=%s", val ? val : "(UNSET)");
  if (val && std::string(val) == "1") return true;
  return false;
}

bool DebugWrite() {
  char* val = std::getenv("DEBUG_WRITE");
  // log("DEBUG_WRITE=%s", val ? val : "(UNSET)");
  if (val && std::string(val) == "1") return true;
  return false;
}

}  // namespace testing
}  // namespace code
