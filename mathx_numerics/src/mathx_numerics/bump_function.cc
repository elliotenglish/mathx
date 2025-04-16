#include "bump_function.h"

#include <cmath>

namespace code {
namespace numerics {

float BumpFunction(const float x)
{
  return std::exp(1/(x*x-1));
}

}
}  // namespace code
