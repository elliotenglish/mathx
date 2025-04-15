#pragma once

#include "log.h"

#define MathXMsg(HEADING, MSG)                          \
  {                                                    \
    MathX::log("---------- %s ----------",HEADING);      \
    MathX::log("Message=%s", std::string(MSG).c_str()); \
    MathX::log("Line=(%s:%d)", __FILE__, __LINE__);       \
    MathX::log("Function=%s", __PRETTY_FUNCTION__);     \
  }

#define MathXAssertMsg(EXPR, MSG)         \
  {                                      \
    if (!(EXPR)) {                       \
      MathXMsg("Assertion failed!", MSG); \
      MathX::log("Expression=%s", #EXPR); \
      throw std::exception();            \
    }                                    \
  }

#define MathXAssert(EXPR) MathXAssertMsg(EXPR, "")
#define MathXAssertNotImplemented() MathXAssertMsg(false, "Not implemented.")
#define MathXWarning(MSG) MathXMsg("Warning!", MSG)
