#pragma once

#include "log.h"

namespace mathx {

class FunctionScope_ {
public:
  std::string function_symbol;

  FunctionScope_(const char *function_symbol_)
      : function_symbol(function_symbol_) {
    log(function_symbol + "() begin");
  }

  ~FunctionScope_() { log(function_symbol + "() end"); }
};

#define function_scope() FunctionScope_ function_scope_(__FUNCTION__)
// #define FunctionScope() FunctionScope_ function_scope_(__PRETTY_FUNCTION__)

} // namespace mathx
