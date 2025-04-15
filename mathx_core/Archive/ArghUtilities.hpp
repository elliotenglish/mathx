#pragma once

#include <argh.h>
#include <picojson.h>

namespace Code
{

  picojson::value ArghToPicoJSON(const argh::parser& parser);
  int ArghToInt(const argh::parser& parser,const std::string& key,const int default_value);
  std::string ArghToString(const argh::parser& parser,const std::string& key,const std::string& default_value);
  bool ArghToBool(const argh::parser& parser,const std::string& key);

}
