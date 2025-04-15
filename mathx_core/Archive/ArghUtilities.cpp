#include "ArghUtilities.hpp"

namespace Code
{

  picojson::value ArghToPicoJSON(const argh::parser& parser)
  {
    picojson::value args=picojson::object();

    picojson::value positional_args=picojson::array();
    for(auto& pos_arg:parser.pos_args())
      positional_args.get<picojson::array>().push_back(picojson::value(pos_arg));
    args.get<picojson::object>()["positional"]=positional_args;

    for(auto& flag:parser.flags())
      args.get<picojson::object>()[flag]=true;

    for(auto& param:parser.params())
    {
      try
      {
        args.get<picojson::object>()[param.first]=std::stod(param.second);
      }
      catch(std::exception& e)
      {
        args.get<picojson::object>()[param.first]=param.second;
      }
    }

    return args;
  }
  
  int ArghToInt(const argh::parser& parser,const std::string& key,const int default_value)
  {
    if(parser.params().count(key))
    {
      int value=-1;
      parser(key) >> value;
      return value;
    }
    return default_value;
  }

  std::string ArghToString(const argh::parser& parser,const std::string& key,const std::string& default_value)
  {
    if(parser.params().count(key))
    {
      std::string value;
      parser(key) >> value;
      return value;
    }
    return default_value;
  }

  bool ArghToBool(const argh::parser& parser,const std::string& key)
  {
    return parser["--"+key];
  }

}
