#include "PicoJSONUtilities.hpp"
#include "Logger.hpp"

#include <fstream>

namespace Code
{

  std::string PicoJSONToString(const picojson::value& params,const std::string& key,const std::string& default_value)
  {
    const picojson::object& params_obj=params.get<picojson::object>();
    if(params_obj.count(key))
      return params_obj.at(key).get<std::string>();
    return default_value;
  }

  bool PicoJSONToBoolean(const picojson::value& params,const std::string& key,const bool default_value)
  {
    const picojson::object& params_obj=params.get<picojson::object>();
    if(params_obj.count(key))
      return params_obj.at(key).get<bool>();
    return default_value;
  }

  template<typename T> T PicoJSONToScalar(const picojson::value& params,const std::string& key,const T default_value)
  {
    const picojson::object& params_obj=params.get<picojson::object>();
    if(params_obj.count(key))
      return params_obj.at(key).get<double>();
    return default_value;
  }

  template<typename T,int D> Eigen::Matrix<T,D,1> PicoJSONToVector(const picojson::value& params)
  {
    const picojson::array& params_arr=params.get<picojson::array>();
    Eigen::Matrix<T,D,1> r;
    for(int idx=0;idx<D;idx++)
      r(idx)=params_arr[idx].get<double>();
    return r;
  }

  template<typename T,int D> Eigen::Matrix<T,D,1> PicoJSONToVector(const picojson::value& params,const std::string& key,const Eigen::Matrix<T,D,1>& default_value)
  {
    const picojson::object& params_obj=params.get<picojson::object>();
    if(params_obj.count(key))
      return PicoJSONToVector<T,D>(params);
    return default_value;
  }

  template<typename T,int D> picojson::value PicoJSONFromVector(const Eigen::Matrix<T,D,1>& value)
  {
    picojson::array params_arr;
    for(int idx=0;idx<D;idx++)
      params_arr.push_back(picojson::value(value(idx)));
    return picojson::value(params_arr);
  }

  template<typename T> Eigen::Matrix<T,Eigen::Dynamic,1> PicoJSONToVectorX(const picojson::value& params)
  {
    const picojson::array& params_arr=params.get<picojson::array>();
    Eigen::Matrix<T,Eigen::Dynamic,1> r(params_arr.size());
    for(unsigned int idx=0;idx<params_arr.size();idx++)
      r(idx)=params_arr[idx].get<double>();
    return r;
  }

  template<typename T> picojson::value PicoJSONFromVectorX(const Eigen::Matrix<T,Eigen::Dynamic,1>& value)
  {
    picojson::array params_arr;
    for(unsigned int idx=0;idx<value.size();idx++)
      params_arr.push_back(picojson::value(value(idx)));
    return picojson::value(params_arr);
  }

  template<typename T>
  picojson::value PJFromStdVector(const std::vector<T>& value)
  {
    picojson::value pj=picojson::array();
    for(unsigned int idx=0;idx<value.size();idx++)
      pj.get<picojson::array>().push_back(picojson::value(value[idx]));
    return pj;
  }

  template<typename T>
  std::vector<T> PJToStdVector(const picojson::value& value)
  {
    std::vector<T> v;
    for(unsigned int idx=0;idx<value.get<picojson::array>().size();idx++)
      v.push_back((T)value.get<picojson::array>()[idx]);
    return v;
  }

#define InstantiationHelper(T)                                          \
  template picojson::value PJFromStdVector(const std::vector<T>& value); \
  template std::vector<T> PJToStdVector(const picojson::value& value);

  //InstantiationHelper(int);
  InstantiationHelper(float);
  InstantiationHelper(double);

  picojson::value ReadFileJSON(const std::string& path)
  {
    picojson::value params;
    std::ifstream file(path);
    std::istream_iterator<uint8_t> it(file);
    std::string err;
    picojson::parse(params,it,std::istream_iterator<uint8_t>(),&err);
    if(err.size())
    {
      Log("Error parsing %s: %s",path.c_str(),err.c_str());
      assert(!err.size());
    }
    return params;
  }

#define InstantiationHelperT(T)                                         \
  template T PicoJSONToScalar(const picojson::value& params,const std::string& key,const T default_value); \
  template Eigen::Matrix<T,Eigen::Dynamic,1> PicoJSONToVectorX(const picojson::value& params); \
  template picojson::value PicoJSONFromVectorX(const Eigen::Matrix<T,Eigen::Dynamic,1>& value);
  
#define InstantiationHelperTD(T,D)                                      \
  template Eigen::Matrix<T,D,1> PicoJSONToVector(const picojson::value& params); \
  template Eigen::Matrix<T,D,1> PicoJSONToVector(const picojson::value& params,const std::string& key,const Eigen::Matrix<T,D,1>& default_value); \
  template picojson::value PicoJSONFromVector(const Eigen::Matrix<T,D,1>& value); \

  InstantiationHelperT(int);
  InstantiationHelperT(float);
  InstantiationHelperT(double);

  InstantiationHelperTD(float,3);
  InstantiationHelperTD(double,3);

}
