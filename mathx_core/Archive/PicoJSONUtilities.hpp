#pragma once

//#define PICOJSON_USE_INT64

#include <picojson.h>
#include <Eigen/Dense>

namespace Code
{

  // template<typename T> T PicoJSONGet(const picojson::value& params,const std::string& key,const T& default_value);
  // template<typename T> void PicoJSONSet(picojson::value& params,const std::string& key,const T& value);
  // template<typename T> void PicoJSONContains(const picojson::value& params,const std::string& key);

  std::string PicoJSONToString(const picojson::value& params,const std::string& key,const std::string& default_value);
  bool PicoJSONToBoolean(const picojson::value& params,const std::string& key,const bool default_value);
  template<typename T> T PicoJSONToScalar(const picojson::value& params,const std::string& key,const T default_value);

//Fixed length vector
  template<typename T,int D> Eigen::Matrix<T,D,1> PicoJSONToVector(const picojson::value& params);
  template<typename T,int D> Eigen::Matrix<T,D,1> PicoJSONToVector(const picojson::value& params,const std::string& key,const Eigen::Matrix<T,D,1>& default_value);
  template<typename T,int D> picojson::value PicoJSONFromVector(const Eigen::Matrix<T,D,1>& value);

//Dynamic length vector
  template<typename T>
  Eigen::Matrix<T,Eigen::Dynamic,1> PicoJSONToVectorX(const picojson::value& params);
  template<typename T>
  picojson::value PicoJSONFromVectorX(const Eigen::Matrix<T,Eigen::Dynamic,1>& value);

  template<typename T>
  picojson::value PJFromStdVector(const std::vector<T>& value);
  template<typename T>
  std::vector<T> PJToStdVector(const picojson::value& value);

  picojson::value ReadFileJSON(const std::string& path);
  // void WriteFileJSON(const std::string& path,const picojson::value& value);

}
