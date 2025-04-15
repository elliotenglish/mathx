#include "parameters.h"

namespace mathx {

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
JSONToEigen(const nlohmann::json &json) {
  if (json[0].is_array()) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x(json.size(),
                                                       json[0].size());
    for (int i = 0; i < json.size(); i++)
      for (int j = 0; j < json[0].size(); j++)
        x(i, j) = json[i][j];
    return x;
  } else {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x(json.size(), 1);
    for (int i = 0; i < json.size(); i++)
      x(i) = json[i];
    return x;
  }
}

template <typename T>
nlohmann::json
EigenToJSON(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data) {
  if (data.cols() == 1) {
    nlohmann::json json;
    for (int i = 0; i < data.rows(); i++)
      json.push_back(data(i));
    return json;
  } else {
    nlohmann::json json;
    for (int i = 0; i < data.rows(); i++) {
      nlohmann::json json_inner;
      for (int j = 0; j < data.cols(); j++)
        json_inner.push_back(data(i, j));
      json.push_back(json_inner);
    }
    return json;
  }
}

const nlohmann::json &JSONGetDefault(const nlohmann::json &json,
                                     const std::string &key,
                                     const nlohmann::json &def) {
  if (json.count(key) >= 1)
    return json[key];
  else
    return def;
}

#define INSTANTIATION_HELPER(T)                                                \
  template Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> JSONToEigen(       \
      const nlohmann::json &json);                                             \
  template nlohmann::json EigenToJSON(                                         \
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data);

INSTANTIATION_HELPER(float);
INSTANTIATION_HELPER(double);
INSTANTIATION_HELPER(int);

} // namespace mathx
