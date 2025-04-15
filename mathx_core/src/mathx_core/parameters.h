#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace mathx {
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
JSONToEigen(const nlohmann::json &json);

template <typename T>
nlohmann::json
EigenToJSON(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &data);

const nlohmann::json &JSONGetDefault(const nlohmann::json &json,
                                     const std::string &key,
                                     const nlohmann::json &def);
} // namespace mathx
