#pragma once

#include <stdarg.h>

#include <Eigen/Dense>
#include <iostream>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mathx {

//////////////////////////////////////////////////////////////////////////////
// F
#pragma GCC diagnostic ignored "-Wformat-security"
template <typename... Args> std::string F(Args... args) {
  int len = snprintf(NULL, 0, args...);
  char buf[len + 1];
  snprintf(buf, len + 1, args...);
  return buf;
}

template <typename... Args> std::string F(std::string &fmt, Args... args) {
  return F(fmt.c_str(), args...);
}

std::string header(const std::string header, const int width = 80);

//////////////////////////////////////////////////////////////////////////////
// operator<<

template <size_t n, typename... T>
typename std::enable_if<(n >= sizeof...(T))>::type
print_tuple(std::ostream &, const std::tuple<T...> &) {}

template <size_t n, typename... T>
typename std::enable_if<(n < sizeof...(T))>::type
print_tuple(std::ostream &os, const std::tuple<T...> &tup) {
  if (n != 0)
    os << ",";
  os << std::get<n>(tup);
  print_tuple<n + 1>(os, tup);
}

template <typename... T>
std::ostream &operator<<(std::ostream &os, const std::tuple<T...> &tup) {
  os << "[";
  print_tuple<0>(os, tup);
  os << "]";
  return os;
}

//////////////////////////////////////////////////////////////////////////////
// Str

template <typename T> std::string str(const T &x) {
  std::stringstream o;
  o << x;
  return o.str();
}

template <typename Derived>
std::string str(const Eigen::MatrixBase<Derived> &x) {
  std::stringstream st;
  st << "[";
  for (int idx0 = 0; idx0 < x.rows(); idx0++) {
    st << (idx0 > 0 ? "," : "") << "[";
    for (int idx1 = 0; idx1 < x.cols(); idx1++)
      st << (idx1 > 0 ? "," : "") << (typename Derived::Scalar)(x(idx0, idx1));
    st << "]";
  }
  st << "]";
  return st.str();
}

template <typename T, int D1>
std::string str(const Eigen::MatrixBase<Eigen::Matrix<T, D1, 1>> &x) {
  std::stringstream st;
  st << "[";
  for (int idx0 = 0; idx0 < x.rows(); idx0++)
    st << (idx0 > 0 ? "," : "") << x(idx0);
  st << "]";
  return st.str();
}

template <typename T, int D1, int D2>
std::string str(const Eigen::Matrix<T, D1, D2> &x) {
  return str((Eigen::MatrixBase<Eigen::Matrix<T, D1, D2>> &)x);
}

template <typename T> std::string str(const std::vector<T> &x) {
  std::stringstream os;
  os << "[";
  for (std::size_t idx = 0; idx < x.size(); idx++)
    os << (idx ? "," : "") << str(x[idx]);
  os << "]";
  return os.str();
}

template <typename K, typename V>
std::string str(const std::unordered_map<K, V> &x, const bool ordered = false,
                const char spacer = '\n') {
  std::stringstream os;
  os << "{" << spacer;
  if (ordered) {
    typename std::set<K> keys;
    for (typename std::unordered_map<K, V>::const_iterator it = x.begin();
         it != x.end(); it++)
      keys.insert(it->first);
    for (typename std::set<K>::iterator it = keys.begin(); it != keys.end();
         it++) {
      if (it != keys.begin())
        os << ',' << spacer;
      os << str(*it) << ":" << str(x.at(*it));
    }
  } else {
    for (typename std::unordered_map<K, V>::const_iterator it = x.begin();
         it != x.end(); it++) {
      if (it != x.begin())
        os << ',' << spacer;
      os << str(it->first) << ":" << str(it->second);
    }
  }
  os << spacer << "}";
  return os.str();
}

template <typename K>
std::string str(const std::unordered_set<K> &x, const bool ordered = false,
                const char spacer = '\n') {
  std::stringstream os;
  os << "{" << spacer;
  if (ordered) {
    typename std::set<K> keys;
    for (typename std::unordered_set<K>::const_iterator it = x.begin();
         it != x.end(); it++)
      keys.insert(*it);
    for (typename std::set<K>::iterator it = keys.begin(); it != keys.end();
         it++) {
      if (it != keys.begin())
        os << ',' << spacer;
      os << str(*it);
    }
  } else {
    for (typename std::unordered_set<K>::const_iterator it = x.begin();
         it != x.end(); it++) {
      if (it != x.begin())
        os << ',' << spacer;
      os << str(*it);
    }
  }
  os << spacer << "}";
  return os.str();
}

template <typename T> std::string str_idx(const T &x, const char spacer = ' ') {
  return str(x);
}

template <typename T>
std::string str_idx(const std::vector<T> &x, const char spacer = ' ') {
  std::ostringstream o;
  o << "[";
  for (int idx = 0; idx < x.size(); idx++)
    o << (idx ? "," : "") << spacer << idx << "=" << str_idx(x[idx], spacer);
  o << spacer << "]";
  return o.str();
}

inline std::string to_json(const double x) { return str(x); }

inline std::string to_json(const std::string &x) { return x; }

template <typename T> std::string to_json(const std::vector<T> &x) {
  std::ostringstream o;
  o << "[";
  for (int idx = 0; idx < x.size(); idx++)
    o << (idx ? "," : "") << JSON(x[idx]);
  o << "]";
  return o.str();
}

std::string ProgramHeader(const char *name, const int argc, const char **argv);

//////////////////////////////////////////////////////////////////////////////
// Log
void log_implementation(const std::string &x);

// This is the main logging method
template <typename... Args> void log(Args... args) {
  log_implementation(F(args...) + "\n");
}

// Helper for logging variables with name

#define log_var(X, ...)                                                        \
  mathx::log("%s=%s", #X, mathx::str(X __VA_OPT__(, ) __VA_ARGS__).c_str());

} // namespace mathx
