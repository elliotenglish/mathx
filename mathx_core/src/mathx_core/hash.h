#pragma once

#include <functional>

namespace mathx {

inline int hash(const int x0, const int x1) {
  size_t h = (size_t(x0) << 32) + size_t(x1);
  h *= 1231231557ull; // "random" uneven integer
  h ^= (h >> 32);
  return (int)h;
}

// class HashTII
// {
// public:
//   size_t operator()(const std::tuple<int,int>& v) const
//   {
//     return Hash(std::get<0>(v),std::get<1>(v));
//   }
// };

} // namespace mathx

namespace std {
template <> struct hash<std::tuple<int, int>> {
  size_t operator()(const std::tuple<int, int> &v) const {
    return mathx::hash(std::get<0>(v), std::get<1>(v));
  }
};
} // namespace std
