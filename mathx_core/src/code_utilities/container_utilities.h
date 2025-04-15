#pragma once

namespace mathx {

template <typename K, typename V>
std::vector<K> GetKeys(const std::unordered_map<K, V> &map) {
  std::vector<K> keys;
  for (typename std::unordered_map<K, V>::const_iterator it = map.cbegin();
       it != map.cend(); it++)
    keys.push_back(it->first);
  return keys;
}

template <typename T> T Min(const std::vector<T> &vector) {
  T min_value = vector[0];
  for (typename std::vector<T>::const_iterator it = vector.cbegin();
       it != vector.cend(); it++)
    min_value = std::min(min_value, *it);
  return min_value;
}

template <typename T> T Max(const std::vector<T> &vector) {
  T min_value = vector[0];
  for (typename std::vector<T>::const_iterator it = vector.cbegin();
       it != vector.cend(); it++)
    min_value = std::max(min_value, *it);
  return min_value;
}

} // namespace mathx
