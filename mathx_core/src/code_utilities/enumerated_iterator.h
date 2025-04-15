#pragma once

namespace mathx {

template <typename It> class EnumeratedIterator : public It {
private:
  int idx;

public:
  EnumeratedIterator(const It &other) : It(other), idx(0) {}

  int Index() const { return idx; }

  EnumeratedIterator operator++() {
    It::operator++(0);
    idx++;
    return *this;
  }

  EnumeratedIterator operator++(int) {
    EnumeratedIterator it = *this;
    It::operator++(0);
    idx++;
    return it;
  }
};

} // namespace mathx
