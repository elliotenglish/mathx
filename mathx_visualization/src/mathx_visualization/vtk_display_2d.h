#pragma once

#include <stdint.h>

#include <opencv2/core.hpp>

#include "vtk_display_base.h"

namespace mathx {
namespace visualization {

class VTKDisplay2D : public VTKDisplayBase {
 public:
  VTKDisplay2D(const int width, const int height);
  ~VTKDisplay2D();

  void SetImage(const uint8_t* bytes, const int width, const int height,const bool fit_window);
  void SetImage(cv::Mat& image,const bool fit_window);

 private:
  VTKDisplay2D(const VTKDisplay2D&);
};

}  // namespace visualization
}  // namespace mathx
