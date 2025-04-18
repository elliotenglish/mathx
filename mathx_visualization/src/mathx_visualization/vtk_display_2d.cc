#include "vtk_display_2d.h"

#include <mathx_core/assert.h>
#include <vtkCamera.h>
#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkInteractorStyle.h>

#include <opencv2/imgproc.hpp>

namespace mathx {
namespace visualization {

VTKDisplay2D::VTKDisplay2D(const int width, const int height)
    : VTKDisplayBase(width, height,
                     new VTKInteractorStyle<vtkInteractorStyle>(this)) {
  vtkCamera* camera = renderer->GetActiveCamera();
  camera->ParallelProjectionOn();
  // camera->SetViewUp(0,-1,0);
}

VTKDisplay2D::~VTKDisplay2D() {}

void VTKDisplay2D::SetImage(const uint8_t* bytes, const int width,
                            const int height, const bool fit_window) {
  Clear();

  vtkNew<vtkImageData> imageData;
  imageData->SetDimensions(width, height, 1);
  imageData->SetSpacing(1, -1, 1);
  imageData->SetOrigin(0, 0, 0);
  imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
  memcpy(imageData->GetScalarPointer(), bytes, 3 * width * height);

  vtkNew<vtkImageActor> imageActor;
  imageActor->SetInputData(imageData);

  AddProp(imageActor);

  vtkCamera* camera = renderer->GetActiveCamera();
  float d = camera->GetDistance();

  float xc = 0.5 * (width - 1);
  float yc = 0.5 * (height - 1);
  //  float xd = (extent[1] - extent[0] + 1)*spacing[0]; // not used
  float yd = height;

  camera->SetParallelScale(0.5f * static_cast<float>(yd));
  camera->SetFocalPoint(xc, -yc, 0.0);
  camera->SetPosition(xc, -yc, d);

  if (fit_window) window->SetSize(width, height);
}

void VTKDisplay2D::SetImage(cv::Mat& image, const bool fit_window) {
  cv::Mat image_uint8_rgb(image.rows, image.cols, CV_8UC3);
  if (image.type() == CV_8UC3)
    cv::cvtColor(image, image_uint8_rgb, cv::COLOR_BGR2RGB);
  if (image.type() == CV_32FC3)
    cv::cvtColor(image * 255., image_uint8_rgb, cv::COLOR_BGR2RGB);
  SetImage(image_uint8_rgb.data, image.cols, image.rows, fit_window);
}

}  // namespace visualization
}  // namespace mathx
