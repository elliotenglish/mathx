#pragma once

#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkDataSet.h>

#include <Eigen/Dense>
#include <mutex>
#include <thread>

#include "vtk_display_base.h"
#include "vtk_scene.h"

namespace mathx {
namespace visualization {

class VTKCameraObserver : public vtkCommand {
 public:
  void Execute(vtkObject* caller, unsigned long eventId, void* callData);
};

class VTKDisplay3D : public VTKDisplayBase {
 public:
  vtkSmartPointer<VTKCameraObserver> camera_observer;
  vtkSmartPointer<vtkCamera> camera;

  VTKDisplay3D(const int width, const int height);
  ~VTKDisplay3D();
  void FrameBoundingBox(const Eigen::Vector3f& x_min,
                        const Eigen::Vector3f& x_max);
  void FrameCamera();

  void Clear();
  void SetScene(const VTKScene& scene, const bool frame_scene);

 private:
  VTKDisplay3D(const VTKDisplay3D&);
};

}  // namespace visualization
}  // namespace mathx
