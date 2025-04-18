#include "vtk_display_3d.h"

#include <mathx_core/log.h>
#include <vtkAxesActor.h>
#include <vtkCommand.h>
#include <vtkDataSetMapper.h>
#include <vtkFollower.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include <chrono>
#include <thread>

namespace mathx {
namespace visualization {
typedef Eigen::Vector3f V3;

/**
 * https://discourse.slicer.org/t/how-can-i-disable-automatic-reset-of-camera-clipping-range/10839/3
 *
 * As noted above we need to set the clipping range in a camera observer.
 */
void VTKCameraObserver::Execute(vtkObject* caller, unsigned long eventId,
                                void* callData) {
  vtkCamera* camera = (vtkCamera*)caller;
  camera->SetClippingRange(0.1, 1000);
}

VTKDisplay3D::VTKDisplay3D(const int width, const int height)
    : VTKDisplayBase(
          width, height,
          new VTKInteractorStyle<vtkInteractorStyleTrackballCamera>(this)) {
  camera = vtkCamera::New();
  camera_observer = new VTKCameraObserver;
  camera->AddObserver(vtkCommand::AnyEvent, camera_observer);
  renderer->SetActiveCamera(camera);
}

VTKDisplay3D::~VTKDisplay3D() {}

void VTKDisplay3D::FrameBoundingBox(const V3& x_min, const V3& x_max) {
  // log("framing bounding box");
  // V3 bounds_min(bounds[0], bounds[2], bounds[4]);
  // V3 bounds_max(bounds[1], bounds[3], bounds[5]);
  V3 center = (x_min + x_max) * .5;
  float distance = (x_max - x_min).norm();

  // log_var(x_min);
  // log_var(x_max);
  // log_var(center);
  // log_var(distance);

  camera->SetPosition(center(0) - distance, center(1) - distance,
                      center(2) - distance);
  camera->SetFocalPoint(center(0), center(1), center(2));
  camera->SetViewUp(0, -1, 0);
  // camera->SetViewAngle(90);
}

void VTKDisplay3D::FrameCamera() {
  // std::lock_guard guard(mutex);
  // V3 x_min(FLT_MAX,FLT_MAX,FLT_MAX);
  // V3 x_max(-FLT_MAX,-FLT_MAX,-FLT_MAX);

  // for(int i=0;i<renderer->GetActors()
  renderer->GetActiveCamera()->Azimuth(50);
  renderer->GetActiveCamera()->Elevation(-30);
  renderer->ResetCamera();
}

void VTKDisplay3D::Clear() {
  VTKDisplayBase::Clear();
  AddProp(vtkSmartPointer<vtkProp>(vtkAxesActor::New()));
}

void VTKDisplay3D::SetScene(const VTKScene& scene,const bool frame_scene) {
  for (int i = 0; i < scene.props.size(); i++) AddProp(scene.props[i]);
  if(frame_scene)
    FrameBoundingBox(V3(-2,-2,-2),V3(2,2,2));
}

}  // namespace visualization
}  // namespace mathx
