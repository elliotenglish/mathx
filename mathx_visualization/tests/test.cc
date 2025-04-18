#include <mathx_core/log.h>
#include <mathx_visualization/vtk_display_2d.h>
#include <mathx_visualization/vtk_display_3d.h>
#include <mathx_visualization/vtk_geometry.h>

#include <Eigen/Dense>

#include "test_utilities.h"

using namespace mathx;
using namespace mathx::visualization;

typedef Eigen::Vector3f V3;

int main(int argc, char** argv) {
  /////////////////////////////////////////////////////////
  // 3D Display
  VTKDisplay3D display_3d(1024, 768);

  log("building geometry");
  std::vector<V3> position;
  std::vector<V3> color;
  std::vector<float> radius;

  position.push_back(V3(.1, -.2, .3));
  color.push_back(V3(1, 0, 0));
  radius.push_back(.2);

  position.push_back(V3(.3, .4, -.1));
  color.push_back(V3(0, 0, 1));
  radius.push_back(.3);

  log("Adding particles");
  vtkSmartPointer<vtkUnstructuredGrid> particles =
      convert_particles_to_vtk(position, color, radius);
  display_3d.AddProp(VTKScene::GenerateObject(particles, VTKScene::DrawVertices,
                                              Eigen::Affine3f::Identity()));

  log("Adding pyramid");
  vtkSmartPointer<vtkPolyData> mesh = generate_pyramid();
  Eigen::Affine3f transform;
  transform.setIdentity();
  transform.prerotate(Eigen::AngleAxis<float>(.1, Eigen::Vector3f(1, 0, 0)));
  transform.pretranslate(V3(0, 0, 1));
  display_3d.AddProp(
      VTKScene::GenerateObject(mesh, VTKScene::DrawEdges, transform));

  log("Adding text");
  display_3d.AddProp(VTKScene::GenerateText("some text", V3(-.4, .1, -.9), 1));

  display_3d.FrameCamera();

  /////////////////////////////////////////////////////////
  // 2D Display
  VTKDisplay2D display_2d(640, 480);

  cv::Mat image = GenerateTestImage(640, 480);
  display_2d.SetImage(image, true);

  /////////////////////////////////////////////////////////
  // Event loop
  int it = 0;
  while (true) {
    std::vector<VTKEvent> events3 = display_3d.EventLoopOnce();
    std::vector<VTKEvent> events2 = display_2d.EventLoopOnce();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    log("event loop % 5d events3=% 2d events2=% 2d", it, events3.size(),
        events2.size());
    it++;
  }

  return 0;
}
