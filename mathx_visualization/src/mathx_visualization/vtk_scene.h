#pragma once

#include <vtkActor.h>
#include <vtkDataSet.h>
#include <vtkProp.h>

#include <Eigen/Dense>

namespace mathx {
namespace visualization {

class VTKScene {
 public:
  enum DrawMode { DrawFaces = 0, DrawEdges = 1, DrawVertices = 2 };

  std::vector<vtkSmartPointer<vtkProp>> props;

  static void SetActorTransform(vtkActor* actor,
                                const Eigen::Affine3f& transform);

  static vtkSmartPointer<vtkProp> GenerateObject(
      const vtkSmartPointer<vtkDataSet>& data, const DrawMode draw_mode,
      const Eigen::Affine3f& transform);

  static vtkSmartPointer<vtkProp> GenerateText(const std::string& text,
                                               const Eigen::Vector3f& origin,
                                               const float scale);
};

}  // namespace visualization
}  // namespace mathx
