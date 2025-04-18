#include "vtk_geometry.h"

#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkTriangle.h>

#include <mathx_numerics/principal_component_analysis.h>
#include <mathx_core/log.h>

typedef Eigen::Vector3f V3;
typedef Eigen::Vector3i V3i;
typedef Eigen::Matrix3f M3;

namespace mathx {
namespace visualization {

vtkSmartPointer<vtkUnstructuredGrid> convert_particles_to_vtk(
    std::vector<Eigen::Vector3f>& position, std::vector<Eigen::Vector3f>& color,
    std::vector<float>& radius) {
  vtkNew<vtkPoints> points;
  vtkNew<vtkCellArray> vertices;

  vtkNew<vtkDoubleArray> id_vec;
  id_vec->SetNumberOfComponents(1);
  id_vec->SetName("id");

  vtkNew<vtkUnsignedCharArray> color_vec;
  color_vec->SetNumberOfComponents(3);
  color_vec->SetName("color");

  vtkNew<vtkDoubleArray> radius_vec;
  radius_vec->SetNumberOfComponents(1);
  radius_vec->SetName("radius");

  // HACK: REMOVE POINTS NOT WITH 10 STDDEVS
  V3 eigenvalues=V3::Zero();
  M3 eigenvectors=M3::Zero();
  V3 mean=V3::Zero();
  numerics::PrincipalComponentAnalysis((float*)position.data(), 3, position.size(),
                                       mean.data(), eigenvectors.data(),
                                       eigenvalues.data());
  float stddev=sqrt(eigenvalues[2]);
  // log_var(mean);
  // log_var(stddev);

  for (int i = 0; i < position.size(); i++) {
    if ((position[i] - mean).norm() > 10 * stddev) continue;

    vtkIdType id =
        points->InsertNextPoint(position[i](0), position[i](1), position[i](2));
    vtkIdType ids[] = {id};
    vertices->InsertNextCell(1, ids);

    double id_ = id;
    id_vec->InsertNextTuple(&id_);
    Eigen::Matrix<uint8_t, 3, 1> c = (color[i] * 255.).cast<uint8_t>();
    color_vec->InsertNextTypedTuple(c.data());
    double r = radius[i];
    radius_vec->InsertNextTuple(&r);
  }

  // log_var(position.size());
  // log_var(points->GetNumberOfPoints());

  vtkNew<vtkUnstructuredGrid> grid;
  grid->SetPoints(points);
  grid->SetCells(VTK_VERTEX, vertices);
  // grid->GetPointData()->AddArray(id_vec);
  // grid->GetPointData()->AddArray(color_vec);
  // grid->GetPointData()->AddArray(radius_vec);
  grid->GetPointData()->SetScalars(color_vec);

  return grid;
}

vtkSmartPointer<vtkPolyData> convert_mesh_to_vtk(
    std::vector<Eigen::Vector3f>& position,
    std::vector<Eigen::Vector3i>& triangle) {
  vtkNew<vtkPoints> verts;
  for (int i = 0; i < position.size(); i++)
    verts->InsertNextPoint(position[i](0), position[i](1), position[i](2));

  vtkNew<vtkCellArray> tris;
  for (int i = 0; i < triangle.size(); i++) {
    vtkNew<vtkTriangle> tri;
    tri->GetPointIds()->SetId(0, triangle[i][0]);
    tri->GetPointIds()->SetId(1, triangle[i][1]);
    tri->GetPointIds()->SetId(2, triangle[i][2]);
    tris->InsertNextCell(tri);
  }

  vtkNew<vtkPolyData> data;
  data->SetPoints(verts);
  data->SetPolys(tris);

  return data;
}

vtkSmartPointer<vtkPolyData> generate_pyramid() {
  std::vector<Eigen::Vector3f> position;
  std::vector<Eigen::Vector3i> triangle;

  position.push_back(V3(0, 0, 0));
  position.push_back(V3(-1, -1, 1));
  position.push_back(V3(1, -1, 1));
  position.push_back(V3(1, 1, 1));
  position.push_back(V3(-1, 1, 1));

  triangle.push_back(V3i(0, 1, 2));
  triangle.push_back(V3i(0, 2, 3));
  triangle.push_back(V3i(0, 3, 4));
  triangle.push_back(V3i(0, 4, 1));
  triangle.push_back(V3i(1, 2, 3));
  triangle.push_back(V3i(3, 4, 1));

  return convert_mesh_to_vtk(position, triangle);
}

}  // namespace visualization
}  // namespace mathx
