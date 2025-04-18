#pragma once

#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>

#include <Eigen/Dense>
#include <vector>

namespace mathx {
namespace visualization {

vtkSmartPointer<vtkUnstructuredGrid> convert_particles_to_vtk(
    std::vector<Eigen::Vector3f>& position, std::vector<Eigen::Vector3f>& color,
    std::vector<float>& radius);

vtkSmartPointer<vtkPolyData> convert_mesh_to_vtk(
    std::vector<Eigen::Vector3f>& position,
    std::vector<Eigen::Vector3i>& triangle);

/**
 * Generates a unit sized pyramid with the peak at the origin and the base
 * offset along the positive z axis.
 */
vtkSmartPointer<vtkPolyData> generate_pyramid();

}  // namespace visualization
}  // namespace mathx
