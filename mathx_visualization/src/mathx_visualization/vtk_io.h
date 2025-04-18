#pragma once

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

namespace mathx {
namespace visualization {

void WriteUnstructuredGridToFile(
    const vtkSmartPointer<vtkUnstructuredGrid>& grid, const std::string& path);
void WritePolyDataToFile(const vtkSmartPointer<vtkPolyData>& poly_data,
                         const std::string& path);

}  // namespace geometry
}  // namespace mathx
