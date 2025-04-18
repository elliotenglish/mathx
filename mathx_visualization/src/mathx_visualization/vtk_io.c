#include "vtk_io.h"

#include <vtkPolyDataWriter.h>
#include <vtkUnstructuredGridWriter.h>

namespace mathx {
namespace visualization {

void WriteUnstructuredGridToFile(
    const vtkSmartPointer<vtkUnstructuredGrid>& grid, const std::string& path) {
  // vtkSmartPointer<vtkPolyData> poly_data=vtkPolyData::New();
  // poly_data->SetPoints(points);
  // poly_data->PrintSelf(std::cout,vtkIndent(2));

  // vtkSmartPointer<vtkPolyDataWriter> writer=vtkPolyDataWriter::New();
  // vtkSmartPointer<vtkXMLPolyDataWriter>
  // writer=vtkXMLPolyDataWriter::New(); writer->DebugOn();
  // writer->SetFileName("data.vtp");
  // writer->SetInputData(poly_data);
  // writer->Write();
  // writer->PrintSelf(std::cout,vtkIndent(2));

  vtkSmartPointer<vtkUnstructuredGridWriter> writer =
      vtkUnstructuredGridWriter::New();
  writer->SetInputData(grid);
  writer->SetFileName(path.c_str());
  writer->Write();
}

void WritePolyDataToFile(const vtkSmartPointer<vtkPolyData>& poly_data,
                         const std::string& path) {
  // vtkSmartPointer<vtkPolyData>
  // poly_data=RenderSurface(*geometry,V3(0,0,1),V3(1,0,0)*scale,V3(0,1,0)*scale,40,40);
  vtkSmartPointer<vtkPolyDataWriter> writer = vtkPolyDataWriter::New();
  writer->SetInputData(poly_data);
  writer->SetFileName(path.c_str());
  writer->Write();
}

}  // namespace geometry
}  // namespace mathx
