#include "vtk_scene.h"

#include <vtkBillboardTextActor3D.h>
#include <vtkDataSetMapper.h>
#include <vtkFollower.h>
#include <vtkMatrix4x4.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkTextProperty.h>
#include <vtkVectorText.h>

namespace mathx {
namespace visualization {

void VTKScene::SetActorTransform(vtkActor* actor,
                                 const Eigen::Affine3f& transform) {
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> matrix =
      transform.matrix().cast<double>();
  vtkNew<vtkMatrix4x4> vtk_transform;
  vtk_transform->DeepCopy(matrix.data());
  actor->PokeMatrix(vtk_transform.Get());

  // std::cout << "transform" << std::endl << transform.matrix() << std::endl;
  // std::cout << "matrix " << std::endl << matrix.matrix() << std::endl;
  // vtk_transform->PrintSelf(std::cout,vtkIndent(2));
  // std::cout << "actor matrix" << std::endl;
  // actor->GetMatrix()->PrintSelf(std::cout,vtkIndent(2));
}

vtkSmartPointer<vtkProp> VTKScene::GenerateObject(
    const vtkSmartPointer<vtkDataSet>& data, const DrawMode draw_mode,
    const Eigen::Affine3f& transform) {
  vtkNew<vtkDataSetMapper> rendererMapper;
  rendererMapper->SetInputData(data);

  vtkNew<vtkActor> rendererActor;
  rendererActor->SetMapper(rendererMapper);
  SetActorTransform(rendererActor, transform);

  if (draw_mode == DrawVertices) {
    rendererActor->GetProperty()->SetRepresentationToPoints();
    // rendererActor->GetProperty()->SetVertexVisibility(true);
    rendererActor->GetProperty()->SetRenderPointsAsSpheres(false);
    rendererActor->GetProperty()->SetPointSize(10);
  } else if (draw_mode == DrawEdges) {
    rendererActor->GetProperty()->SetRepresentationToWireframe();
    // rendererActor->GetProperty()->SetEdgeVisibility(true);
    rendererActor->GetProperty()->SetRenderLinesAsTubes(false);
    rendererActor->GetProperty()->SetPointSize(10);
  } else  // DrawFaces
  {
    rendererActor->GetProperty()->SetRepresentationToSurface();
  }

  rendererActor->GetProperty()->SetDiffuseColor(0, 0, 1);

  return rendererActor;
}

vtkSmartPointer<vtkProp> VTKScene::GenerateText(const std::string& text,
                                                const Eigen::Vector3f& origin,
                                                const float scale) {
  bool use_follower = false;
  if (use_follower) {
    vtkNew<vtkVectorText> atext;
    atext->SetText(text.c_str());
    vtkNew<vtkPolyDataMapper> textMapper;
    textMapper->SetInputConnection(atext->GetOutputPort());

    vtkNew<vtkFollower> textActor;
    textActor->SetMapper(textMapper);
    textActor->SetScale(scale, scale, scale);
    textActor->AddPosition(origin[0], origin[1], origin[2]);
    textActor->GetProperty()->SetColor(0, 0, 0);

    // This makes the text face the camera
    // textActor->SetCamera(renderer->GetActiveCamera());

    return textActor;
  } else {
    vtkNew<vtkBillboardTextActor3D> textActor;
    textActor->SetInput(text.c_str());
    textActor->SetPosition(origin[0], origin[1], origin[2]);
    textActor->GetTextProperty()->SetFontSize(12 * scale);
    textActor->GetTextProperty()->SetColor(0, 0, 0);
    textActor->GetTextProperty()->SetJustificationToCentered();

    return textActor;
  }
}

}  // namespace visualization
}  // namespace mathx
