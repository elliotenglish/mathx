#include "vtk_display_base.h"

#include <thread>

namespace mathx {
namespace visualization {

VTKDisplayBase::VTKDisplayBase(const int width, const int height,
                               vtkInteractorStyle* interactor_style) {
  colors = vtkNamedColors::New();

  renderer = vtkRenderer::New();
  renderer->SetBackground(colors->GetColor3d("white").GetData());
  renderer->SetViewport(0, 0, 1, 1);

  window = vtkRenderWindow::New();
  window->SetSize(width, height);
  window->AddRenderer(renderer);
  window->SetPosition(10, 10);

  interactor = vtkRenderWindowInteractor::New();
  interactor->SetInteractorStyle(interactor_style);
  interactor->SetRenderWindow(window);
  interactor->Initialize();

  Clear();
}

VTKDisplayBase::~VTKDisplayBase() {}

void VTKDisplayBase::Clear() {
  renderer->GetViewProps()->RemoveAllItems();
  props.clear();
}

void VTKDisplayBase::AddProp(const vtkSmartPointer<vtkProp>& prop) {
  props.push_back(prop);
  renderer->AddViewProp(prop);
}

std::vector<VTKEvent> VTKDisplayBase::EventLoopOnce() {
  window->Render();
  interactor->ProcessEvents();
  // camera->PrintSelf(std::cout, (vtkIndent)2);
  // double clip[2];
  // camera->GetClippingRange(clip);
  // log("clip min=%g max=%g", clip[0], clip[1]);
  std::vector<VTKEvent> events_=events;
  events.clear();
  return events_;
}

}  // namespace visualization
}  // namespace mathx
