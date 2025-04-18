#pragma once

#include <vtkInteractorStyle.h>
#include <vtkNamedColors.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

namespace mathx {
namespace visualization {

class VTKEvent {
 public:
  int keymathx;
};

class VTKDisplayBase;

/**
 * Follows:
 * https://examples.vtk.org/site/Cxx/Interaction/KeypressEvents/
 */
template <typename BaseInteractor>
class VTKInteractorStyle : public BaseInteractor {
 public:
  VTKDisplayBase* display;

  VTKInteractorStyle(VTKDisplayBase* display_) : display(display_) {}

  virtual void OnKeyPress();
};

class VTKDisplayBase {
 public:
  vtkSmartPointer<vtkRenderWindow> window;
  vtkSmartPointer<vtkRenderer> renderer;
  vtkSmartPointer<vtkNamedColors> colors;
  vtkSmartPointer<vtkRenderWindowInteractor> interactor;

  std::vector<vtkSmartPointer<vtkProp> > props;

  VTKDisplayBase(const int width, const int height,
                 vtkInteractorStyle* interactor_style);
  virtual ~VTKDisplayBase();

  virtual void Clear();
  void AddProp(const vtkSmartPointer<vtkProp>& prop);

  std::vector<VTKEvent> events;

  std::vector<VTKEvent> EventLoopOnce();
};

template <typename BaseInteractor>
void VTKInteractorStyle<BaseInteractor>::OnKeyPress() {
  // Get the keypress.
  vtkRenderWindowInteractor* rwi = this->Interactor;

  // std::string key = rwi->GetKeySym();
  // // Output the key that was pressed.
  // std::cout << "Pressed " << key << std::endl;

  display->events.push_back(VTKEvent{rwi->GetKeyCode()});

  // Forward events.
  BaseInteractor::OnKeyPress();
}

}  // namespace visualization
}  // namespace mathx
