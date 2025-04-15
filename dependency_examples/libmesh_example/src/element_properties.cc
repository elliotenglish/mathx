// C++ include files that we need
#include <mathx_core/assert.h>
#include <mathx_core/log.h>
#include <math.h>

#include <algorithm>
#include <iostream>

// Basic include files needed for the mesh functionality.
#include "libmesh/equation_systems.h"
#include "libmesh/libmesh.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/vtk_io.h"

// Define the Finite Element object.
#include "libmesh/fe.h"

// Define Gauss quadrature rules.
#include "libmesh/quadrature_gauss.h"

// Define useful datatypes for finite element
// matrix and vector components.
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"
#include "libmesh/enum_solver_package.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/sparse_matrix.h"

// Define the DofMap, which handles degree of freedom
// indexing.
#include "libmesh/dof_map.h"

using namespace mathx;
using namespace libMesh;

void analyze_element(const FEBase& fe, const Elem& elem, const QBase& qrule) {
  const std::vector<Real>& JxW = fe.get_JxW();
  const std::vector<Point>& q_point = fe.get_xyz();
  const std::vector<std::vector<Real>>& phi = fe.get_phi();
  const std::vector<std::vector<RealGradient>>& dphi = fe.get_dphi();
  const std::vector<std::vector<Real>>& dphidx = fe.get_dphidx();
  const std::vector<std::vector<Real>>& dphidy = fe.get_dphidy();
  const std::vector<std::vector<Real>>& dphidz = fe.get_dphidz();

  log_var(q_point);
  log_var(phi);
  log_var(dphi);
  log_var(dphidx);
  log_var(dphidy);
  log_var(dphidz);

  int num_qp = q_point.size();
  int num_dof = phi.size();
  for (int i = 0; i < num_qp; i++)
    for (int j = 0; j < num_dof; j++) {
      // Assert that dphi is in physical coordinates
      MathXAssert(dphi[j][i](0) == dphidx[j][i]);
      MathXAssert(dphi[j][i](1) == dphidy[j][i]);
      MathXAssert(dphi[j][i](2) == dphidz[j][i]);
    }
}

void assemble_function(EquationSystems& es,
                       const std::string& libmesh_dbg_var(system_name)) {
  log("system assembly");

  log("get mesh");
  // libmesh_assert_equal_to (system_name, "sys");
  const MeshBase& mesh = es.get_mesh();
  const unsigned int dim = mesh.mesh_dimension();

  log("get system");
  LinearImplicitSystem& system = es.get_system<LinearImplicitSystem>("sys");
  SparseMatrix<Number>& matrix = system.get_system_matrix();

  log("get dof_map");
  const DofMap& dof_map = system.get_dof_map();
  FEType fe_type = dof_map.variable_type(0);

  log("get elements and quadrature rules");
  // Interior elements
  std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
  QGauss qrule(dim, FIFTH);
  fe->attach_quadrature_rule(&qrule);

  // Boundary condition elements
  std::unique_ptr<FEBase> fe_face(FEBase::build(dim, fe_type));
  QGauss qface(dim - 1, FIFTH);
  fe_face->attach_quadrature_rule(&qface);

  // Get element properties. Note that these are all references and will contain
  // the latest values after the element reinitialization.

  const std::vector<Real>& JxW = fe->get_JxW();
  const std::vector<Point>& q_point = fe->get_xyz();
  const std::vector<std::vector<Real>>& phi = fe->get_phi();
  const std::vector<std::vector<RealGradient>>& dphi = fe->get_dphi();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;
  std::vector<dof_id_type> dof_indices;
  for (const auto& elem : mesh.active_local_element_ptr_range()) {
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = cast_int<unsigned int>(dof_indices.size());
    fe->reinit(elem);

    log("element");
    analyze_element(*fe, *elem, qrule);

    // Local matrix/RHS that we will assemble into the full versions later
    Ke.resize(n_dofs, n_dofs);
    Fe.resize(n_dofs);

    for (auto side : elem->side_index_range())
      if (elem->neighbor_ptr(side) == nullptr) {
        log("face");
        fe_face->reinit(elem, side);
        analyze_element(*fe_face, *elem, qface);
      }

    matrix.add_matrix(Ke, dof_indices);
    system.rhs->add_vector(Fe, dof_indices);
  }
}

int main(int argc, char** argv) {
  // Initialize libraries, like in example 2.
  LibMeshInit init(argc, argv);
  ReferenceCounter::disable_print_counter_info();

  log(ProgramHeader("element_properties", argc, (const char**)argv));

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());

  log("mesh begin");
  MeshTools::Generation::build_square(mesh, 4, 4, -1., 1., -1., 1., TRI3);
  // mesh.print_info();
  log("mesh end");

  log("equation system begin");
  EquationSystems equation_systems(mesh);
  equation_systems.add_system<LinearImplicitSystem>("sys");
  int u_lagrange_id = equation_systems.get_system("sys").add_variable(
      "u_lagrange", FIRST, LAGRANGE);
  // int u_monomial_id = equation_systems.get_system("sys").add_variable(
  //     "u_monomial", FIRST, MONOMIAL);
  // int u_monomial_vec_id = equation_systems.get_system("sys").add_variable(
  //     "v_monomial_vec", FIRST, MONOMIAL_VEC);
  equation_systems.get_system("sys").attach_assemble_function(
      assemble_function);
  equation_systems.init();
  // equation_systems.print_info();
  log("equation system end");

  log("solve begin");
  equation_systems.get_system("sys").solve();
  log("solve end");

  // #if defined(LIBMESH_HAVE_VTK) && !defined(LIBMESH_ENABLE_PARMESH)
  //   VTKIO(mesh).write_equation_systems("out.pvtu", equation_systems);
  // #endif  // #ifdef LIBMESH_HAVE_VTK

  return 0;
}
