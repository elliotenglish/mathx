// The libMesh Finite Element Library.
// Copyright (C) 2002-2025 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



// <h1>Vector Finite Elements Example 1 - Solving an uncoupled Poisson Problem</h1>
// \author Paul Bauman
// \date 2012
//
// This is the first vector FE example program.  It builds on
// the introduction_ex3 example program by showing how to solve a simple
// uncoupled Poisson system using vector Lagrange elements.

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <math.h>

// Basic include files needed for the mesh functionality.
#include "libmesh/enum_solver_type.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/linear_solver.h"
#include "libmesh/equation_systems.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/gmv_io.h"

// Include files for options
#include "libmesh/enum_norm_type.h"
#include "libmesh/enum_solver_package.h"
#include "libmesh/getpot.h"
#include "libmesh/string_to_enum.h"

// Define the Finite Element object.
#include "libmesh/fe.h"
#include "libmesh/fe_interface.h"

// Define Gauss quadrature rules.
#include "libmesh/quadrature_gauss.h"

// Define useful datatypes for finite element
// matrix and vector components.
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"

// Define the DofMap, which handles degree of freedom
// indexing.
#include "libmesh/dof_map.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// Function prototype.  This is the function that will assemble
// the linear system for our Poisson problem.  Note that the
// function will take the  EquationSystems object and the
// name of the system we are assembling as input.  From the
//  EquationSystems object we have access to the  Mesh and
// other objects we might need.
void assemble_poisson(EquationSystems & es,
                      const std::string & system_name);

// Function prototype for the exact solution.
Real exact_solution (const int component,
                     const Real x,
                     const Real y,
                     const Real z = 0.);

int main (int argc, char ** argv)
{
  // Initialize libraries.
  LibMeshInit init (argc, argv);

  // This example requires a linear solver package.
  libmesh_example_requires(libMesh::default_solver_package() != INVALID_SOLVER_PACKAGE,
                           "--enable-petsc, --enable-trilinos, or --enable-eigen");

  // Brief message to the user regarding the program name
  // and command line arguments.
  libMesh::out << "Running " << argv[0];

  for (int i=1; i<argc; i++)
    libMesh::out << " " << argv[i];

  libMesh::out << std::endl << std::endl;

  // Skip this 2D example if libMesh was compiled as 1D-only.
  libmesh_example_requires(2 <= LIBMESH_DIM, "2D support");

  // Get the mesh size from the command line.
  const int nx = libMesh::command_line_next("-nx", 15),
            ny = libMesh::command_line_next("-ny", 15);

  // Create a mesh, with dimension to be overridden later, on the
  // default MPI communicator.
  Mesh mesh(init.comm());

  // Use the MeshTools::Generation mesh generator to create a uniform
  // 2D grid on the square [-1,1]^2.  We instruct the mesh generator
  // to build a mesh of 15x15 QUAD9 elements.
  MeshTools::Generation::build_square (mesh,
                                       nx, ny,
                                       -1., 1.,
                                       -1., 1.,
                                       TRI3);

  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);

  // Declare the Poisson system and its variables.
  // The Poisson system is another example of a steady system.
  LinearImplicitSystem & poisson = equation_systems.add_system<LinearImplicitSystem> ("Poisson");

  // Read FE order from command line
  std::string order_str = "FIRST";
  order_str = libMesh::command_line_next("-o", order_str);
  order_str = libMesh::command_line_next("-Order", order_str);
  const Order order = Utility::string_to_enum<Order>(order_str);

  // Read FE Family from command line
  std::string family_str = "LAGRANGE_VEC";
  family_str = libMesh::command_line_next("-f", family_str);
  family_str = libMesh::command_line_next("-FEFamily", family_str);
  const FEFamily family = Utility::string_to_enum<FEFamily>(family_str);

  libmesh_error_msg_if(FEInterface::field_type(family) != TYPE_VECTOR,
                       "FE family " + family_str + " isn't vector-valued");

  // Adds the variable "u" to "Poisson".  "u" will be approximated
  // using the requested order of approximation and vector element
  // type. Since the mesh is 2-D, "u" will have two components.
  poisson.add_variable("u", order, family);

  // Give the system a pointer to the matrix assembly
  // function.  This will be called when needed by the
  // library.
  poisson.attach_assemble_function (assemble_poisson);

  // Initialize the data structures for the equation system.
  equation_systems.init();

  // Prints information about the system to the screen.
  equation_systems.print_info();

  // If we're using Eigen, the default BiCGStab solver does not seem
  // to converge robustly for this system.  Let's try some other
  // settings for them.
  if (libMesh::default_solver_package() == EIGEN_SOLVERS)
    poisson.get_linear_solver()->set_solver_type(GMRES);

  // Solve the system "Poisson".  Note that calling this
  // member will assemble the linear system and invoke
  // the default numerical solver.  With PETSc the solver can be
  // controlled from the command line.  For example,
  // you can invoke conjugate gradient with:
  //
  // ./vector_fe_ex1 -ksp_type cg
  //
  // You can also get a nice X-window that monitors the solver
  // convergence with:
  //
  // ./vector_fe_ex1 -ksp_xmonitor
  //
  // if you linked against the appropriate X libraries when you
  // built PETSc.
  poisson.solve();

  // const Real l2_norm =
  //   poisson.calculate_norm(*poisson.solution, 0, L2);

  // libMesh::out << "L2 norm of solution = " << std::setprecision(17) <<
  //   l2_norm << std::endl;

  // libmesh_error_msg_if (libmesh_isnan(l2_norm),
  //                       "Failed to calculate solution");

  // const Real error_in_norm = std::abs(l2_norm - sqrt(Real(2)));

  // libMesh::out << "error in L2 norm = " << std::setprecision(17) <<
  //   error_in_norm << std::endl;

  // The error in the norm converges faster than the norm of the
  // error, at least until it gets low enough that floating-point
  // roundoff (and the penalty method) kill us.
  // const int n = std::min(nx, ny);
  // const int p = static_cast<int>(order);
  // const Real expected_error_bound = 2*std::pow(n, -p*2);
  // libMesh::out << "error bound = " << std::setprecision(17) <<
  //   expected_error_bound << std::endl;
  // libMesh::out << "error ratio = " << std::setprecision(17) <<
  //   error_in_norm / expected_error_bound << std::endl;
  // libmesh_error_msg_if (error_in_norm > expected_error_bound,
  //                       "Error exceeds expected bound of " <<
  //                       expected_error_bound);

#ifdef LIBMESH_HAVE_EXODUS_API
  ExodusII_IO(mesh).write_equation_systems("out.e", equation_systems);
#endif

#ifdef LIBMESH_HAVE_GMV
  GMVIO(mesh).write_equation_systems("out.gmv", equation_systems);
#endif

  // All done.
  return 0;
}



// We now define the matrix assembly function for the
// Poisson system.  We need to first compute element
// matrices and right-hand sides, and then take into
// account the boundary conditions, which will be handled
// via a penalty method.
void assemble_poisson(EquationSystems & es,
                      const std::string & libmesh_dbg_var(system_name))
{

  // It is a good idea to make sure we are assembling
  // the proper system.
  libmesh_assert_equal_to (system_name, "Poisson");

  // Get a constant reference to the mesh object.
  const MeshBase & mesh = es.get_mesh();

  // The dimension that we are running
  const unsigned int dim = mesh.mesh_dimension();

  // Get a reference to the LinearImplicitSystem we are solving
  LinearImplicitSystem & system = es.get_system<LinearImplicitSystem> ("Poisson");

  // A reference to the  DofMap object for this system.  The  DofMap
  // object handles the index translation from node and element numbers
  // to degree of freedom numbers.  We will talk more about the  DofMap
  // in future examples.
  const DofMap & dof_map = system.get_dof_map();

  // Get a constant reference to the Finite Element type
  // for the first (and only) variable in the system.
  FEType fe_type = dof_map.variable_type(0);

  // Build a Finite Element object of the specified type.
  // Note that FEVectorBase is a typedef for the templated FE
  // class.
  std::unique_ptr<FEVectorBase> fe (FEVectorBase::build(dim, fe_type));

  // A 2*p+1 order Gauss quadrature rule for numerical integration.
  QGauss qrule (dim, fe_type.default_quadrature_order());

  // Tell the finite element object to use our quadrature rule.
  fe->attach_quadrature_rule (&qrule);

  // Declare a special finite element object for
  // boundary integration.
  std::unique_ptr<FEVectorBase> fe_face (FEVectorBase::build(dim, fe_type));

  // Boundary integration requires one quadrature rule,
  // with dimensionality one less than the dimensionality
  // of the element.
  QGauss qface(dim-1, fe_type.default_quadrature_order());

  // Tell the finite element object to use our
  // quadrature rule.
  fe_face->attach_quadrature_rule (&qface);

  // Here we define some references to cell-specific data that
  // will be used to assemble the linear system.
  //
  // The element Jacobian * quadrature weight at each integration point.
  const std::vector<Real> & JxW = fe->get_JxW();

  // The physical XY locations of the quadrature points on the element.
  // These might be useful for evaluating spatially varying material
  // properties at the quadrature points.
  const std::vector<Point> & q_point = fe->get_xyz();

  // The element shape functions evaluated at the quadrature points.
  // Notice the shape functions are a vector rather than a scalar.
  const std::vector<std::vector<RealGradient>> & phi = fe->get_phi();

  // The element shape function gradients evaluated at the quadrature
  // points. Notice that the shape function gradients are a tensor.
  const std::vector<std::vector<RealTensor>> & dphi = fe->get_dphi();

  // Define data structures to contain the element matrix
  // and right-hand-side vector contribution.  Following
  // basic finite element terminology we will denote these
  // "Ke" and "Fe".  These datatypes are templated on
  //  Number, which allows the same code to work for real
  // or complex numbers.
  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  // This vector will hold the degree of freedom indices for
  // the element.  These define where in the global system
  // the element degrees of freedom get mapped.
  std::vector<dof_id_type> dof_indices;

  // The global system matrix
  SparseMatrix<Number> & matrix = system.get_system_matrix();

  // Now we will loop over all the elements in the mesh.
  // We will compute the element matrix and right-hand-side
  // contribution.
  //
  // Element iterators are a nice way to iterate through all the
  // elements, or all the elements that have some property.  The
  // iterator el will iterate from the first to the last element on
  // the local processor.  The iterator end_el tells us when to stop.
  // It is smart to make this one const so that we don't accidentally
  // mess it up!  In case users later modify this program to include
  // refinement, we will be safe and will only consider the active
  // elements; hence we use a variant of the active_elem_iterator.
  for (const auto & elem : mesh.active_local_element_ptr_range())
    {
      // Get the degree of freedom indices for the
      // current element.  These define where in the global
      // matrix and right-hand-side this element will
      // contribute to.
      dof_map.dof_indices (elem, dof_indices);

      // Compute the element-specific data for the current
      // element.  This involves computing the location of the
      // quadrature points (q_point) and the shape functions
      // (phi, dphi) for the current element.
      fe->reinit (elem);

      // Zero the element matrix and right-hand side before
      // summing them.  We use the resize member here because
      // the number of degrees of freedom might have changed from
      // the last element.  Note that this will be the case if the
      // element type is different (i.e. the last element was a
      // triangle, now we are on a quadrilateral).

      // The  DenseMatrix::resize() and the  DenseVector::resize()
      // members will automatically zero out the matrix  and vector.
      Ke.resize (dof_indices.size(),
                 dof_indices.size());

      Fe.resize (dof_indices.size());

      // We'll use an element-size-dependent h below, so the FDM error
      // doesn't easily dominate FEM error.
      const Real eps = 1.e-3 * elem->hmin();

      // Now loop over the quadrature points.  This handles
      // the numeric integration.
      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
        {
          // Now we will build the element matrix.  This involves
          // a double loop to integrate the test functions (i) against
          // the trial functions (j).
          for (std::size_t i=0; i<phi.size(); i++)
            for (std::size_t j=0; j<phi.size(); j++)
              Ke(i,j) += JxW[qp] * dphi[i][qp].contract(dphi[j][qp]);

          // This is the end of the matrix summation loop
          // Now we build the element right-hand-side contribution.
          // This involves a single loop in which we integrate the
          // "forcing function" in the PDE against the test functions.
          {
            const Real x = q_point[qp](0);
            const Real y = q_point[qp](1);

            // "f" is the forcing function for the Poisson equation.
            // In this case we set f to be a finite difference
            // Laplacian approximation to the (known) exact solution.
            //
            // We will use the second-order accurate FD Laplacian
            // approximation, which in 2D is
            //
            // u_xx + u_yy = (u(i,j-1) + u(i,j+1) +
            //                u(i-1,j) + u(i+1,j) +
            //                -4*u(i,j))/h^2

            // Since the value of the forcing function depends only
            // on the location of the quadrature point (q_point[qp])
            // we will compute it here, outside of the i-loop
            const Real fx = -(exact_solution(0, x, y-eps) +
                              exact_solution(0, x, y+eps) +
                              exact_solution(0, x-eps, y) +
                              exact_solution(0, x+eps, y) -
                              4.*exact_solution(0, x, y))/eps/eps;

            const Real fy = -(exact_solution(1, x, y-eps) +
                              exact_solution(1, x, y+eps) +
                              exact_solution(1, x-eps, y) +
                              exact_solution(1, x+eps, y) -
                              4.*exact_solution(1, x, y))/eps/eps;

            const RealGradient f(fx, fy);

            for (std::size_t i=0; i<phi.size(); i++)
              Fe(i) += JxW[qp]*f*phi[i][qp];
          }
        }

      // We have now reached the end of the RHS summation,
      // and the end of quadrature point loop, so
      // the interior element integration has
      // been completed.  However, we have not yet addressed
      // boundary conditions.  For this example we will only
      // consider simple Dirichlet boundary conditions.
      //
      // There are several ways Dirichlet boundary conditions
      // can be imposed.  A simple approach, which works for
      // interpolary bases like the standard Lagrange polynomials,
      // is to assign function values to the
      // degrees of freedom living on the domain boundary. This
      // works well for interpolary bases, but is more difficult
      // when non-interpolary (e.g Legendre or Hierarchic) bases
      // are used.
      //
      // Dirichlet boundary conditions can also be imposed with a
      // "penalty" method.  In this case essentially the L2 projection
      // of the boundary values are added to the matrix. The
      // projection is multiplied by some large factor so that, in
      // floating point arithmetic, the existing (smaller) entries
      // in the matrix and right-hand-side are effectively ignored.
      //
      // This amounts to adding a term of the form (in latex notation)
      //
      // \frac{1}{\epsilon} \int_{\delta \Omega} \phi_i \phi_j = \frac{1}{\epsilon} \int_{\delta \Omega} u \phi_i
      //
      // where
      //
      // \frac{1}{\epsilon} is the penalty parameter, defined such that \epsilon << 1
      {
        // The following loop is over the sides of the element.
        // If the element has no neighbor on a side then that
        // side MUST live on a boundary of the domain.
        for (auto side : elem->side_index_range())
          if (elem->neighbor_ptr(side) == nullptr)
            {
              // The value of the shape functions at the quadrature
              // points.
              const std::vector<std::vector<RealGradient>> & phi_face = fe_face->get_phi();

              // The Jacobian * Quadrature Weight at the quadrature
              // points on the face.
              const std::vector<Real> & JxW_face = fe_face->get_JxW();

              // The XYZ locations (in physical space) of the
              // quadrature points on the face.  This is where
              // we will interpolate the boundary value function.
              const std::vector<Point> & qface_point = fe_face->get_xyz();

              // Compute the shape function values on the element
              // face.
              fe_face->reinit(elem, side);

              // Loop over the face quadrature points for integration.
              for (unsigned int qp=0; qp<qface.n_points(); qp++)
                {
                  // The location on the boundary of the current
                  // face quadrature point.
                  const Real xf = qface_point[qp](0);
                  const Real yf = qface_point[qp](1);

                  // The penalty value.  \frac{1}{\epsilon}
                  // in the discussion above.
                  const Real penalty = 1.e10;

                  // The boundary values.
                  const RealGradient f(exact_solution(0, xf, yf),
                                       exact_solution(1, xf, yf));

                  // Matrix contribution of the L2 projection.
                  for (std::size_t i=0; i<phi_face.size(); i++)
                    for (std::size_t j=0; j<phi_face.size(); j++)
                      Ke(i,j) += JxW_face[qp]*penalty*phi_face[i][qp]*phi_face[j][qp];

                  // Right-hand-side contribution of the L2
                  // projection.
                  for (std::size_t i=0; i<phi_face.size(); i++)
                    Fe(i) += JxW_face[qp]*penalty*f*phi_face[i][qp];
                }
            }
      }

      // We have now finished the quadrature point loop,
      // and have therefore applied all the boundary conditions.

      // If this assembly program were to be used on an adaptive mesh,
      // we would have to apply any hanging node constraint equations
      //dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

      // The element matrix and right-hand-side are now built
      // for this element.  Add them to the global matrix and
      // right-hand-side vector.  The  SparseMatrix::add_matrix()
      // and  NumericVector::add_vector() members do this for us.
      matrix.add_matrix         (Ke, dof_indices);
      system.rhs->add_vector    (Fe, dof_indices);

      
    }

  // All done!
}
