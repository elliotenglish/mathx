/**
 * This code tests the usage of autodiff with Eigen types.
 */

#include <Eigen/Dense>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <iostream>

int main(int argc, char** argv) {
  typedef Eigen::Quaternion<autodiff::var> QAD;

  QAD q0(1, 1, 0, 0);
  QAD q1(1, 0, 1, 0);

  QAD qmult = q0 * q1;

  std::array<double, 1> dd00 = autodiff::derivatives(qmult.x(), wrt(q0.x()));
  Eigen::MatrixXd jacobian0 = autodiff::gradient(qmult.x(), q0.coeffs());

  std::cout << "quaternion" << std::endl
            << "q0=" << q0 << std::endl
            << "q1=" << q1 << std::endl
            << "qmult=" << qmult << std::endl
            << "dd00" << dd00[0] << std::endl
            << "jacobian0" << jacobian0.transpose() << std::endl;

  return 0;
}
