#include <gtest/gtest.h>
#include <mathx_core/log.h>
#include <mathx_numerics/principal_component_analysis.h>

#include <Eigen/Dense>

using namespace mathx;
using namespace numerics;

TEST(mathx_numerics, PrincipalComponentAnalysis) {
  typedef float T;
  typedef Eigen::Vector<T, 3> V3;

  T s1 = 0.1;

  std::vector<Eigen::Vector3f> xs;
  V3 m(.12, -.321, 1.1);
  V3 a0 = V3(0, 1, 1).normalized();
  V3 a1 = V3(.5, 1, -1).normalized();
  for (int i = 0; i < 10; i++) {
    xs.push_back(m + a0 + s1 * a1);
    xs.push_back(m + a0 - s1 * a1);
    xs.push_back(m - a0 - s1 * a1);
    xs.push_back(m - a0 + s1 * a1);
  }

  V3 eigenvalues;
  Eigen::Matrix<T, 3, 3, Eigen::RowMajor> eigenvectors;
  V3 mean;

  PrincipalComponentAnalysis((T *)xs.data(), 3, xs.size(), mean.data(),
                             eigenvectors.data(), eigenvalues.data());

  log_var(xs);
  log_var(mean);
  log_var(eigenvalues);
  log_var(eigenvectors);

  T tolerance = 1e-5;

  EXPECT_LT(eigenvalues(0), tolerance);
  EXPECT_LT(eigenvalues(0), eigenvalues(1));
  EXPECT_LT(eigenvalues(1), eigenvalues(2));

  log_var(eigenvectors.row(2));
  log_var(eigenvectors.row(1));

  EXPECT_GT(std::fabs(eigenvectors.row(2) * a0), 1 - tolerance);
  EXPECT_GT(std::fabs(eigenvectors.row(1) * a1), 1 - tolerance);
}
