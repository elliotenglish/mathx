#include "principal_component_analysis.h"

#include <mathx_core/log.h>

#include <Eigen/Dense>

namespace mathx {
namespace numerics {

template <typename T>
void PrincipalComponentAnalysis(const T* points, int D, int num_points, T* mean,
                                T* eigenvectors, T* eigenvalues) {
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTd;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorTd;

  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
      X((T*)points, num_points, D);

  // log_var(X);
  VectorTd mean_ = (X.colwise().sum().transpose())/num_points;
  // log_var(mean_);
  MatrixTd X_centered = X.rowwise() - mean_.transpose();

  MatrixTd covariance = X_centered.transpose() * X_centered;

  Eigen::SelfAdjointEigenSolver<MatrixTd> solver(covariance);

  memcpy(mean, mean_.data(), sizeof(T) * D);
  memcpy(eigenvectors, solver.eigenvectors().data(), sizeof(T) * D * D);
  memcpy(eigenvalues, solver.eigenvalues().data(), sizeof(T) * D);
}

#define INSTANTIATION_HELPER(T)                                                \
  template void PrincipalComponentAnalysis(const T* xs, int D, int num_points, \
                                           T* mean, T* eigenvectors,           \
                                           T* eigenvalues);

INSTANTIATION_HELPER(float);
INSTANTIATION_HELPER(double);

}  // namespace numerics
}  // namespace code
