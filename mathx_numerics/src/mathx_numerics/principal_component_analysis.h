#pragma once

namespace mathx {
namespace numerics {

/**
 * Computes the principle components.
 *
 * The input points and and eigenvectors are stored in row major order.
 */
template <typename T>
void PrincipalComponentAnalysis(const T *points, int D, int num_points, T *mean,
                                T *eigenvectors, T *eigenvalues);

} // namespace numerics
} // namespace mathx
