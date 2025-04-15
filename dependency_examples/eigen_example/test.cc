/**
 * This code examples the Eigen Transform class. It does not use a compact representation (3 DOF as in Rodrigues vectors adn 4 DOF as in Quaternions).
 */

#include <Eigen/Dense>
#include <iostream>

int main(int argc, char** argv) {
  Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
  std::cout << "transform" << std::endl
            << "matrix()=" << transform.matrix() << std::endl
            << "linear()=" << transform.linear() << std::endl
            << "translation()=" << transform.translation() << std::endl;
}
