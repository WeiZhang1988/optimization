#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "line_searcher.hpp"
#include "optimizer.hpp"
using namespace opt;

double obj_fun(Eigen::VectorXd _var) {
  return 0.5 * _var.norm();
}
Eigen::VectorXd jac_fun(Eigen::VectorXd _var) {
  return _var;
}
Eigen::MatrixXd hes_fun(Eigen::VectorXd _var) {
  return _var * _var.transpose();
}

int main(int argc, char **argv) {
  Eigen::VectorXd var = Eigen::VectorXd::Random(10);
  std::cout<<var<<std::endl;
  std::shared_ptr<Exact_Line_Searcher> line_searcher = std::make_shared<Exact_Line_Searcher>(var, jac_fun(var), obj_fun, jac_fun, hes_fun);
  Gradient_Descent_Solver solver(var, 0.5, 1e-15, 1000, obj_fun, jac_fun, hes_fun, line_searcher);
  var = solver.solve();
  std::cout<<"-----------------------"<<std::endl;
  std::cout<<var<<std::endl;
  return 0;
}
