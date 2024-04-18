#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "line_searcher.hpp"
#include "solver.hpp"
#include "optimizer.hpp"
using namespace opt;

int main(int argc, char **argv) {
  Eigen::VectorXd var = Eigen::VectorXd::Random(10);
  std::cout<<"varT "<<var.transpose()<<std::endl;
  Eigen::VectorXd target = Eigen::VectorXd::Random(10);
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  Eigen::MatrixXd weight_mat = Eigen::MatrixXd::Identity(10,10);
  Eigen::VectorXd coe_vec    = Eigen::VectorXd::Zero(10);
  std::shared_ptr<QP_wo_Constraint> sptr_problem = std::make_shared<QP_wo_Constraint>(var,target,weight_mat,coe_vec);
  std::shared_ptr<Exact_Line_Searcher> sptr_line_searcher = std::make_shared<Exact_Line_Searcher>(var, var);
  std::shared_ptr<Gradient_Descent_Solver> sptr_solver = std::make_shared<Gradient_Descent_Solver>(var);
  std::shared_ptr<Optimizer> sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_line_searcher,sptr_solver,1e-10,1000);
  var = sptr_optimizer->optimize();
  std::cout<<"-----------------------"<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  return 0;
}
