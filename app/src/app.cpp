#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "line_searcher.hpp"
#include "solver.hpp"
#include "optimizer.hpp"
using namespace opt;

int main(int argc, char **argv) {
  int max_iter_num = 100;
  Eigen::VectorXd var = Eigen::VectorXd::Random(10) * 100;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  Eigen::VectorXd target = Eigen::VectorXd::Random(10);
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  Eigen::MatrixXd weight_mat = Eigen::MatrixXd::Identity(10,10);
  Eigen::VectorXd coe_vec    = Eigen::VectorXd::Zero(10);
  std::shared_ptr<QP_wo_Constraint> sptr_problem = std::make_shared<QP_wo_Constraint>(var,target,weight_mat,coe_vec);
  std::shared_ptr<InExact_Line_Searcher> sptr_line_searcher = std::make_shared<InExact_Line_Searcher>(var, var);
  /*----------------------*/
  std::shared_ptr<Gradient_Descent_Solver> sptr_gd_solver = std::make_shared<Gradient_Descent_Solver>(var);
  std::shared_ptr<Optimizer> sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_line_searcher,sptr_gd_solver,1e-10,max_iter_num);
  var = sptr_optimizer->optimize();
  std::cout<<"----------------------- gradient descent *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  /*----------------------*/
  std::shared_ptr<Newton_Solver> sptr_nt_solver = std::make_shared<Newton_Solver>(var);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_line_searcher,sptr_nt_solver,1e-10,max_iter_num);
  var = sptr_optimizer->optimize();
  std::cout<<"----------------------- Newton *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  /*----------------------*/
  Eigen::MatrixXd hes = Eigen::MatrixXd::Identity(10,10);
  std::shared_ptr<DFP_Solver> sptr_dfp_solver = std::make_shared<DFP_Solver>(var,hes);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_line_searcher,sptr_dfp_solver,1e-10,max_iter_num);
  var = sptr_optimizer->optimize();
  std::cout<<"----------------------- DFP *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  /*----------------------*/
  hes = Eigen::MatrixXd::Identity(10,10);
  std::shared_ptr<BFGS_Solver> sptr_bfgs_solver = std::make_shared<BFGS_Solver>(var,hes);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_line_searcher,sptr_bfgs_solver,1e-10,max_iter_num);
  var = sptr_optimizer->optimize();
  std::cout<<"----------------------- BFGS *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  /*----------------------*/
  std::shared_ptr<LBFGS_Solver> sptr_lbfgs_solver = std::make_shared<LBFGS_Solver>(var,10);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_line_searcher,sptr_lbfgs_solver,1e-10,max_iter_num);
  var = sptr_optimizer->optimize();
  std::cout<<"----------------------- LBFGS *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  /*----------------------*/
  std::shared_ptr<Conjugate_Gradient_Solver> sptr_cg_solver = std::make_shared<Conjugate_Gradient_Solver>(var);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_line_searcher,sptr_cg_solver,1e-10,max_iter_num);
  var = sptr_optimizer->optimize();
  std::cout<<"----------------------- CG *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  return 0;
}
