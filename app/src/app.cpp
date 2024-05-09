#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "line_searcher.hpp"
#include "solver.hpp"
#include "optimizer.hpp"
using namespace opt;

int main(int argc, char **argv) {
  int max_iter_num = 10000;
  double epsilon = 1e-10;
  int var_dim = 2;
  Eigen::VectorXd var = Eigen::VectorXd::Random(var_dim) * 10;
  Eigen::VectorXd target = Eigen::VectorXd::Random(var_dim);
  Eigen::MatrixXd weight_mat = Eigen::MatrixXd::Identity(var_dim,var_dim);
  Eigen::VectorXd coe_vec    = Eigen::VectorXd::Zero(var_dim);
  std::shared_ptr<QP_wo_Constraint> sptr_problem = std::make_shared<QP_wo_Constraint>(var,target,weight_mat,coe_vec);
  std::shared_ptr<Exact_Line_Searcher> sptr_line_searcher = std::make_shared<Exact_Line_Searcher>(var, var);
  /*----------------------*/
  std::shared_ptr<Gradient_Descent_Solver> sptr_gd_solver = std::make_shared<Gradient_Descent_Solver>(var,epsilon,sptr_line_searcher);
  std::shared_ptr<Optimizer> sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_gd_solver,epsilon,max_iter_num);
  Eigen::VectorXd res = sptr_optimizer->optimize();
  std::cout<<"----------------------- gradient descent *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  std::cout<<"resT "<<res.transpose()<<std::endl;
  /*----------------------*/
  std::shared_ptr<Newton_Solver> sptr_nt_solver = std::make_shared<Newton_Solver>(var,epsilon,sptr_line_searcher);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_nt_solver,epsilon,max_iter_num);
  res = sptr_optimizer->optimize();
  std::cout<<"----------------------- Newton *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  std::cout<<"resT "<<res.transpose()<<std::endl;
  /*----------------------*/
  Eigen::MatrixXd hes = Eigen::MatrixXd::Identity(var_dim,var_dim);
  std::shared_ptr<DFP_Solver> sptr_dfp_solver = std::make_shared<DFP_Solver>(var,hes,epsilon,sptr_line_searcher);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_dfp_solver,epsilon,max_iter_num);
  res = sptr_optimizer->optimize();
  std::cout<<"----------------------- DFP *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  std::cout<<"resT "<<res.transpose()<<std::endl;
  /*----------------------*/
  hes = Eigen::MatrixXd::Identity(var_dim,var_dim);
  std::shared_ptr<BFGS_Solver> sptr_bfgs_solver = std::make_shared<BFGS_Solver>(var,hes,epsilon,sptr_line_searcher);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_bfgs_solver,epsilon,max_iter_num);
  res = sptr_optimizer->optimize();
  std::cout<<"----------------------- BFGS *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  std::cout<<"resT "<<res.transpose()<<std::endl;
  /*----------------------*/
  std::shared_ptr<LBFGS_Solver> sptr_lbfgs_solver = std::make_shared<LBFGS_Solver>(var,epsilon,10,sptr_line_searcher);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_lbfgs_solver,epsilon,max_iter_num);
  res = sptr_optimizer->optimize();
  std::cout<<"----------------------- LBFGS *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  std::cout<<"resT "<<res.transpose()<<std::endl;
  /*----------------------*/
  std::shared_ptr<Conjugate_Gradient_Solver> sptr_cg_solver = std::make_shared<Conjugate_Gradient_Solver>(var,epsilon,sptr_line_searcher);
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem,sptr_cg_solver,epsilon,max_iter_num);
  res = sptr_optimizer->optimize();
  std::cout<<"----------------------- CG *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  std::cout<<"resT "<<res.transpose()<<std::endl;
  /*----------------------*/
  std::shared_ptr<QP_w_Constraint> sptr_problem_cons = std::make_shared<QP_w_Constraint>(var,target,weight_mat,coe_vec);
  std::shared_ptr<Augmented_Lagrangian_Solver> sptr_al_solver = std::make_shared<Augmented_Lagrangian_Solver>(var,epsilon,sptr_line_searcher,Eigen::VectorXd::Ones(1),Eigen::VectorXd::Zero(1));
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem_cons,sptr_al_solver,epsilon,max_iter_num);
  res = sptr_optimizer->optimize();
  std::cout<<"----------------------- AL *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  std::cout<<"resT "<<res.transpose()<<std::endl;
  std::cout<<"eq_cons "<<sptr_problem_cons->eq_cons(res).transpose()<<std::endl;
  std::cout<<"ieq_cons "<<sptr_problem_cons->ieq_cons(res).transpose()<<std::endl;
  /*----------------------*/
  std::shared_ptr<Primal_Dual_Solver> sptr_pd_solver = std::make_shared<Primal_Dual_Solver>(var,epsilon,sptr_line_searcher,Eigen::VectorXd::Ones(1),Eigen::VectorXd::Ones(1));
  sptr_optimizer = std::make_shared<Optimizer>(sptr_problem_cons,sptr_pd_solver,epsilon,max_iter_num);
  res = sptr_optimizer->optimize();
  std::cout<<"----------------------- PD *- iter -* "<< sptr_optimizer->get_iter_num()<<std::endl;
  std::cout<<"varT "<<var.transpose()<<std::endl;
  std::cout<<"tarT "<<target.transpose()<<std::endl;
  std::cout<<"resT "<<res.transpose()<<std::endl;
  std::cout<<"eq_cons "<<sptr_problem_cons->eq_cons(res).transpose()<<std::endl;
  std::cout<<"ieq_cons "<<sptr_problem_cons->ieq_cons(res).transpose()<<std::endl;
  return 0;
}
