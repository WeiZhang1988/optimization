#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Dense>
#include "type_define.hpp"
#include "problem.hpp"
#include "solver.hpp"

namespace opt{
class Optimizer{
  public:
  Optimizer(std::shared_ptr<Base_Problem> _sptr_problem, \
            std::shared_ptr<Base_Solver> _sptr_solver, 
            double _epsilon = 1e-10, \
            int _max_iter_num = 1000) : 
            sptr_problem_(_sptr_problem), 
            sptr_solver_(_sptr_solver),
            max_iter_num_(_max_iter_num) {
    var_ = _sptr_problem->get_var();
    var_pre_ = Eigen::VectorXd::Ones(var_.size()) * std::numeric_limits<double>::max() * 0.5;
    vars_.push_back(var_);
    obj_fun_      = std::bind(&Base_Problem::obj_fun,sptr_problem_,std::placeholders::_1);
    jac_fun_      = std::bind(&Base_Problem::jac_fun,sptr_problem_,std::placeholders::_1);
    hes_fun_      = std::bind(&Base_Problem::hes_fun,sptr_problem_,std::placeholders::_1);
    eq_cons_      = std::bind(&Base_Problem::eq_cons,sptr_problem_,std::placeholders::_1);    
    ieq_cons_     = std::bind(&Base_Problem::ieq_cons,sptr_problem_,std::placeholders::_1);  
    jac_eq_cons_  = std::bind(&Base_Problem::jac_eq_cons,sptr_problem_,std::placeholders::_1); 
    jac_ieq_cons_ = std::bind(&Base_Problem::jac_ieq_cons,sptr_problem_,std::placeholders::_1);
    hes_eq_cons_  = std::bind(&Base_Problem::hes_eq_cons,sptr_problem_,std::placeholders::_1);
    hes_ieq_cons_ = std::bind(&Base_Problem::hes_ieq_cons,sptr_problem_,std::placeholders::_1);
    _sptr_solver->set_obj_fun(obj_fun_);
    _sptr_solver->set_jac_fun(jac_fun_);
    _sptr_solver->set_hes_fun(hes_fun_);
    _sptr_solver->set_eq_cons(eq_cons_);
    _sptr_solver->set_ieq_cons(ieq_cons_);
    _sptr_solver->set_jac_eq_cons(jac_eq_cons_);
    _sptr_solver->set_jac_ieq_cons(jac_ieq_cons_);
    _sptr_solver->set_hes_eq_cons(hes_eq_cons_);
    _sptr_solver->set_hes_ieq_cons(hes_ieq_cons_);
  }
  Eigen::VectorXd optimize() {
    sptr_solver_->init_for_solve();
    iter_num_ = 0;
    while (!sptr_solver_->ending_condition() &&
           iter_num_ < max_iter_num_) {
      iter_num_++;
      var_pre_ = var_;
      sptr_solver_->set_var(var_);
      var_ = sptr_solver_->solve();
      vars_.push_back(var_);
    }
    return var_;
  }
  std::vector<Eigen::VectorXd> get_vars() {
    return vars_;
  }
  int get_iter_num() {
    return iter_num_;
  }
  protected:
  std::shared_ptr<Base_Problem> sptr_problem_;
  std::shared_ptr<Base_Solver> sptr_solver_;
  int max_iter_num_ = 1000;
  int iter_num_ = 0;
  double alpha_ = 1.0;
  std::vector<Eigen::VectorXd> vars_;
  Eigen::VectorXd var_, var_pre_;
  ObjFun obj_fun_;
  JacFun jac_fun_;
  HesFun hes_fun_;
  ConsFun eq_cons_;     
  ConsFun ieq_cons_;    
  JacCons jac_eq_cons_;
  JacCons jac_ieq_cons_;
  HesCons hes_eq_cons_;
  HesCons hes_ieq_cons_;
};
} //namespace opt
#endif //OPTIMIZER_HPP