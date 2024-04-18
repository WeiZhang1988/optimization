#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Dense>
#include "type_define.hpp"
#include "problem.hpp"
#include "line_searcher.hpp"
#include "solver.hpp"

namespace opt{
class Optimizer{
  public:
  Optimizer(std::shared_ptr<Base_Problem> _sptr_problem, \
            std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
            std::shared_ptr<Base_Solver> _sptr_solver, 
            double _epsilon = 1e-10, \
            int _max_iter_num = 1000) : 
            sptr_problem_(_sptr_problem), 
            sptr_line_searcher_(_sptr_line_searcher),
            sptr_solver_(_sptr_solver),
            epsilon_(_epsilon),
            max_iter_num_(_max_iter_num) {
    var_ = _sptr_problem->get_var();
    var_pre_ = Eigen::VectorXd::Ones(var_.size()) * std::numeric_limits<double>::max() * 0.5;
    vars_.push_back(var_);
    obj_fun_ = std::bind(&Base_Problem::obj_fun,sptr_problem_,std::placeholders::_1);
    jac_fun_ = std::bind(&Base_Problem::jac_fun,sptr_problem_,std::placeholders::_1);
    hes_fun_ = std::bind(&Base_Problem::hes_fun,sptr_problem_,std::placeholders::_1);
    _sptr_line_searcher->set_obj_fun(obj_fun_);
    _sptr_line_searcher->set_jac_fun(jac_fun_);
    _sptr_line_searcher->set_hes_fun(hes_fun_);
    _sptr_solver->set_obj_fun(obj_fun_);
    _sptr_solver->set_jac_fun(jac_fun_);
    _sptr_solver->set_hes_fun(hes_fun_);
  }
  Eigen::VectorXd optimize() {
    int iter_num = 0;
    while ((var_ - var_pre_).lpNorm<1>() > epsilon_ && \
           (std::abs(obj_fun_(var_) - obj_fun_(var_pre_)) > epsilon_) && \
           jac_fun_(var_).lpNorm<1>() > epsilon_ && \
           iter_num < max_iter_num_) {
      iter_num++;
      var_pre_ = var_;
      sptr_solver_->set_var(var_);
      Eigen::VectorXd dir = sptr_solver_->solve();
      sptr_line_searcher_->set_var(var_);
      sptr_line_searcher_->set_dir(dir);
      alpha_ = sptr_line_searcher_->search();
      var_ += (alpha_ * dir);
      vars_.push_back(var_);
    }
    return var_;
  }
  std::vector<Eigen::VectorXd> get_vars() {
    return vars_;
  }
  protected:
  std::shared_ptr<Base_Problem> sptr_problem_;
  std::shared_ptr<Base_Line_Searcher> sptr_line_searcher_;
  std::shared_ptr<Base_Solver> sptr_solver_;
  double epsilon_ = 1e-10;
  int max_iter_num_ = 1000;
  double alpha_;
  std::vector<Eigen::VectorXd> vars_;
  Eigen::VectorXd var_, var_pre_;
  ObjFun obj_fun_;
  JacFun jac_fun_;
  HesFun hes_fun_;
};
} //namespace opt
#endif //OPTIMIZER_HPP