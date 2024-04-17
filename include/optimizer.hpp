#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <cmath>
#include <limits>
#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Dense>
#include "type_define.hpp"
#include "line_searcher.hpp"

namespace opt{
class Base_Solver{
  public:
  Base_Solver(Eigen::VectorXd _var, \
              double _epsilon, \
              int _max_iter_num, \
              ObjFun _obj_fun, \
              JacFun _jac_fun, \
              HesFun _hes_fun, \
              std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher) : 
              var_(_var), 
              epsilon_(_epsilon), 
              max_iter_num_(_max_iter_num),
              obj_fun_(_obj_fun),
              jac_fun_(_jac_fun),
              hes_fun_(_hes_fun),
              sptr_line_searcher_(_sptr_line_searcher){}
  protected:
  Eigen::VectorXd var_;
  double epsilon_;
  int max_iter_num_;
  ObjFun obj_fun_;
  JacFun jac_fun_;
  HesFun hes_fun_;
  std::shared_ptr<Base_Line_Searcher> sptr_line_searcher_;
};
class Gradient_Descent_Solver : public Base_Solver{
  public:
  Gradient_Descent_Solver(Eigen::VectorXd _var, \
                          double _alpha, \
                          double _epsilon, \
                          int _max_iter_num, \
                          ObjFun _obj_fun, \
                          JacFun _jac_fun, \
                          HesFun _hes_fun, \
                          std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher) :
                          Base_Solver(_var, \
                                      _epsilon, \
                                      _max_iter_num, \
                                      _obj_fun, \
                                      _jac_fun, \
                                      _hes_fun,
                                      _sptr_line_searcher),
                          alpha_(_alpha) {
    var_pre_ = Eigen::VectorXd::Ones(var_.size()) * std::numeric_limits<double>::max() * 0.5;
  }
  Eigen::VectorXd solve() {
    int iter_num = 0;
    while ((var_ - var_pre_).lpNorm<1>() > epsilon_ && \
           (std::abs(obj_fun_(var_) - obj_fun_(var_pre_)) > epsilon_) && \
           jac_fun_(var_).lpNorm<1>() > epsilon_ && \
           iter_num < max_iter_num_) {
      iter_num++;
      var_pre_ = var_;
      Eigen::VectorXd jac = jac_fun_(var_);
      var_ -= (alpha_ * jac);
      sptr_line_searcher_->update_var(var_);
      sptr_line_searcher_->update_dir(jac);
      sptr_line_searcher_->search();
    }
    return var_;
  }
  protected:
  double alpha_;
  Eigen::VectorXd var_pre_;
};
class Newton_Solver : public Base_Solver{
  public:
  protected:
  double alpha_;
};
} //namespace opt
#endif //OPTIMIZER_HPP