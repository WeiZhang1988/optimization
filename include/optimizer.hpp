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
              HesFun _hes_fun) : 
              var_(_var), 
              epsilon_(_epsilon), 
              max_iter_num_(_max_iter_num),
              obj_fun_(_obj_fun),
              jac_fun_(_jac_fun),
              hes_fun_(_hes_fun) {}
  protected:
  Eigen::VectorXd var_;
  double epsilon_;
  int max_iter_num_;
  ObjFun obj_fun_;
  JacFun jac_fun_;
  HesFun hes_fun_;
};
class Gradient_Descent_Solver : public Base_Solver{
  public:
  Gradient_Descent_Solver(Eigen::VectorXd _var, \
                          double _alpha, \
                          double _epsilon, \
                          int _max_iter_num, \
                          ObjFun _obj_fun, \
                          JacFun _jac_fun, \
                          HesFun _hes_fun) :
                          Base_Solver(_var, \
                                      _epsilon, \
                                      _max_iter_num, \
                                      _obj_fun, \
                                      _jac_fun, \
                                      _hes_fun),
                          alpha_(_alpha) {
    var_pre_ = Eigen::VectorXd::Ones(var_.size()) * std::numeric_limits<double>::max() * 0.5;
  }
  void solve() {
    int iter_num = 0;
    while ((var_ - var_pre_).lpNorm<1>() > epsilon_ && \
           (std::abs(obj_fun_(var_) - obj_fun_(var_pre_)) > epsilon_) && \
           jac_fun_(var_).lpNorm<1>() > epsilon_ && \
           iter_num < max_iter_num_) {
      iter_num++;
      var_pre_ = var_;
      var_ -= (alpha_ * jac_fun_(var_));
    }
  }
  protected:
  double alpha_;
  Eigen::VectorXd var_pre_;
};
} //namespace opt
#endif //OPTIMIZER_HPP