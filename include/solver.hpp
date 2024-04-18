#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <cmath>
#include <limits>
#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Dense>
#include "type_define.hpp"

namespace opt{
class Base_Solver{
  public:
  Base_Solver(Eigen::VectorXd _var, \
              ObjFun _obj_fun = nullptr, \
              JacFun _jac_fun = nullptr, \
              HesFun _hes_fun = nullptr) : 
              var_(_var), 
              obj_fun_(_obj_fun),
              jac_fun_(_jac_fun),
              hes_fun_(_hes_fun) {}
  void set_var(Eigen::VectorXd _var) {
    var_ = _var;
  }
  void set_obj_fun(ObjFun _obj_fun) {
    obj_fun_ = _obj_fun;
  }
  void set_jac_fun(JacFun _jac_fun) {
    jac_fun_ = _jac_fun;
  }
  void set_hes_fun(HesFun _hes_fun) {
    hes_fun_ = _hes_fun;
  }
  virtual Eigen::VectorXd solve() = 0;
  protected:
  Eigen::VectorXd var_;
  ObjFun obj_fun_;
  JacFun jac_fun_;
  HesFun hes_fun_;
};
class Gradient_Descent_Solver : public Base_Solver{
  public:
  Gradient_Descent_Solver(Eigen::VectorXd _var, \
                          ObjFun _obj_fun = nullptr, \
                          JacFun _jac_fun = nullptr, \
                          HesFun _hes_fun = nullptr) :
                          Base_Solver(_var, \
                                      _obj_fun, \
                                      _jac_fun, \
                                      _hes_fun) {}
  Eigen::VectorXd solve() override {
    return -jac_fun_(var_);
  }
  protected:
};
class Newton_Solver : public Base_Solver{
  public:
  protected:
};
} //namespace opt
#endif //SOLVER_HPP