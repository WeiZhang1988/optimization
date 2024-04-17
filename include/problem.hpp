#ifndef PROBLEM_HPP
#define PROBLEM_HPP

#include <cmath>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include "type_define.hpp"

namespace opt{
class Base_Problem {
  public:
  Base_Problem(Eigen::VectorXd _var, \
               Eigen::VectorXd _target) :
               var_(_var),
               target_(_target) {}
  Eigen::VectorXd get_var() {
    return var_;
  }
  Eigen::VectorXd get_target() {
    return target_;
  }
  void set_var(Eigen::VectorXd _var) {
    var_ = _var;
  }
  void set_target(Eigen::VectorXd _target) {
    target_ = _target;
  }
  virtual double obj_fun(Eigen::VectorXd _var)          = 0;
  virtual Eigen::VectorXd jac_fun(Eigen::VectorXd _var) = 0;
  virtual Eigen::MatrixXd hes_fun(Eigen::VectorXd _var) = 0;
  protected:
  Eigen::VectorXd var_;
  Eigen::VectorXd target_;
};
class QP_wo_Constraint : public Base_Problem {
  public:
  QP_wo_Constraint(Eigen::VectorXd _var, \
                   Eigen::VectorXd _target, \
                   Eigen::MatrixXd _weight_mat, \
                   Eigen::VectorXd _coe_vec) :
                   Base_Problem(_var, _target),
                   weight_mat_(_weight_mat),
                   coe_vec_(_coe_vec) {}
  double obj_fun(Eigen::VectorXd _var) override {
    return 0.5 * (_var - target_).transpose() * weight_mat_ * (_var - target_) + coe_vec_.transpose() * _var;
  }
  Eigen::VectorXd jac_fun(Eigen::VectorXd _var) {
    return weight_mat_ * (_var - target_) + coe_vec_;
  }
  Eigen::MatrixXd hes_fun(Eigen::VectorXd _var) {
    return weight_mat_;
  }
  protected:
  Eigen::MatrixXd weight_mat_;
  Eigen::VectorXd coe_vec_;
};
} //namespace opt
#endif //PROBLEM_HPP