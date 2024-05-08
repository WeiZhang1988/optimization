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
  void set_var(Eigen::VectorXd _var) {
    var_ = _var;
  }
  virtual double obj_fun(Eigen::VectorXd _var)          = 0;
  virtual Eigen::VectorXd jac_fun(Eigen::VectorXd _var) = 0;
  virtual Eigen::MatrixXd hes_fun(Eigen::VectorXd _var) {return Eigen::MatrixXd::Identity(1,1);}
  virtual Eigen::VectorXd eq_cons(Eigen::VectorXd _var) {return Eigen::VectorXd::Zero(1);}
  virtual Eigen::VectorXd ieq_cons(Eigen::VectorXd _var){return Eigen::VectorXd::Zero(1);}
  virtual Eigen::MatrixXd jac_eq_cons(Eigen::VectorXd _var)  {return Eigen::MatrixXd::Identity(1,1);}
  virtual Eigen::MatrixXd jac_ieq_cons(Eigen::VectorXd _var) {return Eigen::MatrixXd::Identity(1,1);}
  virtual std::vector<Eigen::MatrixXd> hes_eq_cons(Eigen::VectorXd _var)  {return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Identity(1,1)};}
  virtual std::vector<Eigen::MatrixXd> hes_ieq_cons(Eigen::VectorXd _var)  {return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Identity(1,1)};}
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
    return 0.5 * double((_var - target_).transpose() * weight_mat_ * (_var - target_)) + double(coe_vec_.transpose() * _var);
  }
  Eigen::VectorXd jac_fun(Eigen::VectorXd _var) override {
    return weight_mat_ * (_var - target_) + coe_vec_;
  }
  Eigen::MatrixXd hes_fun(Eigen::VectorXd _var) override {
    return weight_mat_;
  }
  protected:
  Eigen::MatrixXd weight_mat_;
  Eigen::VectorXd coe_vec_;
};
class QP_w_Constraint : public Base_Problem {
  public:
  QP_w_Constraint(Eigen::VectorXd _var, \
                  Eigen::VectorXd _target, \
                  Eigen::MatrixXd _weight_mat, \
                  Eigen::VectorXd _coe_vec) :
                  Base_Problem(_var, _target),
                  weight_mat_(_weight_mat),
                  coe_vec_(_coe_vec) {}
  double obj_fun(Eigen::VectorXd _var) override {
    return 0.5 * double((_var - target_).transpose() * weight_mat_ * (_var - target_)) + double(coe_vec_.transpose() * _var);
  }
  Eigen::VectorXd jac_fun(Eigen::VectorXd _var) override {
    return weight_mat_ * (_var - target_) + coe_vec_;
  }
  Eigen::MatrixXd hes_fun(Eigen::VectorXd _var) override {
    return weight_mat_;
  }
  Eigen::VectorXd eq_cons(Eigen::VectorXd _var) override {
    Eigen::VectorXd eq_value(1);
    eq_value << _var(0) * _var(0) + 2 * _var(0) - _var(1);
    return eq_value;
  }
  Eigen::VectorXd ieq_cons(Eigen::VectorXd _var) override {
    Eigen::VectorXd G(2);
    G << 1.0, -1.0;
    Eigen::VectorXd ieq_value(1);
    ieq_value << G.dot(_var) - 1.0;
    return ieq_value;
  }
  Eigen::MatrixXd jac_eq_cons(Eigen::VectorXd _var) override  {
    Eigen::MatrixXd jac_eq_value(2,1);
    jac_eq_value << 2 * _var(0) + 2, -1;
    return jac_eq_value;
  }
  Eigen::MatrixXd jac_ieq_cons(Eigen::VectorXd _var) override {
    Eigen::MatrixXd jac_ieq_value(2,1);
    jac_ieq_value << 1.0, -1.0;
    return jac_ieq_value;
  }
  std::vector<Eigen::MatrixXd> hes_eq_cons(Eigen::VectorXd _var) override  {
    std::vector<Eigen::MatrixXd> hes_eq_values;
    Eigen::MatrixXd hes_eq_value(2,2);
    hes_eq_value << 2.0, 0.0, 0.0, 0.0;
    hes_eq_values.push_back(hes_eq_value);
    return hes_eq_values;
  }
  std::vector<Eigen::MatrixXd> hes_ieq_cons(Eigen::VectorXd _var) override {
    std::vector<Eigen::MatrixXd> hes_ieq_values;
    Eigen::MatrixXd hes_ieq_value(2,2);
    hes_ieq_value << 0.0, 0.0, 0.0, 0.0;
    hes_ieq_values.push_back(hes_ieq_value);
    return hes_ieq_values;
  }
  protected:
  Eigen::MatrixXd weight_mat_;
  Eigen::VectorXd coe_vec_;
};
} //namespace opt
#endif //PROBLEM_HPP