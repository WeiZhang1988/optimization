#ifndef SOLVER_HPP
#define SOLVER_HPP


#include <cmath>
#include <limits>
#include <vector>
#include <cassert>
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
  void set_alpha(double _alpha) {
    alpha_ = _alpha;
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
  double alpha_;
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
  Newton_Solver(Eigen::VectorXd _var, \
                ObjFun _obj_fun = nullptr, \
                JacFun _jac_fun = nullptr, \
                HesFun _hes_fun = nullptr) :
                Base_Solver(_var, \
                            _obj_fun, \
                            _jac_fun, \
                            _hes_fun) {}
  Eigen::VectorXd solve() override {
    return hes_fun_(var_).template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac_fun_(var_));
  }
  protected:
};
class DFP_Solver : public Base_Solver{
  public:
  DFP_Solver(Eigen::VectorXd _var, \
             Eigen::MatrixXd _hes, \
             ObjFun _obj_fun = nullptr, \
             JacFun _jac_fun = nullptr, \
             HesFun _hes_fun = nullptr) :
             Base_Solver(_var, \
                         _obj_fun, \
                         _jac_fun, \
                         _hes_fun),
             hes_(_hes) {}
  Eigen::VectorXd solve() override {
    Eigen::VectorXd jac = jac_fun_(var_);
    Eigen::VectorXd dir = hes_.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac);
    Eigen::VectorXd y = jac_fun_(var_ + 1.0 * dir) - jac;
    hes_ += (dir * dir.transpose()) / (dir.transpose() * y) - \
            (hes_ * y * y.transpose() * hes_) / (y.transpose() * hes_ * y);
    return dir;
  }
  protected:
  Eigen::MatrixXd hes_;
  Eigen::VectorXd jac_pre_;
};
class BFGS_Solver : public Base_Solver{
  public:
  BFGS_Solver(Eigen::VectorXd _var, \
              Eigen::MatrixXd _hes, \
              ObjFun _obj_fun = nullptr, \
              JacFun _jac_fun = nullptr, \
              HesFun _hes_fun = nullptr) :
              Base_Solver(_var, \
                          _obj_fun, \
                          _jac_fun, \
                          _hes_fun),
              hes_(_hes) {}
  Eigen::VectorXd solve() override {
    Eigen::VectorXd jac = jac_fun_(var_);
    Eigen::VectorXd dir = hes_.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac);
    Eigen::VectorXd y = jac_fun_(var_ + 1.0 * dir) - jac;
    hes_ += (y * y.transpose()) / (y.transpose() * dir) - \
            (hes_ * dir * dir.transpose() * hes_) / (dir.transpose() * hes_ * dir);
    return dir;
  }
  protected:
  Eigen::MatrixXd hes_;
  Eigen::VectorXd jac_pre_;
};
class LBFGS_Solver : public Base_Solver{
  public:
  LBFGS_Solver(Eigen::VectorXd _var, \
               int  _storage_size, \
               ObjFun _obj_fun = nullptr, \
               JacFun _jac_fun = nullptr, \
               HesFun _hes_fun = nullptr) :
               Base_Solver(_var, \
                           _obj_fun, \
                           _jac_fun, \
                           _hes_fun),
               storage_size_(_storage_size) {
                 sptr_stored_dirs_ = std::make_shared<FixedQueue<Eigen::VectorXd>>(storage_size_);
                 sptr_stored_ys_   = std::make_shared<FixedQueue<Eigen::VectorXd>>(storage_size_);
               }
  Eigen::VectorXd solve() override {
    Eigen::VectorXd jac = jac_fun_(var_);
    Eigen::MatrixXd hes = Eigen::MatrixXd::Identity(var_.size(),var_.size());
    Eigen::VectorXd dir = hes.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac);
    Eigen::VectorXd y = jac_fun_(var_ + 1.0 * dir) - jac;
    sptr_stored_dirs_->push(dir);
    sptr_stored_ys_->push(y);
    assert(sptr_stored_dirs_->size() == sptr_stored_ys_->size());
    auto dir_iter = sptr_stored_dirs_->cbegin();
    auto y_iter   = sptr_stored_ys_->cbegin();
    for (int i=0; i<sptr_stored_dirs_->size(); i++){
      hes += ((*y_iter) * (*y_iter).transpose()) / ((*y_iter).transpose() * (*dir_iter)) - \
             (hes * (*dir_iter) * (*dir_iter).transpose() * hes) / ((*dir_iter).transpose() * hes * (*dir_iter));
      dir_iter++;
      y_iter++;
    }
    return dir;
  }
  protected:
  int storage_size_;
  Eigen::VectorXd jac_pre_;
  std::shared_ptr<FixedQueue<Eigen::VectorXd>> sptr_stored_dirs_;
  std::shared_ptr<FixedQueue<Eigen::VectorXd>> sptr_stored_ys_;
};
} //namespace opt
#endif //SOLVER_HPP