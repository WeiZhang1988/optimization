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
#include "line_searcher.hpp"

namespace opt{
class Base_Solver{
  public:
  Base_Solver(Eigen::VectorXd _var, \
              double  _epsilon, \
              std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
              ObjFun  _obj_fun      = nullptr, \
              JacFun  _jac_fun      = nullptr, \
              HesFun  _hes_fun      = nullptr, \
              ConsFun _eq_cons      = nullptr, \
              ConsFun _ieq_cons     = nullptr, \
              JacCons _jac_eq_cons  = nullptr, \
              JacCons _jac_ieq_cons = nullptr,
              HesCons _hes_eq_cons  = nullptr, \
              HesCons _hes_ieq_cons = nullptr) : 
              var_(_var), 
              epsilon_(_epsilon), 
              sptr_line_searcher_(_sptr_line_searcher),
              obj_fun_(_obj_fun),
              jac_fun_(_jac_fun),
              hes_fun_(_hes_fun) {
      sptr_line_searcher_->set_obj_fun(obj_fun_);
      sptr_line_searcher_->set_jac_fun(jac_fun_);
      sptr_line_searcher_->set_hes_fun(hes_fun_);
    }
  void set_var(Eigen::VectorXd _var) {
    var_ = _var;
  }
  void set_alpha(double _alpha) {
    alpha_ = _alpha;
  }
  void set_obj_fun(ObjFun _obj_fun) {
    obj_fun_ = _obj_fun;
    sptr_line_searcher_->set_obj_fun(obj_fun_);
  }
  void set_jac_fun(JacFun _jac_fun) {
    jac_fun_ = _jac_fun;
    sptr_line_searcher_->set_jac_fun(jac_fun_);
  }
  void set_hes_fun(HesFun _hes_fun) {
    hes_fun_ = _hes_fun;
    sptr_line_searcher_->set_hes_fun(hes_fun_);
  }
  void set_eq_cons(ConsFun _eq_cons) {
    eq_cons_ = _eq_cons;
  }
  void set_ieq_cons(ConsFun _ieq_cons) {
    ieq_cons_ = _ieq_cons;
  }
  void set_jac_eq_cons(JacCons _jac_eq_cons) {
    jac_eq_cons_ = _jac_eq_cons;
  }
  void set_jac_ieq_cons(JacCons _jac_ieq_cons) {
    jac_ieq_cons_ = _jac_ieq_cons;
  }
  void set_hes_eq_cons(HesCons _hes_eq_cons) {
    hes_eq_cons_ = _hes_eq_cons;
  }
  void set_hes_ieq_cons(HesCons _hes_ieq_cons) {
    hes_ieq_cons_ = _hes_ieq_cons;
  }
  virtual void init_for_solve() {
    dir_ = -jac_fun_(var_);
    sptr_line_searcher_->forward_backward();
  }
  virtual void update_params() {}
  virtual bool ending_condition() {
    return false;
  }
  virtual Eigen::VectorXd solve() = 0;
  protected:
  Eigen::VectorXd var_, dir_;
  ObjFun obj_fun_;
  JacFun jac_fun_;
  HesFun hes_fun_;
  ConsFun eq_cons_;     
  ConsFun ieq_cons_;    
  JacCons jac_eq_cons_;
  JacCons jac_ieq_cons_;
  HesCons hes_eq_cons_;
  HesCons hes_ieq_cons_;
  double alpha_;
  double epsilon_ = 1e-10;
  std::shared_ptr<Base_Line_Searcher> sptr_line_searcher_;
};
class Gradient_Descent_Solver : public Base_Solver{
  public:
  Gradient_Descent_Solver(Eigen::VectorXd _var, \
                          double _epsilon, \
                          std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
                          ObjFun _obj_fun = nullptr, \
                          JacFun _jac_fun = nullptr, \
                          HesFun _hes_fun = nullptr) :
                          Base_Solver(_var, \
                                      _epsilon, \
                                      _sptr_line_searcher, \
                                      _obj_fun, \
                                      _jac_fun, \
                                      _hes_fun) {}
  Eigen::VectorXd solve() override {
    sptr_line_searcher_->set_var(var_);
    sptr_line_searcher_->set_dir(dir_);
    alpha_ = sptr_line_searcher_->search();
    dir_   = -jac_fun_(var_);
    var_ += alpha_ * dir_;
    return var_;
  }
  protected:
};
class Newton_Solver : public Base_Solver{
  public:
  Newton_Solver(Eigen::VectorXd _var, \
                double _epsilon, \
                std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
                ObjFun _obj_fun = nullptr, \
                JacFun _jac_fun = nullptr, \
                HesFun _hes_fun = nullptr) :
                Base_Solver(_var, \
                            _epsilon, \
                            _sptr_line_searcher, \
                            _obj_fun, \
                            _jac_fun, \
                            _hes_fun) {}
  Eigen::VectorXd solve() override {
    sptr_line_searcher_->set_var(var_);
    sptr_line_searcher_->set_dir(dir_);
    alpha_ = sptr_line_searcher_->search();
    dir_   = hes_fun_(var_).template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac_fun_(var_));
    var_ += alpha_ * dir_;
    return var_;
  }
  protected:
};
class DFP_Solver : public Base_Solver{
  public:
  DFP_Solver(Eigen::VectorXd _var, \
             Eigen::MatrixXd _hes, \
             double _epsilon, \
             std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
             ObjFun _obj_fun = nullptr, \
             JacFun _jac_fun = nullptr, \
             HesFun _hes_fun = nullptr) :
             Base_Solver(_var, \
                        _epsilon, \
                        _sptr_line_searcher, \
                         _obj_fun, \
                         _jac_fun, \
                         _hes_fun),
             hes_(_hes) {}
  Eigen::VectorXd solve() override {
    sptr_line_searcher_->set_var(var_);
    sptr_line_searcher_->set_dir(dir_);
    alpha_ = sptr_line_searcher_->search();
    Eigen::VectorXd jac  = jac_fun_(var_);
    dir_ = hes_.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac);
    Eigen::VectorXd y = jac_fun_(var_ + alpha_ * dir_) - jac;
    hes_ += (dir_ * dir_.transpose()) / (dir_.transpose() * y) - \
            (hes_ * y * y.transpose() * hes_) / (y.transpose() * hes_ * y);
    var_ += alpha_ * dir_;
    return var_;
  }
  protected:
  Eigen::MatrixXd hes_;
};
class BFGS_Solver : public Base_Solver{
  public:
  BFGS_Solver(Eigen::VectorXd _var, \
              Eigen::MatrixXd _hes, \
              double _epsilon, \
              std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
              ObjFun _obj_fun = nullptr, \
              JacFun _jac_fun = nullptr, \
              HesFun _hes_fun = nullptr) :
              Base_Solver(_var, \
                          _epsilon, \
                          _sptr_line_searcher, \
                          _obj_fun, \
                          _jac_fun, \
                          _hes_fun),
              hes_(_hes) {}
  Eigen::VectorXd solve() override {
    sptr_line_searcher_->set_var(var_);
    sptr_line_searcher_->set_dir(dir_);
    alpha_ = sptr_line_searcher_->search();
    Eigen::VectorXd jac = jac_fun_(var_);
    dir_ = hes_.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac);
    Eigen::VectorXd y = jac_fun_(var_ + alpha_ * dir_) - jac;
    hes_ += (y * y.transpose()) / (y.transpose() * dir_) - \
            (hes_ * dir_ * dir_.transpose() * hes_) / (dir_.transpose() * hes_ * dir_);
    var_ += alpha_ * dir_;
    return var_;
  }
  protected:
  Eigen::MatrixXd hes_;
};
class LBFGS_Solver : public Base_Solver{
  public:
  LBFGS_Solver(Eigen::VectorXd _var, \
               double _epsilon, \
               int  _storage_size, \
               std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
               ObjFun _obj_fun = nullptr, \
               JacFun _jac_fun = nullptr, \
               HesFun _hes_fun = nullptr) :
               Base_Solver(_var, \
                           _epsilon, \
                           _sptr_line_searcher, \
                           _obj_fun, \
                           _jac_fun, \
                           _hes_fun),
               storage_size_(_storage_size) {
                 sptr_stored_dirs_ = std::make_shared<FixedQueue<Eigen::VectorXd>>(storage_size_);
                 sptr_stored_ys_   = std::make_shared<FixedQueue<Eigen::VectorXd>>(storage_size_);
               }
  Eigen::VectorXd solve() override {
    sptr_line_searcher_->set_var(var_);
    sptr_line_searcher_->set_dir(dir_);
    alpha_ = sptr_line_searcher_->search();
    Eigen::VectorXd jac = jac_fun_(var_);
    Eigen::MatrixXd hes = Eigen::MatrixXd::Identity(var_.size(),var_.size());
    dir_ = hes.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac);
    Eigen::VectorXd y = jac_fun_(var_ + alpha_ * dir_) - jac;
    sptr_stored_dirs_->push(dir_);
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
    var_ += alpha_ * dir_;
    return var_;
  }
  protected:
  int storage_size_;
  std::shared_ptr<FixedQueue<Eigen::VectorXd>> sptr_stored_dirs_;
  std::shared_ptr<FixedQueue<Eigen::VectorXd>> sptr_stored_ys_;
};
class Conjugate_Gradient_Solver : public Base_Solver{
  public:
  enum class Beta_Type {
    Hestenes_stiefel,
    Polak_ribiere,
    Fletcher_reeves,
    Powell
  };
  Conjugate_Gradient_Solver(Eigen::VectorXd _var, \
                            double _epsilon, \
                            std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
                            ObjFun _obj_fun = nullptr, \
                            JacFun _jac_fun = nullptr, \
                            HesFun _hes_fun = nullptr, \
                            Beta_Type _beta_type = Beta_Type::Hestenes_stiefel) :
                            Base_Solver(_var, \
                                        _epsilon, \
                                        _sptr_line_searcher, \
                                        _obj_fun, \
                                        _jac_fun, \
                                        _hes_fun),
                            beta_type_(_beta_type) {}
  Eigen::VectorXd solve() override {
    sptr_line_searcher_->set_var(var_);
    sptr_line_searcher_->set_dir(dir_);
    alpha_ = sptr_line_searcher_->search();
    Eigen::VectorXd jac = jac_fun_(var_);
    dir_ = -jac_fun_(var_) - beta_ * (-jac);
    Eigen::VectorXd next_jac = jac_fun_(var_ + alpha_ * dir_);
    switch (beta_type_){
    case Beta_Type::Hestenes_stiefel:
      beta_ = hestenes_stiefel(dir_, jac, next_jac);
      break;
    case Beta_Type::Polak_ribiere:
      beta_ = polak_ribiere(dir_, jac, next_jac);
      break;
    case Beta_Type::Fletcher_reeves:
      beta_ = fletcher_reeves(dir_, jac, next_jac);
      break;
    case Beta_Type::Powell:
      beta_ = powell(dir_, jac, next_jac);
      break;
    default:
      beta_ = 0.0;
      break;
    }
    var_ += alpha_ * dir_;
    return var_;
  }
  protected:
  double hestenes_stiefel(Eigen::VectorXd _dir, Eigen::VectorXd _jac, Eigen::VectorXd _next_jac) {
    if (0.0 == _dir.dot(_next_jac - _jac)){
      return 0.0;
    } else {
      return -_next_jac.dot(_next_jac - _jac) / _dir.dot(_next_jac - _jac);
    }
  }
  double polak_ribiere(Eigen::VectorXd _dir, Eigen::VectorXd _jac, Eigen::VectorXd _next_jac) {
    if (0.0 == _dir.dot(_jac)){
      return 0.0;
    } else {
      return _next_jac.dot(_next_jac - _jac) / _dir.dot(_jac);
    }
  }
  double fletcher_reeves(Eigen::VectorXd _dir, Eigen::VectorXd _jac, Eigen::VectorXd _next_jac) {
    if (0.0 == _jac.dot(_jac)){
      return 0.0;
    } else {
      return -_next_jac.dot(_next_jac) / _jac.dot(_jac);
    }
  }
  double powell(Eigen::VectorXd _dir, Eigen::VectorXd _jac, Eigen::VectorXd _next_jac) {
    if (0.0 == _jac.dot(_jac)){
      return 0.0;
    } else {
      return std::max(0.0, _next_jac.dot(_next_jac - _jac) / _jac.dot(_jac));
    }
  }
  double beta_ = 0.0;
  Beta_Type beta_type_ = Beta_Type::Hestenes_stiefel;
};
class Augmented_Lagrangian_Solver : public Base_Solver{
  public:
  Augmented_Lagrangian_Solver(Eigen::VectorXd _var, \
                              double _epsilon, \
                              std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
                              Eigen::VectorXd _lambdas, \
                              Eigen::VectorXd _mus, \
                              double _sigma = 2.0, \
                              double _enta  = 1e-6, \
                              double _beta1 = 0.3, \
                              double _beta2 = 0.6, \
                              double _rho   = 5.0, \
                              ObjFun _obj_fun = nullptr, \
                              JacFun _jac_fun = nullptr, \
                              HesFun _hes_fun = nullptr, \
                              ConsFun _eq_cons      = nullptr, \
                              ConsFun _ieq_cons     = nullptr, \
                              JacCons _jac_eq_cons  = nullptr, \
                              JacCons _jac_ieq_cons = nullptr) :
                              Base_Solver(_var, \
                                          _epsilon, \
                                          _sptr_line_searcher, \
                                          _obj_fun, \
                                          _jac_fun, \
                                          _hes_fun, \
                                          _eq_cons, \
                                          _ieq_cons, \
                                          _jac_eq_cons, \
                                          _jac_ieq_cons),  
                              lambdas_(_lambdas),
                              mus_(_mus),
                              sigma_(_sigma),
                              enta_(_enta),
                              beta1_(_beta1),
                              beta2_(_beta2),
                              rho_(_rho){
    assert(sigma_>0 && \
           epsilon_>0 && \
           enta_>0 && \
           beta1_>0 && \
           beta2_>0 && \
           beta1_<1 && \
           beta2_<1 && \
           beta1_<beta2_ && 
           rho_>1);
    entk_ = 1 / sigma_;
    epsk_ = std::pow(sigma_, -beta1_);
  }
  double augmented_lagrange(Eigen::VectorXd _var) {
    Eigen::VectorXd ieq_items_1 = (mus_ / sigma_) + ieq_cons_(_var);
    ieq_items_1.unaryExpr([](double value) { return (value > 0.0) ? value * value : 0.0; });
    Eigen::VectorXd ieq_items_2 = (mus_ / sigma_);
    ieq_items_2.unaryExpr([](double value) { return (value > 0.0) ? value * value : 0.0; });
    Eigen::VectorXd ieq_items = ieq_items_1 - ieq_items_2;
    return obj_fun_(_var) + lambdas_.dot(eq_cons_(_var)) + 0.5 * sigma_ * eq_cons_(_var).squaredNorm() + 0.5 * sigma_ * ieq_items.sum();
  }
  Eigen::VectorXd jac_augmented_lagrange(Eigen::VectorXd _var) {
    Eigen::VectorXd ieq_rlx_value = -((mus_ / sigma_) + ieq_cons_(_var));
    Eigen::MatrixXd jac_ieq_cons_value = jac_ieq_cons_(_var);
    for (int i = 0; i < ieq_rlx_value.size(); i++) {
      if (ieq_rlx_value(i) < 0.0) {
        jac_ieq_cons_value.col(i).setZero();
      }
    }
    ieq_rlx_value.unaryExpr([](double value) { return (value > 0.0) ? value : 0.0; });
    return (jac_fun_(_var) + \
            (jac_eq_cons_(_var) * (lambdas_ + sigma_ * eq_cons_(_var))) + \
            jac_ieq_cons_value * (mus_ + sigma_ * (ieq_cons_(_var) + ieq_rlx_value)));
  }
  Eigen::MatrixXd hes_augmented_lagrange(Eigen::VectorXd _var) {
    Eigen::VectorXd ieq_rlx_value = -((mus_ / sigma_) + ieq_cons_(_var));
    Eigen::MatrixXd jac_ieq_cons_value = jac_ieq_cons_(_var);
    for (int i = 0; i < ieq_rlx_value.size(); i++) {
      if (ieq_rlx_value(i) < 0.0) {
        jac_ieq_cons_value.col(i).setZero();
      }
    }
    ieq_rlx_value.unaryExpr([](double value) { return (value > 0.0) ? value : 0.0; });
    return hes_fun_(_var) + jac_eq_cons_(_var) * sigma_ * jac_eq_cons_(_var).transpose() + jac_ieq_cons_value * sigma_ * jac_ieq_cons_value.transpose();
  }
  virtual void init_for_solve() {
    sptr_line_searcher_->set_obj_fun(std::bind(&Augmented_Lagrangian_Solver::augmented_lagrange,this,std::placeholders::_1));
    sptr_line_searcher_->set_jac_fun(std::bind(&Augmented_Lagrangian_Solver::jac_augmented_lagrange,this,std::placeholders::_1));
    sptr_line_searcher_->set_hes_fun(std::bind(&Augmented_Lagrangian_Solver::hes_augmented_lagrange,this,std::placeholders::_1));
    sptr_line_searcher_->forward_backward();
  }
  Eigen::VectorXd solve() override {
    Eigen::VectorXd ieq_rlx_value = -((mus_ / sigma_) + ieq_cons_(var_));
    Eigen::MatrixXd jac_ieq_cons_value = jac_ieq_cons_(var_);
    for (int i = 0; i < ieq_rlx_value.size(); i++) {
      if (ieq_rlx_value(i) < 0.0) {
        jac_ieq_cons_value.col(i).setZero();
      }
    }
    ieq_rlx_value.unaryExpr([](double value) { return (value > 0.0) ? value : 0.0; });
    int iter_num=0;
    do {
      jac_lag_ = (jac_fun_(var_) + \
                  (jac_eq_cons_(var_) * (lambdas_ + sigma_ * eq_cons_(var_))) + \
                  jac_ieq_cons_value * (mus_ + sigma_ * (ieq_cons_(var_) + ieq_rlx_value)));
      Eigen::MatrixXd hes_lag_ = hes_fun_(var_) + jac_eq_cons_(var_) * sigma_ * jac_eq_cons_(var_).transpose()+ jac_ieq_cons_value * sigma_ * jac_ieq_cons_value.transpose();
      dir_ = hes_lag_.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac_lag_);
      sptr_line_searcher_->set_var(var_);
      sptr_line_searcher_->set_dir(dir_);
      alpha_ = sptr_line_searcher_->search();
      var_ += alpha_ * dir_;
      iter_num++;
    } while (jac_lag_.norm() > entk_ && std::abs(eq_cons_(var_).norm()) > entk_ && ieq_cons_(var_).maxCoeff() > -std::abs(entk_));
    cons_violation_ = std::sqrt(eq_cons_(var_).squaredNorm() + (ieq_cons_(var_).cwiseMax(-(mus_ / sigma_))).squaredNorm());
    if (cons_violation_ < epsk_){
      if (cons_violation_ < epsilon_ && jac_lag_.norm() < enta_) {
        ending_cond_ = true;
        return var_;
      } else {
        lambdas_ += sigma_ * eq_cons_(var_);
        mus_ += sigma_ * ieq_cons_(var_);
        mus_.unaryExpr([](double value) { return (value > 0.0) ? value : 0.0; });
        entk_ /= sigma_;
        epsk_ *= std::pow(sigma_,-beta2_);
      }
    } else {
      sigma_ *= rho_;
      entk_ = 1 / sigma_;
      epsk_ = std::pow(sigma_, -beta1_);
    }
    ending_cond_ = false;
    return var_;
  }
  bool ending_condition() {
    return ending_cond_;
  }
  protected:
  Eigen::VectorXd jac_lag_, lambdas_, mus_;
  double sigma_, enta_, epsk_, entk_, beta1_, beta2_, rho_;
  double cons_violation_;
  bool ending_cond_ = false;
};
/*class Log_Barrier_Solver : public Base_Solver{
  public:
  Log_Barrier_Solver(Eigen::VectorXd _var, \
                     double _epsilon, \
                     std::shared_ptr<Base_Line_Searcher> _sptr_line_searcher, \
                     Eigen::VectorXd _lambdas, \
                     double _t = 1e-6, \
                     double _mu  = 2.0, \
                     ObjFun _obj_fun = nullptr, \
                     JacFun _jac_fun = nullptr, \
                     HesFun _hes_fun = nullptr, \
                     ConsFun _eq_cons      = nullptr, \
                     ConsFun _ieq_cons     = nullptr, \
                     JacCons _jac_eq_cons  = nullptr, \
                     JacCons _jac_ieq_cons = nullptr) :
                     Base_Solver(_var, \
                                 _epsilon, \
                                 _sptr_line_searcher, \
                                 _obj_fun, \
                                 _jac_fun, \
                                 _hes_fun, \
                                 _eq_cons, \
                                 _ieq_cons, \
                                 _jac_eq_cons, \
                                 _jac_ieq_cons),  
                     lambdas_(_lambdas),
                     t_(_t),
                     mu_(_mu){
    assert(t_>0 && mu_>1.0);
  }
  double lagrange(Eigen::VectorXd _var) {
    Eigen::VectorXd ieq_items = - ieq_cons_(_var);
    ieq_items.unaryExpr([](double value) { return -log(value); });
    return t_ * obj_fun_(_var) + ieq_items.sum();
  }
  Eigen::VectorXd jac_lagrange(Eigen::VectorXd _var) {
    Eigen::MatrixXd jac_ieq_cons_items = jac_ieq_cons_(_var);
    Eigen::VectorXd ieq_cons_values     = ieq_cons_(_var);
    assert(jac_ieq_values.cols() == ieq_cons_values.size())
    for (int i = 0; i < ieq_cons_values.size(); i++) {
      jac_ieq_cons_items.col(i) /= ieq_cons_values(i);
    }
    return (t_ * jac_fun_(_var) - jac_ieq_cons_items.sum());
  }
  Eigen::MatrixXd hes_lagrange(Eigen::VectorXd _var) {
    Eigen::MatrixXd jac_ieq_cons_items = jac_ieq_cons_(_var);
    Eigen::VectorXd ieq_cons_values    = ieq_cons_(_var);
    assert(jac_ieq_values.cols() == ieq_cons_values.size())
    Eigen::MatrixXd hes_tmp1 = Eigen::MatrixXd::Zeros(_var.size(),_var.size());
    for (int i = 0; i < ieq_cons_values.size(); i++) {
      Eigen::MatrixXd hes_tmp1 += jac_ieq_cons_items.col(i) * jac_ieq_cons_items.col(i).transpose() / std::sqrt(ieq_cons_values(i));
    }
    Eigen::MatrixXd hes_tmp2 = Eigen::MatrixXd::Zeros(_var.size(),_var.size());
    for (int i = 0; i < ieq_cons_values.size(); i++) {
      Eigen::MatrixXd hes_tmp2 += hes_ieq_cons_items[i] / ieq_cons_values(i);
    }
    return (t_ * hes_fun_(_var) + hes_tmp1 - hes_tmp2);
  }
  virtual void init_for_solve() {
    sptr_line_searcher_->set_obj_fun(std::bind(&Log_Barrier_Solver::lagrange,this,std::placeholders::_1));
    sptr_line_searcher_->set_jac_fun(std::bind(&Log_Barrier_Solver::jac_lagrange,this,std::placeholders::_1));
    sptr_line_searcher_->set_hes_fun(std::bind(&Log_Barrier_Solver::hes_lagrange,this,std::placeholders::_1));
    sptr_line_searcher_->forward_backward();
  }
  Eigen::VectorXd rhs(Eigen::VectorXd _var, Eigen::VectorXd _nv) {
    
  }
  Eigen::VectorXd solve() override {
    int iter_num=0;
    Eigen::MatrixXd rhs;
    jac_lagrange(_var) + jac_eq_cons_(_var) * nv;

    do {
      jac_lag_ = (jac_fun_(var_) + \
                  (jac_eq_cons_(var_) * (lambdas_ + sigma_ * eq_cons_(var_))) + \
                  jac_ieq_cons_value * (mus_ + sigma_ * (ieq_cons_(var_) + ieq_rlx_value)));
      Eigen::MatrixXd hes_lag_ = hes_fun_(var_) + jac_eq_cons_(var_) * sigma_ * jac_eq_cons_(var_).transpose()+ jac_ieq_cons_value * sigma_ * jac_ieq_cons_value.transpose();
      dir_ = hes_lag_.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-jac_lag_);
      sptr_line_searcher_->set_var(var_);
      sptr_line_searcher_->set_dir(dir_);
      alpha_ = sptr_line_searcher_->search();
      var_ += alpha_ * dir_;
      iter_num++;
    } while (iter_num < 1000);
    return var_;
  }
  bool ending_condition() {
    return ending_cond_;
  }
  protected:
  Eigen::VectorXd lambdas_;
  double t_, mu_;
  bool ending_cond_ = false;
};*/
} //namespace opt
#endif //SOLVER_HPP