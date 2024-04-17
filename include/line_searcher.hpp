#ifndef LINE_SEARCHER_HPP
#define LINE_SEARCHER_HPP

#include <cmath>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include "type_define.hpp"

namespace opt{
class Base_Line_Searcher{
  public:
  Base_Line_Searcher(double _alpha_init = 0.5, \
                     double _stepsize = 1.0, \
                     double _multiplier = 1.0) : 
                     alpha_init_(_alpha_init),
                     stepsize_(_stepsize),
                     multiplier_(_multiplier) {}
  protected:
  void forward_backward(Eigen::VectorXd _var, \
                        Eigen::VectorXd _dir, \
                        ObjFun _obj_fun) {
    double alpha0 = alpha_init_;
    double alpha1 = alpha_init_ + stepsize_;
    double alpha2 = alpha0;
    int iter_num = 0;
    while(true) {
      if (_obj_fun(_var + alpha0 * _dir) < _obj_fun(_var + (alpha1 + stepsize_) * _dir)) {
        if (iter_num == 0) {
          stepsize_ = -stepsize_;
          alpha1 = alpha0 + stepsize_;
          alpha2 = alpha1;
        } else {
          alpha_low_  = std::min(alpha1,alpha2);
          alpha_high_ = std::max(alpha1,alpha2);
          break;
        }
      } else {
        stepsize_ = multiplier_ * stepsize_;
        alpha2 = alpha0;
        alpha0 = alpha1;
        alpha1 += stepsize_;
      }
      iter_num++;
    }
  }
  double alpha_init_ = 0.5;
  double stepsize_   = 1.0;
  double multiplier_  = 1.0;
  double alpha_low_  = 0.0;
  double alpha_high_ = 1.0;
};
class Exact_Line_Searcher  : public Base_Line_Searcher{
  public:
  Exact_Line_Searcher(double _alpha_init  = 0.5, \
                      double _stepsize    = 1.0, \
                      double _multiplier = 1.0, \
                      double _epsilon    = 1e-6, \
                      int _max_iter_num  = 100) :
                      Base_Line_Searcher(_alpha_init, _stepsize, _multiplier), 
                      epsilon_(_epsilon),
                      max_iter_num_(_max_iter_num) {}
  double search_by_bisection(Eigen::VectorXd _var, \
                             Eigen::VectorXd _dir, \
                             ObjFun _obj_fun,
                             JacFun _jac_fun) {
    forward_backward(_var, _dir, _obj_fun);
    double obj = _obj_fun(_var);
    Eigen::VectorXd jac = _jac_fun(_var);
    int iter_num = 0;
    while (std::abs(alpha_high_ - alpha_low_) > epsilon_ && iter_num < max_iter_num_){
      iter_num++;
      double alpha_mid = (alpha_low_ + alpha_high_) / 2.0;
      if (_obj_fun(_var + alpha_mid * _dir) < (obj + jac.dot(_dir) * alpha_mid)) {
        alpha_low_ = alpha_mid;
      } else {
        alpha_high_ = alpha_mid;
      }
    }
    return (alpha_low_ + alpha_high_) / 2.0;
  }
  double search_by_golden_section(Eigen::VectorXd _var, \
                                  Eigen::VectorXd _dir, \
                                  ObjFun _obj_fun) {
    forward_backward(_var, _dir, _obj_fun);
    double golden_ratio   = (std::sqrt(5.0) - 1.0) / 2.0;
    double alpha_low_try  = alpha_high_ - (alpha_high_ - alpha_low_) * golden_ratio;
    double alpha_high_try = alpha_low_  + (alpha_high_ - alpha_low_) * golden_ratio;
    int iter_num = 0;
    while (std::abs(alpha_high_ - alpha_low_) > epsilon_ && iter_num < max_iter_num_){
      iter_num++;
      if (_obj_fun(_var + alpha_low_try * _dir) < _obj_fun(_var + alpha_high_try * _dir)) {
        alpha_high_    = alpha_high_try;
        alpha_high_try = alpha_low_try;
        alpha_low_try  = alpha_high_ - (alpha_high_ - alpha_low_) * golden_ratio;
      } else {
        alpha_low_     = alpha_low_try;
        alpha_low_try  = alpha_high_try;
        alpha_high_try = alpha_low_ + (alpha_high_ - alpha_low_) * golden_ratio;
      }
    }
    return (alpha_low_ + alpha_high_) / 2.0;
  }
  double search_by_Fibonacci(Eigen::VectorXd _var, \
                             Eigen::VectorXd _dir, \
                             ObjFun _obj_fun) {
    forward_backward(_var, _dir, _obj_fun);
    std::vector<int> fib(max_iter_num_ + 1);
    fib[0] = 0;
    fib[1] = 1;
    for (int i = 2; i < max_iter_num_ + 1; i++){
      fib[i] = fib[i-1] + fib[i-2];
    }
    double alpha_low_try  = alpha_low_ + (double)fib[max_iter_num_-2]/(double)fib[max_iter_num_] * (alpha_high_ - alpha_low_);
    double alpha_high_try = alpha_low_ + (double)fib[max_iter_num_-1]/(double)fib[max_iter_num_] * (alpha_high_ - alpha_low_);
    for (int i = 0; i < max_iter_num_ - 1; i++) {
      if (_obj_fun(_var + alpha_low_try * _dir) < _obj_fun(_var + alpha_high_try * _dir)) {
        alpha_high_    = alpha_high_try;
        alpha_high_try = alpha_low_try;
        alpha_low_try  = alpha_low_ - (alpha_high_ - alpha_low_) * (double)fib[max_iter_num_-i-2]/(double)fib[max_iter_num_-i];
      } else {
        alpha_low_     = alpha_low_try;
        alpha_low_try  = alpha_high_try;
        alpha_high_try = alpha_low_ + (alpha_high_ - alpha_low_) * (double)fib[max_iter_num_-i-1]/(double)fib[max_iter_num_-i];
      }
    }
    return (alpha_low_ + alpha_high_) / 2.0;
  }
  double search_by_Newton(Eigen::VectorXd _var, \
                          ObjFun _obj_fun, \
                          JacFun _jac_fun, \
                          HesFun _hes_fun) {
    Eigen::VectorXd jac = _jac_fun(_var);
    Eigen::MatrixXd hes = _hes_fun(_var);
    Eigen::VectorXd dir = hes.ldlt().solve(-jac);
    forward_backward(_var, dir, _obj_fun);
    double alpha = alpha_high_;
    double alpha_delta = epsilon_ + alpha;
    int iter_num = 0;
    while (std::abs(alpha_delta) > epsilon_ && std::abs(dir.dot(_jac_fun(_var + alpha * dir))) > epsilon_ && iter_num < max_iter_num_) {
      iter_num++;
      alpha_delta = dir.dot(_jac_fun(_var + alpha * dir)) / (dir.transpose() * _hes_fun(_var + alpha * dir) * dir);
      alpha -= alpha_delta;
    }
    return alpha;
  }
  double search_by_secant(Eigen::VectorXd _var, \
                          Eigen::VectorXd _dir, \
                          ObjFun _obj_fun, \
                          JacFun _jac_fun) {
    forward_backward(_var, _dir, _obj_fun);
    Eigen::VectorXd jac = _jac_fun(_var);
    double alpha = alpha_high_;
    double alpha_pre = alpha;
    double alpha_delta = epsilon_ + alpha;
    int iter_num = 0;
    while (std::abs(alpha_delta) > epsilon_ && std::abs(_dir.dot(_jac_fun(_var + alpha * _dir))) > epsilon_ && iter_num < max_iter_num_) {
      iter_num++;
      alpha_delta = (alpha == alpha_pre) ? 1.1 * epsilon_ : _dir.dot(_jac_fun(_var + alpha * _dir)) * (alpha - alpha_pre) / (_dir.dot(_jac_fun(_var + alpha * _dir)) - _dir.dot(_jac_fun(_var + alpha_pre * _dir)));
      alpha_pre = alpha;
      alpha -= alpha_delta;
    }
    return alpha;
  }
  double search_by_2pt_quad_interpo(Eigen::VectorXd _var, \
                                    Eigen::VectorXd _dir, \
                                    ObjFun _obj_fun, \
                                    JacFun _jac_fun) {
    forward_backward(_var, _dir, _obj_fun);
    Eigen::VectorXd jac = _jac_fun(_var);
    double alpha = alpha_high_;
    double alpha_pre = alpha;
    double alpha_delta = epsilon_ + alpha;
    int iter_num = 0;
    while (std::abs(alpha_delta) > epsilon_ && std::abs(_dir.dot(_jac_fun(_var + alpha * _dir))) > epsilon_ && iter_num < max_iter_num_) {
      iter_num++;
      alpha_delta = (alpha == alpha_pre) ? 1.1 * epsilon_ : 0.5 * _dir.dot(_jac_fun(_var + alpha * _dir)) * (alpha - alpha_pre) / (_dir.dot(_jac_fun(_var + alpha * _dir)) - (_obj_fun(_var + alpha * _dir) - _obj_fun(_var + alpha_pre * _dir)) / (alpha - alpha_pre));
      alpha_pre = alpha;
      alpha -= alpha_delta;
    }
    return alpha;
  }
  double search_by_3pt_quad_interpo(Eigen::VectorXd _var, \
                                    Eigen::VectorXd _dir, \
                                    ObjFun _obj_fun) {
    forward_backward(_var, _dir, _obj_fun);
    double alpha_mid = 0.5 * (alpha_high_ + alpha_low_);
    return 0.5 * (alpha_mid + alpha_low_) + 0.5 * (_obj_fun(_var + alpha_low_ * _dir) - _obj_fun(_var + alpha_mid * _dir)) * (alpha_mid - alpha_high_) * (alpha_high_ - alpha_low_) / ((alpha_mid - alpha_high_) * _obj_fun(_var + alpha_low_ * _dir) + (alpha_high_ - alpha_low_) * _obj_fun(_var + alpha_mid * _dir) + (alpha_low_ - alpha_mid) * _obj_fun(_var + alpha_high_ * _dir));
  }
  double search_by_2pt_cubic_interpo(Eigen::VectorXd _var, \
                                     Eigen::VectorXd _dir, \
                                     ObjFun _obj_fun, \
                                     JacFun _jac_fun) {
    forward_backward(_var, _dir, _obj_fun);
    int iter_num = 0;
    double alpha = alpha_high_;
    while (std::abs(_dir.dot(_jac_fun(_var + alpha * _dir))) > epsilon_ && iter_num < max_iter_num_) {
      iter_num++;
      double obj_low  = _obj_fun(_var + alpha_low_  * _dir);
      double obj_high = _obj_fun(_var + alpha_high_ * _dir);
      double jac_low  = _dir.dot(_jac_fun(_var + alpha_low_ * _dir));
      double jac_high = _dir.dot(_jac_fun(_var + alpha_high_ * _dir));
      double jac_low_high = jac_low * jac_high;
      double omega = 3.0 * (obj_high - obj_low) / (alpha_high_ - alpha_low_) - jac_low_high;
      double eta = std::sqrt(omega * omega - jac_low_high);
      double alpha = alpha_low_ + (eta - jac_low - omega) * (alpha_high_ - alpha_low_) / (2 * eta - jac_low + jac_high);
      if (_dir.dot(_jac_fun(_var + alpha * _dir)) > 0.0) {
        alpha_high_ = alpha;
      } else {
        alpha_low_  = alpha;
      }
    }
    return alpha;
  }
  protected:
  double epsilon_    = 1e-6;
  int max_iter_num_  = 100;
};
class InExact_Line_Searcher : public Base_Line_Searcher {
  public:
  InExact_Line_Searcher(double _alpha_low  = 0.0, \
                        double _alpha_high = 1.0, \
                        double _rho        = 0.1, \
                        double _sigma      = 0.4, \
                        int _max_iter_num  = 100) :
                        alpha_low_(_alpha_low),
                        alpha_high_(_alpha_high),
                        rho_(_rho),
                        sigma_(_sigma),
                        max_iter_num_(_max_iter_num) {}
  double search_by_armijo_goldstein(Eigen::VectorXd _var, \
                                    Eigen::VectorXd _dir, \
                                    ObjFun _obj_fun, \
                                    JacFun _jac_fun) {
    forward_backward(_var, _dir, _obj_fun);
    double alpha = 0.5 * (alpha_low_ + alpha_high_);
    double obj_zero  = _obj_fun(_var);
    double jac_zero  = _dir.dot(_jac_fun(_var));
    double obj_alpha = _obj_fun(_var + alpha * _dir);
    int iter_num = 0;
    while ((obj_alpha > obj_zero + rho_ * jac_zero * alpha || obj_alpha < obj_zero + sigma_ * jac_zero * alpha) && iter_num < max_iter_num_){
      iter_num++;
      if (obj_alpha > obj_zero + rho_ * jac_zero * alpha){
        alpha_low_  = alpha;
      } else if (obj_alpha < obj_zero + sigma_ * jac_zero * alpha){
        alpha_high_ = alpha;
      }
      alpha = 0.5 * (alpha_low_ + alpha_high_);
    }
    return alpha;
  }
  double search_by_wolfe_powell(Eigen::VectorXd _var, \
                                Eigen::VectorXd _dir, \
                                ObjFun _obj_fun, \
                                JacFun _jac_fun) {
    forward_backward(_var, _dir, _obj_fun);
    double alpha = 0.5 * (alpha_low_ + alpha_high_);
    double obj_zero  = _obj_fun(_var);
    double jac_zero  = _dir.dot(_jac_fun(_var));
    double obj_alpha = _obj_fun(_var + alpha * _dir);
    double jac_alpha = _dir.dot(_jac_fun(_var + alpha * _dir));
    int iter_num = 0;
    while ((obj_alpha > obj_zero + rho_ * jac_zero * alpha || jac_alpha < sigma_ * jac_zero) && iter_num < max_iter_num_){
      iter_num++;
      if (obj_alpha > obj_zero + rho_ * jac_zero * alpha){
        alpha_low_  = alpha;
      } else if (jac_alpha < sigma_ * jac_zero){
        alpha_high_ = alpha;
      }
      alpha = 0.5 * (alpha_low_ + alpha_high_);
      obj_alpha = _obj_fun(_var + alpha * _dir);
      jac_alpha = _dir.dot(_jac_fun(_var + alpha * _dir));
    }
    return alpha;
  }
  double search_by_strong_wolfe_powell(Eigen::VectorXd _var, \
                                       Eigen::VectorXd _dir, \
                                       ObjFun _obj_fun, \
                                       JacFun _jac_fun) {
    forward_backward(_var, _dir, _obj_fun);
    double alpha = 0.5 * (alpha_low_ + alpha_high_);
    double obj_zero  = _obj_fun(_var);
    double jac_zero  = _dir.dot(_jac_fun(_var));
    double obj_alpha = _obj_fun(_var + alpha * _dir);
    double jac_alpha = _dir.dot(_jac_fun(_var + alpha * _dir));
    int iter_num = 0;
    while ((obj_alpha > obj_zero + rho_ * jac_zero * alpha || std::abs(jac_alpha) > sigma_ * std::abs(jac_zero)) && iter_num < max_iter_num_){
      iter_num++;
      if (obj_alpha > obj_zero + rho_ * jac_zero * alpha){
        alpha_low_  = alpha;
      } else if (std::abs(jac_alpha) > sigma_ * std::abs(jac_zero)){
        alpha_high_ = alpha;
      }
      alpha = 0.5 * (alpha_low_ + alpha_high_);
      obj_alpha = _obj_fun(_var + alpha * _dir);
      jac_alpha = _dir.dot(_jac_fun(_var + alpha * _dir));
    }
    return alpha;
  }
  protected:
  double alpha_low_  = 0.0;
  double alpha_high_ = 1.0;
  double rho_        = 0.1;
  double sigma_      = 0.4;
  int max_iter_num_  = 100;
};
} //namespace opt
#endif //LINE_SEARCHER_HPP