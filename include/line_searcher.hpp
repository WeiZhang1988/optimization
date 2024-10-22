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
  Base_Line_Searcher(Eigen::VectorXd _var, \
                     Eigen::VectorXd _dir, \
                     ObjFun _obj_fun = nullptr, \
                     JacFun _jac_fun = nullptr, \
                     HesFun _hes_fun = nullptr, \
                     double _alpha_init = 0.5, \
                     double _stepsize   = 1.0, \
                     double _multiplier = 1.0, \
                     int _max_iter_num = 1000) : 
                     var_(_var),
                     dir_(_dir),
                     obj_fun_(_obj_fun), 
                     jac_fun_(_jac_fun), 
                     hes_fun_(_hes_fun), 
                     alpha_init_(_alpha_init),
                     stepsize_(_stepsize),
                     multiplier_(_multiplier),
                     max_iter_num_(_max_iter_num) {}
  void set_var(Eigen::VectorXd _var) {
    var_ = _var;
  }
  void set_dir(Eigen::VectorXd _dir) {
    dir_ = _dir;
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
  void forward_backward() {
    if (std::abs(obj_fun_(var_ + alpha_init_ * dir_) - obj_fun_(var_ + (alpha_init_ + stepsize_) * dir_)) < 1e-6){
      alpha_low_  = alpha_init_;
      alpha_high_ = alpha_init_ + stepsize_;
    } else if (obj_fun_(var_ + alpha_init_ * dir_) < obj_fun_(var_ + (alpha_init_ + stepsize_) * dir_)){
      alpha_high_ = alpha_init_ + stepsize_;
      while (obj_fun_(var_ + (alpha_init_ - multiplier_ * stepsize_) * dir_) < obj_fun_(var_ + alpha_init_ * dir_)){
        multiplier_ += 1.0;
      }
      alpha_low_  = std::max(eps_,alpha_init_ - multiplier_ * stepsize_);
    } else {
      alpha_low_  = alpha_init_ + stepsize_;
      do {
        multiplier_ += 1.0;
      } while (obj_fun_(var_ + (alpha_init_ + multiplier_ * stepsize_) * dir_) < obj_fun_(var_ + (alpha_init_ + stepsize_) * dir_));
      alpha_high_ = alpha_init_ + multiplier_ * stepsize_;
    }
  }
  virtual double search() = 0;
  protected:
  Eigen::VectorXd var_;
  Eigen::VectorXd dir_;
  ObjFun obj_fun_;
  JacFun jac_fun_;
  HesFun hes_fun_;
  double alpha_init_ = 0.5;
  double stepsize_   = 1.0;
  double multiplier_ = 1.0;
  double alpha_low_  = 0.0;
  double alpha_high_ = 1e10;
  int max_iter_num_  = 1000;
  double eps_ = 1e-10;
};
class Exact_Line_Searcher  : public Base_Line_Searcher{
  public:
  enum class Search_Type {
    Bisection,
    Golden_section,
    Fibonacci,
    Newton,
    Secant,
    Two_pt_quad_interpo,
    Three_pt_quad_interpo,
    Two_pt_cubic_interpo
  };
  Exact_Line_Searcher(Eigen::VectorXd _var, \
                      Eigen::VectorXd _dir, \
                      ObjFun _obj_fun = nullptr, \
                      JacFun _jac_fun = nullptr, \
                      HesFun _hes_fun = nullptr, \
                      double _alpha_init  = 0.5, \
                      double _stepsize    = 1.0, \
                      double _multiplier = 1.0, \
                      int _max_iter_num  = 1000, \
                      double _epsilon    = 1e-4, \
                      Search_Type _search_type = Search_Type::Bisection) :
                      Base_Line_Searcher(_var, _dir, _obj_fun, _jac_fun, _hes_fun, _alpha_init, _stepsize, _multiplier, _max_iter_num), 
                      epsilon_(_epsilon),
                      search_type_(_search_type) {}
  double search() override {
    switch (search_type_){
    case Search_Type::Bisection:
      return search_by_bisection();
    case Search_Type::Golden_section:
      return search_by_golden_section();
    case Search_Type::Fibonacci:
      return search_by_Fibonacci();
    case Search_Type::Newton:
      return search_by_Newton();
    case Search_Type::Secant:
      return search_by_secant();
    case Search_Type::Two_pt_quad_interpo:
      return search_by_2pt_quad_interpo();
    case Search_Type::Three_pt_quad_interpo:
      return search_by_3pt_quad_interpo();
    case Search_Type::Two_pt_cubic_interpo:
      return search_by_2pt_cubic_interpo();
    default:
      return 0.0;
    }
  }
  double search_by_bisection() {
    double obj = obj_fun_(var_);
    Eigen::VectorXd jac = jac_fun_(var_);
    double alpha_mid = 0.5 * (alpha_low_ + alpha_high_);
    int iter_num = 0;
    while (std::abs(dir_.dot(jac_fun_(var_ + alpha_mid * dir_))) > epsilon_ && (alpha_high_ - alpha_low_) > epsilon_ && iter_num < max_iter_num_){
      iter_num++;
      if (dir_.dot(jac_fun_(var_ + alpha_mid * dir_)) > 0) {
        alpha_high_ = alpha_mid;
      } else {
        alpha_low_ = alpha_mid;
      }
      alpha_mid = 0.5 * (alpha_low_ + alpha_high_);
    }
    return std::max(eps_, 0.5 * (alpha_low_ + alpha_high_));
  }
  double search_by_golden_section() {
    double golden_ratio   = 0.5 * (std::sqrt(5.0) - 1.0);
    double alpha_low_try  = alpha_high_ - (alpha_high_ - alpha_low_) * golden_ratio;
    double alpha_high_try = alpha_low_  + (alpha_high_ - alpha_low_) * golden_ratio;
    int iter_num = 0;
    while (std::abs(alpha_high_ - alpha_low_) > epsilon_ && iter_num < max_iter_num_){
      iter_num++;
      if (obj_fun_(var_ + alpha_low_try * dir_) < obj_fun_(var_ + alpha_high_try * dir_)) {
        alpha_high_    = alpha_high_try;
        alpha_high_try = alpha_low_try;
        alpha_low_try  = alpha_high_ - (alpha_high_ - alpha_low_) * golden_ratio;
      } else {
        alpha_low_     = alpha_low_try;
        alpha_low_try  = alpha_high_try;
        alpha_high_try = alpha_low_ + (alpha_high_ - alpha_low_) * golden_ratio;
      }
    }
    return std::max(eps_, 0.5 * (alpha_low_ + alpha_high_));
  }
  double search_by_Fibonacci() {
    std::vector<int> fib(max_iter_num_ + 1);
    fib[0] = 0;
    fib[1] = 1;
    for (int i = 2; i < max_iter_num_ + 1; i++){
      fib[i] = fib[i-1] + fib[i-2];
    }
    double alpha_low_try  = alpha_low_ + (double)fib[max_iter_num_-2]/(double)fib[max_iter_num_] * (alpha_high_ - alpha_low_);
    double alpha_high_try = alpha_low_ + (double)fib[max_iter_num_-1]/(double)fib[max_iter_num_] * (alpha_high_ - alpha_low_);
    for (int i = 0; i < max_iter_num_ - 1; i++) {
      if (obj_fun_(var_ + alpha_low_try * dir_) < obj_fun_(var_ + alpha_high_try * dir_)) {
        alpha_high_    = alpha_high_try;
        alpha_high_try = alpha_low_try;
        alpha_low_try  = alpha_low_ - (alpha_high_ - alpha_low_) * (double)fib[max_iter_num_-i-2]/(double)fib[max_iter_num_-i];
      } else {
        alpha_low_     = alpha_low_try;
        alpha_low_try  = alpha_high_try;
        alpha_high_try = alpha_low_ + (alpha_high_ - alpha_low_) * (double)fib[max_iter_num_-i-1]/(double)fib[max_iter_num_-i];
      }
    }
    return std::max(eps_, 0.5 * (alpha_low_ + alpha_high_));
  }
  double search_by_Newton() {
    Eigen::VectorXd jac = jac_fun_(var_);
    Eigen::MatrixXd hes = hes_fun_(var_);
    Eigen::VectorXd dir = hes.ldlt().solve(-jac);
    double alpha = alpha_high_;
    double alpha_delta = epsilon_ + alpha;
    int iter_num = 0;
    while (std::abs(alpha_delta) > epsilon_ && std::abs(dir.dot(jac_fun_(var_ + alpha * dir))) > epsilon_ && iter_num < max_iter_num_) {
      iter_num++;
      alpha_delta = dir.dot(jac_fun_(var_ + alpha * dir)) / (dir.transpose() * hes_fun_(var_ + alpha * dir) * dir);
      alpha -= alpha_delta;
    }
    return std::max(eps_, alpha);
  }
  double search_by_secant() {
    Eigen::VectorXd jac = jac_fun_(var_);
    double alpha = alpha_high_;
    double alpha_pre = alpha;
    double alpha_delta = epsilon_ + alpha;
    int iter_num = 0;
    while (std::abs(alpha_delta) > epsilon_ && std::abs(dir_.dot(jac_fun_(var_ + alpha * dir_))) > epsilon_ && iter_num < max_iter_num_) {
      iter_num++;
      alpha_delta = (alpha == alpha_pre) ? 1.1 * epsilon_ : dir_.dot(jac_fun_(var_ + alpha * dir_)) * (alpha - alpha_pre) / (dir_.dot(jac_fun_(var_ + alpha * dir_)) - dir_.dot(jac_fun_(var_ + alpha_pre * dir_)));
      alpha_pre = alpha;
      alpha -= alpha_delta;
    }
    return std::max(eps_, alpha);
  }
  double search_by_2pt_quad_interpo() {
    Eigen::VectorXd jac = jac_fun_(var_);
    double alpha = alpha_high_;
    double alpha_pre = alpha + 1.0;
    double alpha_delta = epsilon_ + alpha;
    int iter_num = 0;
    while (std::abs(alpha_delta) > epsilon_ && std::abs(dir_.dot(jac_fun_(var_ + alpha * dir_))) > epsilon_ && iter_num < max_iter_num_) {
      iter_num++;
      alpha_delta = (alpha == alpha_pre) ? epsilon_ : 0.5 * dir_.dot(jac_fun_(var_ + alpha * dir_)) * (alpha - alpha_pre) / (dir_.dot(jac_fun_(var_ + alpha * dir_)) - (obj_fun_(var_ + alpha * dir_) - obj_fun_(var_ + alpha_pre * dir_)) / (alpha - alpha_pre));
      alpha_pre = alpha;
      alpha -= alpha_delta;
    }
    return std::max(eps_, alpha);
  }
  double search_by_3pt_quad_interpo() {
    double alpha_mid = 0.5 * (alpha_high_ + alpha_low_);
    return std::max(eps_, 0.5 * (alpha_mid + alpha_low_) + 0.5 * (obj_fun_(var_ + alpha_low_ * dir_) - obj_fun_(var_ + alpha_mid * dir_)) * (alpha_mid - alpha_high_) * (alpha_high_ - alpha_low_) / ((alpha_mid - alpha_high_) * obj_fun_(var_ + alpha_low_ * dir_) + (alpha_high_ - alpha_low_) * obj_fun_(var_ + alpha_mid * dir_) + (alpha_low_ - alpha_mid) * obj_fun_(var_ + alpha_high_ * dir_)));
  }
  double search_by_2pt_cubic_interpo() {
    int iter_num = 0;
    double alpha = alpha_high_;
    while (std::abs(dir_.dot(jac_fun_(var_ + alpha * dir_))) > epsilon_ && iter_num < max_iter_num_) {
      iter_num++;
      double obj_low  = obj_fun_(var_ + alpha_low_  * dir_);
      double obj_high = obj_fun_(var_ + alpha_high_ * dir_);
      double jac_low  = dir_.dot(jac_fun_(var_ + alpha_low_ * dir_));
      double jac_high = dir_.dot(jac_fun_(var_ + alpha_high_ * dir_));
      double jac_low_high = jac_low * jac_high;
      double omega = 3.0 * (obj_high - obj_low) / (alpha_high_ - alpha_low_) - jac_low_high;
      double eta = std::sqrt(omega * omega - jac_low_high);
      double alpha = alpha_low_ + (eta - jac_low - omega) * (alpha_high_ - alpha_low_) / (2 * eta - jac_low + jac_high);
      if (dir_.dot(jac_fun_(var_ + alpha * dir_)) > 0.0) {
        alpha_high_ = alpha;
      } else {
        alpha_low_  = alpha;
      }
    }
    return std::max(eps_, alpha);
  }
  protected:
  double epsilon_    = 1e-4;
  Search_Type search_type_ = Search_Type::Bisection; 
};
class InExact_Line_Searcher : public Base_Line_Searcher {
  public:
  enum class Search_Type {
    Armijo_goldstein,
    Wolfe_powell,
    Strong_wolfe_powell
  };
  InExact_Line_Searcher(Eigen::VectorXd _var, \
                        Eigen::VectorXd _dir, \
                        ObjFun _obj_fun = nullptr, \
                        JacFun _jac_fun = nullptr, \
                        HesFun _hes_fun = nullptr, \
                        double _alpha_init = 0.5, \
                        double _stepsize   = 1.0, \
                        double _multiplier = 1.0, \
                        int _max_iter_num  = 1000, \
                        double _rho        = 0.1, \
                        double _sigma      = 0.4, \
                        Search_Type _search_type = Search_Type::Armijo_goldstein) :
                        Base_Line_Searcher(_var, _dir, _obj_fun, _jac_fun, _hes_fun, _alpha_init, _stepsize, _multiplier, _max_iter_num),
                        rho_(_rho),
                        sigma_(_sigma),
                        search_type_(_search_type) {}
  double search() override {
    switch (search_type_){
    case Search_Type::Armijo_goldstein:
      return search_by_armijo_goldstein();
    case Search_Type::Wolfe_powell:
      return search_by_wolfe_powell();
    case Search_Type::Strong_wolfe_powell:
      return search_by_strong_wolfe_powell();
    default:
      return 0.0;
    }
  }
  double search_by_armijo_goldstein() {
    double alpha = std::max(eps_, 0.5 * (alpha_low_ + alpha_high_));
    double obj_zero  = obj_fun_(var_);
    double jac_zero  = dir_.dot(jac_fun_(var_));
    double obj_alpha = obj_fun_(var_ + alpha * dir_);
    int iter_num = 0;
    while ((obj_alpha < obj_zero + rho_ * jac_zero * alpha || obj_alpha > obj_zero + (1.0 - rho_) * jac_zero * alpha) && iter_num < max_iter_num_){
      iter_num++;
      obj_alpha = obj_fun_(var_ + alpha * dir_);
      if (obj_alpha < obj_zero + rho_ * jac_zero * alpha){
        alpha_low_  = alpha;
      } else if (obj_alpha > obj_zero + (1.0 - rho_) * jac_zero * alpha){
        alpha_high_ = alpha;
      }
      alpha = std::max(eps_, 0.5 * (alpha_low_ + alpha_high_));
    }
    return alpha;
  }
  double search_by_wolfe_powell() {
    double alpha = 0.5 * (alpha_low_ + alpha_high_);
    double obj_zero  = obj_fun_(var_);
    double jac_zero  = dir_.dot(jac_fun_(var_));
    double obj_alpha = obj_fun_(var_ + alpha * dir_);
    double jac_alpha = dir_.dot(jac_fun_(var_ + alpha * dir_));
    int iter_num = 0;
    while ((obj_alpha > obj_zero + rho_ * alpha * jac_zero || jac_alpha < sigma_ * jac_zero) && iter_num < max_iter_num_){
      iter_num++;
      if (obj_alpha > obj_zero + rho_ * jac_zero * alpha){
        alpha_high_  = alpha;
      } else if (jac_alpha < sigma_ * jac_zero){
        alpha_low_ = alpha;
      }
      alpha = 0.5 * (alpha_low_ + alpha_high_);
      obj_alpha = obj_fun_(var_ + alpha * dir_);
      jac_alpha = dir_.dot(jac_fun_(var_ + alpha * dir_));
    }
    return std::max(eps_, alpha);
  }
  double search_by_strong_wolfe_powell() {
    double alpha = 0.5 * (alpha_low_ + alpha_high_);
    double obj_zero  = obj_fun_(var_);
    double jac_zero  = dir_.dot(jac_fun_(var_));
    double obj_alpha = obj_fun_(var_ + alpha * dir_);
    double jac_alpha = dir_.dot(jac_fun_(var_ + alpha * dir_));
    int iter_num = 0;
    while ((obj_alpha > obj_zero + rho_ * jac_zero * alpha || std::abs(jac_alpha) > sigma_ * std::abs(jac_zero)) && iter_num < max_iter_num_){
      iter_num++;
      if (obj_alpha > obj_zero + rho_ * jac_zero * alpha){
        alpha_high_  = alpha;
      } else if (std::abs(jac_alpha) > sigma_ * std::abs(jac_zero)){
        alpha_low_ = alpha;
      }
      alpha = 0.5 * (alpha_low_ + alpha_high_);
      obj_alpha = obj_fun_(var_ + alpha * dir_);
      jac_alpha = dir_.dot(jac_fun_(var_ + alpha * dir_));
    }
    return std::max(eps_, alpha);
  }
  protected:
  double rho_        = 0.1;
  double sigma_      = 0.4;
  Search_Type search_type_ = Search_Type::Armijo_goldstein;
};
} //namespace opt
#endif //LINE_SEARCHER_HPP