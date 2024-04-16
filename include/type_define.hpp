#ifndef TYPE_DEFINE_HPP
#define TYPE_DEFINE_HPP

#include <Eigen/Dense>
#include <functional>

namespace opt {
typedef std::function<double(Eigen::VectorXd)> ObjFun;
typedef std::function<Eigen::VectorXd(Eigen::VectorXd)> JacFun;
typedef std::function<Eigen::MatrixXd(Eigen::VectorXd)> HesFun;
}  // opt
#endif  //TYPE_DEFINE_HPP