#ifndef TYPE_DEFINE_HPP
#define TYPE_DEFINE_HPP

#include <mutex>
#include <queue>
#include <algorithm>
#include <functional>
#include <Eigen/Dense>

namespace opt {
typedef std::function<double(Eigen::VectorXd)> ObjFun;
typedef std::function<Eigen::VectorXd(Eigen::VectorXd)> JacFun;
typedef std::function<Eigen::MatrixXd(Eigen::VectorXd)> HesFun;
template<typename DataType>
class FixedQueue : public std::deque<DataType> {
  public:
  FixedQueue(std::size_t _max_size=10) : max_size_(_max_size) {}
  void push(const DataType &_data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (this->size() >= max_size_){
      this->pop_front();
    }
    this->push_back(_data);
  }
  protected:
  std::size_t max_size_;
  std::mutex mutex_;
};
}  // opt
#endif  //TYPE_DEFINE_HPP