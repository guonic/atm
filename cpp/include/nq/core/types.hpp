#pragma once

#include <cstdint>
#include <string>

namespace nq {
namespace core {

// 基础类型定义
using Timestamp = int64_t;
using Price = double;
using Volume = int64_t;

// 订单类型
enum class OrderType {
    Market,
    Limit,
    Stop,
    StopLimit
};

// 订单方向
enum class OrderSide {
    Buy,
    Sell
};

// 订单状态
enum class OrderStatus {
    Pending,
    Submitted,
    Filled,
    PartiallyFilled,
    Cancelled,
    Rejected
};

} // namespace core
} // namespace nq

