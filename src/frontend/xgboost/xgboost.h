/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost.h
 * \brief Helper functions for loading XGBoost models
 * \author William Hicks
 */
#ifndef SRC_FRONTEND_XGBOOST_XGBOOST_H_
#define SRC_FRONTEND_XGBOOST_XGBOOST_H_

#include <treelite/contiguous_array.h>

#include <cmath>
#include <string>
#include <vector>

namespace treelite {

struct ModelParam;  // forward declaration

namespace details {
namespace xgboost {

struct ProbToMargin {
  static double Sigmoid(double base_score) {
    return -std::log(1.0 / base_score - 1.0);
  }
  static double Exponential(double base_score) {
    return std::log(base_score);
  }
};

extern const std::vector<std::string> exponential_objectives;

// Get correct prediction transform function, depending on objective function
std::string GetPredTransform(std::string const& objective_name);

// Transform a base score from probability into margin score
double TransformBaseScoreToMargin(std::string const& pred_transform, double base_score);

enum FeatureType { kNumerical = 0, kCategorical = 1 };

}  // namespace xgboost
}  // namespace details
}  // namespace treelite
#endif  // SRC_FRONTEND_XGBOOST_XGBOOST_H_
