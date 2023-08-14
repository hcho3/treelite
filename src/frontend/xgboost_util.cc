/*!
 * Copyright (c) 2020-2021 by Contributors
 * \file xgboost_util.cc
 * \brief Common utilities for XGBoost frontends
 * \author Hyunsu Cho
 */

#include <treelite/logging.h>
#include <treelite/tree.h>

#include <cstring>

#include "xgboost/xgboost.h"

namespace treelite::details::xgboost {

const std::vector<std::string> exponential_objectives{
    "count:poisson", "reg:gamma", "reg:tweedie", "survival:cox", "survival:aft"};

// Get correct prediction transform function, depending on objective function
std::string GetPredTransform(std::string const& objective_name) {
  if (objective_name == "multi:softmax") {
    return "max_index";
  } else if (objective_name == "multi:softprob") {
    return "softmax";
  } else if (objective_name == "reg:logistic" || objective_name == "binary:logistic") {
    return "sigmoid";
  } else if (std::find(
                 exponential_objectives.cbegin(), exponential_objectives.cend(), objective_name)
             != exponential_objectives.cend()) {
    return "exponential";
  } else if (objective_name == "binary:hinge") {
    return "hinge";
  } else if (objective_name == "reg:squarederror" || objective_name == "reg:linear"
             || objective_name == "reg:squaredlogerror" || objective_name == "reg:pseudohubererror"
             || objective_name == "binary:logitraw" || objective_name == "rank:pairwise"
             || objective_name == "rank:ndcg" || objective_name == "rank:map") {
    return "identity";
  } else {
    TREELITE_LOG(FATAL) << "Unrecognized XGBoost objective: " << objective_name;
    return "";
  }
}

// Transform a base score from probability into margin score
double TransformBaseScoreToMargin(std::string const& pred_transform, double base_score) {
  if (pred_transform == "sigmoid") {
    return ProbToMargin::Sigmoid(base_score);
  } else if (pred_transform == "exponential") {
    return ProbToMargin::Exponential(base_score);
  } else {
    return base_score;
  }
}

}  // namespace treelite::details::xgboost
