/*!
 * Copyright (c) 2023 by Contributors
 * \file task_type.h
 * \brief An enum type to indicate the learning task.
 * \author Hyunsu Cho
 */

#ifndef TREELITE_TASK_TYPE_H_
#define TREELITE_TASK_TYPE_H_

#include <treelite/logging.h>

#include <cstdint>
#include <string>

namespace treelite {

/*!
 * \brief Enum type representing the task type.
 *
 * The task type places constraints on the parameters of TaskParam. See the docstring for each
 * enum constants for more details.
 */
enum class TaskType : std::uint8_t {
  /*!
   * \brief Binary classifier
   */
  kBinaryClf = 0,
  /*!
   * \brief Regressor
   */
  kRegressor = 1,
  /*!
   * \brief Multi-class classifier
   */
  kMultiClf = 2,
  /*!
   * \brief Learning-to-rank
   */
  kLearningToRank = 3,
  /*!
   * \brief Isolation forest
   */
  kIsolationForest = 4
};

inline std::string TaskTypeToString(TaskType type) {
  switch (type) {
  case TaskType::kBinaryClf:
    return "kBinaryClf";
  case TaskType::kRegressor:
    return "kRegressor";
  case TaskType::kMultiClf:
    return "kMultiClf";
  case TaskType::kLearningToRank:
    return "kLearningToRank";
  case TaskType::kIsolationForest:
    return "kIsolationForest";
  default:
    return "";
  }
}

inline TaskType StringToTaskType(std::string const& str) {
  if (str == "kBinaryClf") {
    return TaskType::kBinaryClf;
  } else if (str == "kRegressor") {
    return TaskType::kRegressor;
  } else if (str == "kMultiClf") {
    return TaskType::kMultiClf;
  } else if (str == "kLearningToRank") {
    return TaskType::kLearningToRank;
  } else if (str == "kIsolationForest") {
    return TaskType::kIsolationForest;
  } else {
    TREELITE_LOG(FATAL) << "Unknown task type: " << str;
    return TaskType::kBinaryClf;  // to avoid compiler warning
  }
}

}  // namespace treelite

#endif  // TREELITE_TASK_TYPE_H_
