/*!
 * Copyright 2017 by Contributors
 * \file count_descendant.cc
 * \brief Count number of descendants for each AST node
 */
#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(count_descendant);

static int count(ASTNode* node) {
  if (dynamic_cast<CodeFolderNode*>(node)) {
    node->num_descendant_ast_node = 0;
    return 0;  // descendants of CodeFolderNode are exempt from
               // ASTBuilder::BreakUpLargeTranslationUnits
  }
  int accum = 0;
  for (ASTNode* child : node->children) {
    accum += count(child) + 1;
  }
  node->num_descendant_ast_node = accum;
  return accum;
}

void ASTBuilder::CountDescendant() {
  count(this->main_node);
}

}  // namespace compiler
}  // namespace treelite
