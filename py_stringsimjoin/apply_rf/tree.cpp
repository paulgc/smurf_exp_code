#include "tree.h"

Tree::Tree(std::vector<Rule>& rls) {
  rules = rls;
}

Tree::Tree() {}

Tree::~Tree() {}

void Tree::set_tree_id(int tid) {
  tree_id = tid;
} 
