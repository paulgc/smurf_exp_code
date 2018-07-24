#include "rule.h"

Rule::Rule(std::vector<Predicatecpp>& preds) {
  predicates = preds;
}

Rule::Rule() {}

Rule::~Rule() {} 
