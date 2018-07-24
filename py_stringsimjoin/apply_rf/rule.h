#ifndef RULE_H
#define RULE_H

#include "predicatecpp.h"
#include <string>
#include <vector>

class Rule {
  public:
    std::vector<Predicatecpp> predicates;
    
    Rule();
    Rule(std::vector<Predicatecpp>&);
    ~Rule();
};

#endif
