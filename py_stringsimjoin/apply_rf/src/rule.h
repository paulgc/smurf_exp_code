#pragma once
#include "predicate.h"

#include <vector>

using std::vector;

class Rule {                                                               
  public:                                                                       
    vector<Predicate> predicates;                                                                               
 
    Rule();
    Rule(vector<Predicate> predicates);
    ~Rule();

    void add_predicate(Predicate predicate);                                                               
};     
