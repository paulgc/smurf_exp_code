#pragma once
#include "utils.h"

class Predicate {
  public:
    similarity_measure_type sim_type;
    comparison_operator comp_op;
    float threshold;
    
    Predicate(similarity_measure_type sim_type, comparison_operator comp_op, float threshold);
    ~Predicate();
};
