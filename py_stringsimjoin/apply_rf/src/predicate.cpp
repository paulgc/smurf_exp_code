
#include "predicate.h"

Predicate::Predicate(similarity_measure_type sim_type, 
  comparison_operator comp_op, double threshold): sim_type(sim_type), comp_op(comp_op), threshold(threshold) { 
}

Predicate::~Predicate() { } 

