#include "predicatecpp.h"

Predicatecpp::Predicatecpp(std::string& p_name, std::string& f_name, std::string& sim_type, std::string& tok_type, std::string& cmp, double& t) {
  pred_name = p_name;
  feat_name = f_name;
  sim_measure_type = sim_type;
  tokenizer_type = tok_type;
  comp_op = cmp;
  threshold = t;
  cost = 0.0;
  is_tok_sim_measure = true;
  if (tokenizer_type.compare("none") == 0) {
    is_tok_sim_measure = false;
  }
}

Predicatecpp::Predicatecpp() {}

Predicatecpp::~Predicatecpp() {}

void Predicatecpp::set_cost(double& c) {
  cost = c;
}

bool Predicatecpp::is_join_predicate() {
  if (sim_measure_type.compare("JACCARD") == 0 ||
      sim_measure_type.compare("COSINE") == 0 ||
      sim_measure_type.compare("DICE") == 0 ||
      sim_measure_type.compare("OVERLAP") == 0 ||
      sim_measure_type.compare("OVERLAP_COEFFICIENT") == 0) {
    if (comp_op.compare(">") == 0 || comp_op.compare(">=") == 0 || 
        comp_op.compare("=") == 0) {
      return true;
    }
  }
  else if(sim_measure_type.compare("EDIT_DISTANCE") == 0) {
    if (comp_op.compare("<") == 0 || comp_op.compare("<=") == 0) {
      return true;
    }
  }
  return false;
} 
