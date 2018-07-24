#ifndef PREDICATE_H
#define PREDICATE_H

#include <string>

class Predicatecpp {
  public:
    std::string pred_name, feat_name, sim_measure_type, tokenizer_type, comp_op;                      
    double threshold, cost;                                       
    bool is_tok_sim_measure;

    Predicatecpp();
    Predicatecpp(std::string&, std::string&, std::string&, std::string&, std::string&, double&);
    ~Predicatecpp();
 
    void set_cost(double&); 
    bool is_join_predicate();
};

#endif
