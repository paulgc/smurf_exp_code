#include "optimizer.h"

Optimizer::Optimizer(std::vector<Tree>& t, std::map<std::string, std::vector<bool> >& cov) {
  int i;
  for (std::map<std::string, std::vector<bool> >::iterator it=cov.begin(); it!=cov.end(); ++it) {
    for (i=0; i < (it->second).size(); i++) {
      if ((it->second)[i]) {
        coverage[it->first].flip(i);
      }
    }
  }
  trees = t;
}

Optimizer::Optimizer() {}

Optimizer::~Optimizer() {}

void Optimizer::generate_local_optimal_plans() {
  std::vector<std::vector<Node*> > local_optimal_plans(trees.size());
  int i, j;
  for (i=0; i<trees.size(); i++) {
    for (j=0; j<trees[i].rules.size(); j++) {
      Node *ptr;
      generate_optimal_plan_for_rule(trees[i].rules[j], ptr);
      local_optimal_plans[i].push_back(ptr);
    }
  }
}

void Optimizer::generate_optimal_plan_for_rule(Rule& rule, Node* ptr) {
  std::string node_type = "ROOT";
//  Node root(node_type), new_node, curr_node;
  *ptr = Node(node_type);                                                       
/*  curr_node = root;                                               

  bool join_pred = true;
  std::vector<int> optimal_seq;                                                 
  for (std::vector<int>::iterator it = optimal_seq.begin(); it != optimal_seq.end(); ++it) {                                     
    std::vector<Predicatecpp> preds; 
    preds.push_back(rule.predicates[*it]);
    node_type = "FILTER";                                                                              
    if (join_pred) {
      node_type = "JOIN";
      join_pred = false;
    }
    new_node = Node(preds, node_type); 
    curr_node.add_child(new_node);                               
    curr_node = new_node;                                        
  }
  node_type = "OUTPUT";
  curr_node.add_child(Node(node_type));*/          
//  return &root;
}
