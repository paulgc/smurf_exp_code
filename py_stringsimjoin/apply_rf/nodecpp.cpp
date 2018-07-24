#include "nodecpp.h"

Nodecpp::Nodecpp(std::vector<Predicatecpp>& preds, std::string& ntype, std::vector<Nodecpp*>& child_nodes) {
  predicates = preds;
  node_type = ntype;
  children = child_nodes;
}

Nodecpp::Nodecpp(std::string& ntype) {
  node_type = ntype;
}

Nodecpp::Nodecpp(std::vector<Predicatecpp>& preds, std::string& ntype) {
  predicates = preds;                                                           
  node_type = ntype; 
}

Nodecpp::Nodecpp() {}

Nodecpp::~Nodecpp() {}

void Nodecpp::add_child(Nodecpp* n) {
  children.push_back(n);
} 
