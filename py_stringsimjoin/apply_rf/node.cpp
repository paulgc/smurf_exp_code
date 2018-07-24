#include "node.h"

Node::Node(std::vector<Predicatecpp>& preds, std::string& ntype, std::vector<Node>& child_nodes) {
  predicates = preds;
  node_type = ntype;
  children = child_nodes;
}

Node::Node(std::string& ntype) {
  node_type = ntype;
}

Node::Node(std::vector<Predicatecpp>& preds, std::string& ntype) {
  predicates = preds;                                                           
  node_type = ntype; 
}

Node::Node() {}

Node::~Node() {}

void Node::add_child(Node n) {
  children.push_back(n);
}

void Node::set_node_type(std::string& ntype) {
  node_type = ntype;
}

void Node::set_tree_id(int tid) {
  tree_id = tid;
}

void Node::set_rule_id(int rid) {
  rule_id = rid;
}

void Node::remove_child(Node& n) {
  int i = 0;
  while (i < children.size()) {
    if (children[i].predicates[0].pred_name.compare(n.predicates[0].pred_name) == 0) {
      break;
    }
    i += 1;
  }
  children.erase(children.begin() + i);
} 
