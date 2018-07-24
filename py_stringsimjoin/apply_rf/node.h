#include "predicatecpp.h"
#include <string>
#include <vector>

class Node {
  public:
    std::vector<Predicatecpp> predicates;
    std::string node_type;                                       
    std::vector<Node> children;
    int tree_id, rule_id;
 
    Node();
    Node(std::vector<Predicatecpp>&, std::string&, std::vector<Node>&);
    Node(std::vector<Predicatecpp>&, std::string&);         
    Node(std::string&);         
    ~Node();

    void add_child(Node);
    void set_node_type(std::string&);
    void remove_child(Node&);
    void set_tree_id(int);
    void set_rule_id(int);
};
