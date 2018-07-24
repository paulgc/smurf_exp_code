#ifndef TREE_H                                                                  
#define TREE_H   

#include "rule.h"
#include <vector>

class Tree {
  public:
    std::vector<Rule> rules;
    int tree_id;    

    Tree();
    Tree(std::vector<Rule>&);
    ~Tree();
    void set_tree_id(int);
};

#endif
