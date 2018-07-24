#include "node.h"
#include "tree.h"

#include <string>
#include <vector>
#include <bitset>
#include <map>

#define MAX_SIZE 110000                                                               

class Optimizer {
  public:
    std::map<std::string, std::bitset<MAX_SIZE> > coverage;
    std::vector<Tree> trees;
    
    Optimizer();
    Optimizer(std::vector<Tree>&, std::map<std::string, std::vector<bool> >&);
    ~Optimizer();

    void generate_local_optimal_plans();
    int generate_optimal_plan_for_rule(Rule&);
};
