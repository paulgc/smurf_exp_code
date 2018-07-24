#include "predicatecpp.h"
#include <string>
#include <vector>

class Nodecpp {
  public:
    std::vector<Predicatecpp> predicates;
    std::string node_type;                                       
    std::vector<Nodecpp*> children;
    
    Nodecpp();
    Nodecpp(std::vector<Predicatecpp>&, std::string&, std::vector<Nodecpp*>&);
    Nodecpp(std::vector<Predicatecpp>&, std::string&);         
    Nodecpp(std::string&);         
    ~Nodecpp();

    void add_child(Nodecpp*);
};
