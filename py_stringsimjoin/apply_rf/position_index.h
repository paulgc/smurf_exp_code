#include <map>
#include <utility>
#include <vector>
#include <math.h>

class PositionIndex {
  public:
    std::map<int, std::vector< std::pair<int, int> > > index;                       
    int min_len, max_len;                                          
    std::vector<int> size_vector;                                       
    double threshold;

    PositionIndex();
    PositionIndex(std::map<int, std::vector< std::pair<int, int> > >&, std::vector<int>&, int&, int&, double&);
    ~PositionIndex();
    void set_fields(std::map<int, std::vector< std::pair<int, int> > >&, std::vector<int>&, int&, int&, double&);  
};
