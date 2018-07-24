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
    PositionIndex(std::vector< std::vector<int> >&, double&);
    ~PositionIndex();  
};
